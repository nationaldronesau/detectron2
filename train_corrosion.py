import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import pycocotools
import skimage.draw
from PIL import Image, ImageDraw
from progress.bar import Bar
import datetime

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper
import detectron2.utils.comm as comm
import torch
import time
import logging

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.structures import BoxMode

# map points outside image to edges
def fix_polygon(pol, width, height):
    
    for pt in pol:
        if pt['x'] >= width: pt['x'] = width - 1
        if pt['x'] < 0: pt['x'] = 0
        if pt['y'] >= height: pt['y'] = height - 1
        if pt['y'] < 0: pt['y'] = 0
        
    return pol


def convert_to_rle(annotation, width, height):
    """Convert polygons and complex polygons to COCO RLE format.
   Returns: an annotation in encrypted RLE format and a bounding box
    """

    # complex polygons have multiple "paths" (polygons)
    polygons = annotation['complex_polygon']['path']

    mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

    for ind_pol, pol in enumerate(polygons):

        pol = fix_polygon(pol, width, height)# not sure whether assignment is necessary here
        
        all_points_y = []; all_points_x = [];
        
        for pt in pol:

            all_points_y.append(pt['y'])
            all_points_x.append(pt['x'])

        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
        mask[rr, cc, ind_pol] = 1

    # once we sum all the polygons any even values are holes (this should allow for "ring" holes, but it is not tested)
    mask = ((np.sum(mask, axis=2)%2) == 1).astype(np.uint8)

    # Return mask, and array of class IDs of each instance
    return pycocotools.mask.encode(np.asarray(mask, order="F")), Image.fromarray(mask).getbbox()


def get_darwin_dataset(img_dir, train_val):

    json_file = os.path.join(img_dir, train_val, train_val + ".json")
    with open(json_file) as f:
        imgs = json.load(f)

    # imgs = imgs[0:10]# for testing

    dataset_dicts = []

    bar = Bar('Importing Dataset', max=len(imgs))

    for idx, img in enumerate(imgs):
        
        record = {}
        
        filename = os.path.join(img_dir,'images',img["image"]["original_filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = img["annotations"]
        objs = []

        for anno in annos:
            
            # only convert to RLE if the polygon is complex, otherwise leave as polygons
            # set cfg.INPUT.MASK_FORMAT = "bitmask" and it should handle polygons and RLE
            # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/detection_utils.html#annotations_to_instances
            if 'complex_polygon' in anno:
                poly, bbox = convert_to_rle(anno, width, height)      
            else:
                pol = anno['polygon']['path']
                pol = fix_polygon(pol, width, height)# not sure whether assignment is necessary here
                
                all_points_y = []; all_points_x = [];
                
                for pt in pol:
                    all_points_y.append(pt['y'])
                    all_points_x.append(pt['x'])
                
                # nesting is necessary here as segmentation format is list[list[float]]
                poly = [[item for pair in zip(all_points_x, all_points_y) for item in pair]]
                bbox = [min(all_points_x), min(all_points_y), max(all_points_x), max(all_points_y)]

            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": poly,
                "category_id": 0,# change in the future for more than one category
            }
            
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

        bar.next()

    bar.finish()

    return dataset_dicts


# from https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

def setup(args):

    #set the number of GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # REGISTER DATASETS
    dataset_directory = "/home/ndserv05/Documents/Data/Corrosion"

    # register training and validation datasets with detectron
    for d in ["train", "val"]:
        # get_darwin_dataset(dataset_directory, d)
        DatasetCatalog.register("corrosion_" + d, lambda d=d: get_darwin_dataset(dataset_directory, d))
        MetadataCatalog.get("corrosion_" + d).set(thing_classes=["Corrosion"])
        

    # number of epochs to train
    EPOCHS = 60

    NUM_GPU = 2

    # get size of train and val datasets
    TRAIN_SIZE = len(DatasetCatalog.get("corrosion_train"))
    VAL_SIZE = len(DatasetCatalog.get("corrosion_val"))

    # CONFIGURATION
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = "./output/" + "Corrosion_" + "{:%Y%m%dT%H%M}".format(datetime.datetime.now())
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = ("corrosion_train",)
    cfg.DATASETS.TEST = ()
    cfg.TEST.EVAL_PERIOD = 887 # eval period should be one epoch, which is the number of images in training set divided by num_gpu*IMS_PER_BATCH
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = int(TRAIN_SIZE/(NUM_GPU*cfg.SOLVER.IMS_PER_BATCH)*EPOCHS)  # one iteration is 4 images so one epoch is around 887 iterations. 
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (corrosion)

    print(cfg.SOLVER.MAX_ITER)

    return cfg

# TRAINER

# need to subclass in order to implement the build_evaluator() function
class myTrainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset):
#         # the dataset is *not* in COCO format but this is handled by the evaluator
#         return COCOEvaluator(dataset, ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                "corrosion_val",
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


def main(args):

    cfg = setup(args)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = myTrainer(cfg)
    # trainer.build_evaluator(cfg, "corrosion_val")# this is not necessary 
    trainer.build_hooks()
    trainer.resume_or_load(resume=False)
    
    return trainer.train()

if __name__ == '__main__':
    launch(
        main,
        num_gpus_per_machine=2,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=({},)
    )

# INFERENCE

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)

# # TEST INFERENCE
# dataset_dicts = get_darwin_dataset(dataset_directory, 'val')
# for d in dataset_dicts:    
#     im = cv2.imread(d["file_name"])
#     # im = Image.open(d["file_name"])
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     print(outputs)
#     v = Visualizer(im, metadata=corrosion_metadata,  scale=1)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     # cv2_imshow(out.get_image()[:, :, ::-1])
#     Image.fromarray(out.get_image()[:, :, ::-1]).save(str(d['image_id']) + '.jpg')


# # EVALUATE MODEL

# # In[9]:


# evaluator = COCOEvaluator("corrosion_val", ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR)
# val_loader = build_detection_test_loader(cfg, "corrosion_val")
# print(inference_on_dataset(trainer.model, val_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`
