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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

def convert_to_rle(annotation, height, width):
    """Convert polygons and complex polygons to COCO RLE format.
   Returns: a dictionary with 
    """

    # complex polygons have multiple "paths" (polygons)
    if 'complex_polygon' in annotation:
        polygons = annotation['complex_polygon']['path']
    else:
        polygons = [annotation['polygon']['path']]

    mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

    for ind_pol, pol in enumerate(polygons):

        all_points_y = []; all_points_x = [];

        # not clear why this worked in TF...
        for pt in pol:
            if pt['x'] >= width: pt['x'] = width - 1
            if pt['x'] < 0: pt['x'] = 0
            if pt['y'] >= height: pt['y'] = height - 1
            if pt['y'] < 0: pt['y'] = 0

            all_points_y.append(pt['y'])
            all_points_x.append(pt['x'])

        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
        mask[rr, cc, ind_pol] = 1

    # once we sum all the polygons any even values are holes (this should allow for "ring" holes, but it is now tested)
    mask = ((np.sum(mask, axis=2)%2) == 1).astype(np.uint8)

    # Return mask, and array of class IDs of each instance
    return pycocotools.mask.encode(np.asarray(mask, order="F")), Image.fromarray(mask).getbbox()

def get_darwin_dataset(img_dir, train_val):

    json_file = os.path.join(img_dir, train_val, train_val + ".json")
    with open(json_file) as f:
        imgs = json.load(f)

    imgs = imgs[0:10]

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

            poly, bbox = convert_to_rle(anno, height, width)
            
            #check bounding boxes are healthy
            # test_mask = pycocotools.mask.decode(poly)
            # mask_img = Image.fromarray(test_mask.astype(np.bool)).convert('RGB')
            # draw = ImageDraw.Draw(mask_img)
            # draw.rectangle(bbox, fill=None, outline='red', width=3)
            # mask_img.save('img.jpg')

            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": poly,
                "category_id": 0,# change in the future for more than one category
            }
            
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

        # # check masks are healthy
        # test_mask = np.zeros([height, width, len(record["annotations"])], dtype=np.uint8)
        # for idx, obj in enumerate(record["annotations"]):
        #     # decode RLE for all objects, create global mask and save image
        #     test_mask[:,:,idx] = pycocotools.mask.decode(obj['segmentation'])        
        # Image.fromarray(np.sum(test_mask, axis=2).astype(np.bool)).save('masks/' + img["image"]["original_filename"])

        bar.next()

    bar.finish()

    return dataset_dicts

dataset_directory = "/home/ndserv05/Documents/Data/Corrosion"

# register training and validation datasets with detectron
for d in ["train", "val"]:
    # get_darwin_dataset(dataset_directory, d)
    DatasetCatalog.register("corrosion_" + d, lambda d=d: get_darwin_dataset(dataset_directory, d))
    MetadataCatalog.get("corrosion_" + d).set(thing_classes=["Corrosion"])

corrosion_metadata = MetadataCatalog.get("corrosion_val")

# print(corrosion_metadata)

# check annotations
# dataset_dicts = get_darwin_dataset(dataset_directory, 'val')
# for d in random.sample(dataset_dicts, 3):
#     img = Image.open(d["file_name"])
#     visualizer = Visualizer(img, metadata=corrosion_metadata, scale=1)
#     out = visualizer.draw_dataset_dict(d)
#     Image.fromarray(out.get_image()).save(str(d['image_id']) + '.jpg')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# print(cfg)

cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.DATASETS.TRAIN = ("corrosion_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (corrosion)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()