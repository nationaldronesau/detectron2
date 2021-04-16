from detectron2 import model_zoo

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode

import os, cv2, json
import numpy as np
from progress.bar import Bar
import skimage.draw
import pycocotools
from PIL import Image, ImageDraw

from detectron2.utils.visualizer import Visualizer

import pickle
import gzip

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
            
            #check bounding boxes are healthy
#             test_mask = pycocotools.mask.decode(poly)
#                 test_mask = np.zeros([height, width], dtype=np.uint8)
#                 rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
#                 test_mask[rr, cc] = 1
#                 mask_img = Image.fromarray(test_mask.astype(np.bool)).convert('RGB')
#                 draw = ImageDraw.Draw(mask_img)
#                 draw.rectangle(bbox, fill=None, outline='red', width=3)
#                 mask_img.save(filename.split('/')[-1])

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

output_dir = "../Results/Results_detectron_corrosion"
weights_dir = "./output/Corrosion_20210329T0501"
dataset_dir = "../../Data/Corrosion"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_0024999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (corrosion)

predictor = DefaultPredictor(cfg)

# TEST INFERENCE
dataset_dicts = get_darwin_dataset(dataset_dir, 'val')

print(dataset_dicts[0])

bar = Bar('Performing inference', max=len(dataset_dicts))

results = []

for d in dataset_dicts[0:1]:    
    im = cv2.imread(d["file_name"])
    # im = Image.open(d["file_name"])
    outputs = predictor(im)  #format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print(outputs['instances'])
    results.append(outputs)
    v = Visualizer(im, scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    Image.fromarray(out.get_image()[:, :, ::-1]).save(os.path.join(output_dir, d['file_name'].split('/')[-1].split('.')[0] + '.jpg'))
    bar.next()

bar.finish()

pickle.dump(dataset_dicts, open( "dataset.p", "wb" ), protocol=4)

# with open('results_detectron_101_validation.json', 'w') as outfile:
#     json.dump(results, outfile, indent=4)

# pickle.dump( results, open( "results.p", "wb" ), protocol=4)

# with gzip.GzipFile('results.pgz', 'w') as f:
#     pickle.dump(results, f)