from detectron2.structures import BoxMode
import os, json
import cv2
import skimage.draw
import pycocotools
from PIL import Image
from progress.bar import Bar
import numpy as np


# map points outside image to edges
def fix_polygon(pol, width, height):
    """
    This fixes polygons so there are no points outside the edges of the image
    Arguments:
        pol: a polygon (list)
        width, height: of the image
    NOTE: this corrects for a problem with Darwin it may become unnecessary in the future
    Returns: a polygon (list)
    """
    
    for pt in pol:
        if pt['x'] >= width: pt['x'] = width - 1
        if pt['x'] < 0: pt['x'] = 0
        if pt['y'] >= height: pt['y'] = height - 1
        if pt['y'] < 0: pt['y'] = 0
        
    return pol

def convert_to_rle(annotation, width, height):
    """Convert complex polygons to COCO RLE format.
    Arguments:
        annotation: a dictionary for an individual annotation in Darwin's format
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
    """Convert Darwin dataset to Detectron's format
    Arguments:
        img_dir: the directory of the dataset. Must contain three subdirectories: images, train and val
                 the val and train directories must contain val.json and train.json respectively in Darwin format
                 the images directory must contain the images for the training and validation sets
    Detectron format: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
    Darwin format: https://docs.v7labs.com/reference/darwin-json#:~:text=Darwin%20JSON%20is%20one%20of,in%20the%20Darwin%20JSON%20format.
    Returns: a list of dictionaries, each corresponding to an image in the dataset, in Detectron format
    """

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