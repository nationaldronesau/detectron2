import os, grpc
import numpy as np
import image_slicer
from PIL import Image
import io
import requests
import json
import logging
import cv2
from simplification.cutil import simplify_coords
from skimage.measure import find_contours

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


def load_model():
    """
    Load pre-trained Detectron2 model

    @author Dinis Gokaydin <d.gokaydin@nationaldrones.com>
    """

    global predictor

    # CHANGE THIS AFTER TESTING
    weights_dir = "./output/Corrosion_20210329T0501"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_0024999.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (corrosion)
    cfg.MODEL.DEVICE='cpu'

    predictor = DefaultPredictor(cfg)


def perform_inference_url(image_url):
    """
    Wrapper function to call perform_inference() with a url rather than a binary string

    @author Dinis Gokaydin <d.gokaydin@nationaldrones.com>
    """
    inf_img, global_mask, global_contours = perform_inference(requests.get(image_url).content)
    return inf_img, global_mask, global_contours


def perform_inference(image_bytes):
    """
    Main inference function for Detectron2.
    Inputs:
        image_bytes: image in binary format
    Outputs:
        inf_img: the image with annotations overlayed
        global_mask: one binary mask with all the instances detected
        global_contours: a list of contours for all the instances. note that each instance can have multiple polygons
    
    If the image is larger than 1024 pixels in any dimension it is broken into 'tiles' which are sent for inference
    and then reassembled. Each tile is a maximum 1024x1024 pixels in size

    @author Dinis Gokaydin <d.gokaydin@nationaldrones.com>
    """
    
    # class_names = ['BG','Corrosion']
    # colour_dic = {'Corrosion':'orange'}

    if isinstance(image_bytes, str):
        im = cv2.imread(image_bytes) # UNTESTED (but should work)
    else:
        im = cv2.imdecode(np.asarray(bytearray(image_bytes), dtype="uint8"), flags=cv2.IMREAD_COLOR)

    cv2.imwrite('img.jpg', im)# this is necessary for image_slicer

    # get image shape
    im_size = im.shape

    # no need to split it if image is under 1024x1024
    if im_size[0] <= 1024 and im_size[1] <= 1024:

        # perform inference
        output = predictor(im)

        # generate image with polygons, bboxes and confidence levels
        v = Visualizer(im, scale=1)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        inf_img = Image.fromarray(out.get_image()[:, :, ::-1])
        
        # generate mask
        if len(output['instances']) != 0:
            instance_masks = output['instances'].pred_masks.cpu().numpy()
            global_mask = Image.fromarray(np.sum(instance_masks, axis=0, dtype=np.bool))
            global_contours = generate_contours(instance_masks)
        else:          
            global_mask = Image.new('RGB', inf_img.size, None)

    else:

        # slice image into tiles
        tiles = image_slicer.slice('img.jpg', col=int(np.ceil(im_size[0]/1024)), row=int(np.ceil(im_size[1]/1024)), save=False)

        # results for each tile
        results = []

        for tile in tiles: 

            # convert to cv2 format
            tile_img = cv2.cvtColor(np.array(tile.image), cv2.COLOR_RGB2BGR)

            # perform inference
            output = predictor(tile_img)

            results.append(output)

            if len(output['instances']) != 0:

                instance_masks = output['instances'].pred_masks.cpu().numpy()

                contours = generate_contours(instance_masks)

                # generate image with polygons, bboxes and confidence levels
                v = Visualizer(tile_img, scale=1)
                out = v.draw_instance_predictions(output["instances"].to("cpu"))
                inf_img = Image.fromarray(out.get_image()[:, :, ::-1])

                tile.image = inf_img
                tile.contours = contours

            else:

                tile.contours = []


        inf_img = image_slicer.join(tiles)

        global_mask, global_contours = create_global_mask(tiles, results)

    # for testing  
    # global_mask.save('mask.jpg')
    # inf_img.save('inf_img.jpg')

    return inf_img, global_mask, global_contours


def generate_contours(masks, simplify=True):

    """
    Generates contours around masks output by the model
    and also simplifies them using the Ramer-Douglas-Peucker algorithm
    Inputs:
        masks: a numpy array with the masks for each of the instances detected (stacked along first dimension)
    Outputs:
        all_contours: a list of polygons corresponding to each binary mask
    
    NOTE: While tensorflow stacks images along third dimension, pytorch stacks them along first dimension
    so this is NOT exactly the same function as the one used previously in TF-based detection-service

    @author Dinis Gokaydin <d.gokaydin@nationaldrones.com>
    """

    all_contours = []

    N = masks.shape[0]

    for i in range(N):

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        mask = masks[i,:, :]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        if simplify:
            for ind, contour in enumerate(contours):
                contours[ind] = simplify_coords(contour,5)

        all_contours.append(contours)

    return all_contours


def create_global_mask(tiles, results):
    """
    Stitch together the inference results for all the tiles
    Chenges coordinates of polygons in each tile to coordinates of original image.

    inputs:
        tiles: collection of tiles returned by image_slicer
        results: list with the inference results for each tile; results are in Detectron2 format 
                 (https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format)
    outputs:
        mask_img: stiched up image with all the annotations (effectively the original image only annotated)
        contours: list of contours for all the instances across all tiles

    @author Dinis Gokaydin <d.gokaydin@nationaldrones.com>
    """

    # this will be the merged mask
    mask_img = Image.new('RGB', image_slicer.get_combined_size(tiles), None)

    contours = []

    for index, tile in enumerate(tiles):

        # merge masks for polygons in each image and add to global mask
        instance_masks = results[index]['instances'].pred_masks.cpu().numpy()
        merged_mask = Image.fromarray(np.sum(instance_masks, axis=0, dtype=np.bool))
        mask_img.paste(merged_mask, tile.coords)

        # if there are no contours nothing to be done
        if tile.contours:

            # for each detected instance
            for ind_contour, instance in enumerate(tile.contours):

                # each instance mask may be made up of more than one patch
                for ind_patch, patch in enumerate(instance):

                    # flip local coordinates before adding tile coordinates
                    patch = np.array([[k, j] for j, k in patch])

                    # broadcasting
                    tile.contours[ind_contour][ind_patch] = (patch + tile.coords).tolist()

        contours = contours + tile.contours

    return mask_img, contours


def clean_annotations(annotations: list) -> list:
    """
    This will remove the nested list in a list for no reason.
    Sometimes the annotations would contain a list of numpy arrays. This also converts that array to a list.
    @author: Joe Mudryk <j.mudryk@nationaldrones.com>
    """
    cleaned_annotations = []
    if len(annotations) == 0:
        return cleaned_annotations
    for annotation in annotations:
        if type(annotation[0]) is list:
            cleaned_annotations.append(annotation[0])
        else:
            # convert any arrays in the list to list
            try:
                cleaned_annotations.append(annotation[0].tolist())
            except Exception as e:
                logging.error(f"Failed to convert the following annotation array to list: {annotation}")
                logging.error(e)
    return cleaned_annotations


if __name__ == '__main__':
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on single image.')
    parser.add_argument("image_path", metavar="<image_path>")
    parser.add_argument("server", metavar="<server>")
    args = parser.parse_args()

    select_server(args.server)

    perform_inference(args.image_path)
