from detectron2 import model_zoo

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import os, cv2, json
import numpy as np
from progress.bar import Bar
from PIL import Image, ImageDraw

from detectron2.utils.visualizer import Visualizer

import pickle
import gzip

from tools.darwin import *

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

# pickle.dump(dataset_dicts, open( "dataset.p", "wb" ), protocol=4)

# with open('results_detectron_101_validation.json', 'w') as outfile:
#     json.dump(results, outfile, indent=4)

# pickle.dump( results, open( "results.p", "wb" ), protocol=4)

# with gzip.GzipFile('results.pgz', 'w') as f:
#     pickle.dump(results, f)