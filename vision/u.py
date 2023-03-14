import detectron2
import matplotlib.pyplot as plt
import numpy as np
import os, json, cv2, random
import torch
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from tqdm import tqdm


def get_predictor(name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg), MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


faster_cnn = get_predictor("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
faster_cnn_101 = get_predictor("COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml")
retina = get_predictor("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
mask_cnn = get_predictor("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
x101 = get_predictor("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
panoptic = get_predictor("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
lvis = get_predictor("LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")


im = cv2.imread(f'vision/images/2.jpg')
im = cv2.resize(im, dsize=(521, 1156), interpolation=cv2.INTER_CUBIC)

model = retina

outputs = model[0](im)
v = Visualizer(im[:, :, ::-1], model[1], scale=0.5)

out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
out = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1])
out = v.draw_sem_seg(outputs["sem_seg"].to("cpu"))

out.get_image()

plt.imshow(out.get_image())

