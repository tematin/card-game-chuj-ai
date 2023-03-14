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

register_coco_instances("cards", {}, "vision/dataset/coco.json", "")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cards",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 30000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 36

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
predictor = DefaultPredictor(cfg)

predictor

im = cv2.imread(f'vision/images/3.jpg')

original_image = im
self = predictor

height, width = original_image.shape[:2]
image = self.aug.get_transform(original_image).apply_image(original_image)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

inputs = {"image": image, "height": height, "width": width}

self.model.backbone()

images = self.model.preprocess_image([inputs])

features = self.model.backbone(images.tensor)

self = self.model.proposal_generator


features = [features[f] for f in self.in_features]
anchors = self.anchor_generator(features)

pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
# Transpose the Hi*Wi*A dimension to the middle:
pred_objectness_logits = [
    # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
    score.permute(0, 2, 3, 1).flatten(1)
    for score in pred_objectness_logits
]
pred_anchor_deltas = [
    # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
    x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
    .permute(0, 3, 4, 1, 2)
    .flatten(1, -2)
    for x in pred_anchor_deltas
]

if self.training:
    assert gt_instances is not None, "RPN requires gt_instances in training!"
    gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
    losses = self.losses(
        anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
    )
else:
    losses = {}
proposals = self.predict_proposals(
    anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
)
return proposals, losses











from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead


