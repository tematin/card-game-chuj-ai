import detectron2
import matplotlib.pyplot as plt
import numpy as np
import os, json, cv2, random
import torch
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from torch import Tensor
from tqdm import tqdm
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

fpn = build_resnet_fpn_backbone(cfg, input_shape).to("cuda")
self = fpn

im = cv2.imread(f'vision/images/3.jpg')
im = cv2.resize(im, dsize=(int(im.shape[1] / 5), int(im.shape[0] / 5)),
                interpolation=cv2.INTER_CUBIC)
im = np.swapaxes(im, 0, 2)
im = im.reshape(1, *im.shape)
x = Tensor(im).to("cuda")


bottom_up_features = self.bottom_up(x)
results = []
prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
results.append(self.output_convs[0](prev_features))

# Reverse feature maps into top-down order (from low to high resolution)
for idx, (lateral_conv, output_conv) in enumerate(
        zip(self.lateral_convs, self.output_convs)
):
    # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
    # Therefore we loop over all modules but skip the first one
    if idx > 0:
        features = self.in_features[-idx - 1]
        features = bottom_up_features[features]
        top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
        lateral_features = lateral_conv(features)
        prev_features = lateral_features + top_down_features
        if self._fuse_type == "avg":
            prev_features /= 2
        results.insert(0, output_conv(prev_features))

if self.top_block is not None:
    if self.top_block.in_feature in bottom_up_features:
        top_block_in_feature = bottom_up_features[self.top_block.in_feature]
    else:
        top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
    results.extend(self.top_block(top_block_in_feature))
assert len(self._out_features) == len(results)
ret_val = {f: res for f, res in zip(self._out_features, results)}
