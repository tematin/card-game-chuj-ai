import time
from pathlib import Path

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

register_coco_instances("cards_train", {}, "vision/dataset_2/coco.json", "")
register_coco_instances("cards_valid", {}, "vision/valid/coco.json", "")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cards_train",)
cfg.DATASETS.TEST = ("cards_valid",)
cfg.DATALOADER.NUM_WORKERS = 1
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = []
cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 36

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()



meta = MetadataCatalog.get('cards_train')


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0001999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
predictor = DefaultPredictor(cfg)

files = list(Path('vision/valid').iterdir())

file = files[np.random.randint(len(files))]

im = cv2.imread(file.as_posix())

outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], meta, scale=0.5)

out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

plt.imshow(out.get_image())




