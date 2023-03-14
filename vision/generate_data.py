import json

import imgaug.augmenters as iaa
import cv2
import numpy as np
from detectron2.structures import BoxMode
from matplotlib import pyplot as plt
from pathlib import Path
import time

from tqdm import tqdm


background_augmentations = iaa.Sequential([
    iaa.Affine(
        rotate=(-30, 30),
        scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
        shear=(-5, 5),
        cval=(0, 128)
    ),
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.4)),
    iaa.Resize(800)
])

rotation_augmentations = iaa.Sequential([
    iaa.Rotate((-180, 180), fit_output=True, cval=128),
    iaa.ShearX((-40, 40), fit_output=True, cval=128),
    iaa.ScaleY((0.2, 1), fit_output=True, cval=128),
    iaa.Resize({'longer-side': (130, 210), 'shorter-side': 'keep-aspect-ratio'}),
])

target_augmentations = iaa.Sequential([
    iaa.OneOf([
        iaa.Identity(),
        iaa.Crop(percent=(0, 0.05)),
        iaa.Crop(percent=(0, 0.2)),
    ]),
    iaa.Sometimes(0.6, iaa.GaussianBlur(sigma=(0.0, 3.0))),
    iaa.Sometimes(0.3, iaa.GammaContrast((0.5, 2.0), per_channel=True)),
    iaa.Sometimes(0.3, iaa.ChangeColorTemperature((3000, 12000))),
    iaa.Sometimes(0.3, iaa.AddToBrightness((-60, 30))),
    iaa.Sometimes(0.3, iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True)),
    iaa.Sometimes(0.1, iaa.imgcorruptlike.GaussianNoise(severity=1)),
    rotation_augmentations
])

final_augmentations = iaa.Sequential([
    iaa.Sometimes(0.4, iaa.GaussianBlur((0.0, 3.0))),
    iaa.Sometimes(0.3, iaa.SaltAndPepper()),
    iaa.Sometimes(0.3, iaa.imgcorruptlike.GaussianNoise(severity=(1, 2))),
    iaa.Sometimes(0.3, iaa.imgcorruptlike.ShotNoise(severity=(1, 2))),
    iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.4), per_channel=True)),
    iaa.Sometimes(0.3, iaa.AddToBrightness((-5, 5))),
])


def image_str_to_category_id(x):
    colour, value = x
    return int(colour) * 9 + int(value) + 1


def add_target_to_image(background, target_augmentations):
    target_path = np.random.choice(list(Path('vision/target_image').iterdir()))
    target = cv2.imread(str(target_path))
    target = target_augmentations(image=target)

    card_mask = target != 128
    card_mask = card_mask.any(2)

    paste_y = np.random.randint(background.shape[0] - target.shape[0])
    paste_x = np.random.randint(background.shape[1] - target.shape[1])

    background[paste_y:(paste_y + target.shape[0]), paste_x:(paste_x + target.shape[1])][card_mask] = target[card_mask]

    annotation = {
        'bbox': [paste_x, paste_y, target.shape[1], target.shape[0]],
        'bbox_mode': BoxMode.XYWH_ABS,
        'category_id': image_str_to_category_id(target_path.name.split('.')[0])
    }

    return background, annotation


def generate_image(background_augmentations, target_augmentations, final_augmentations):
    background_path = np.random.choice(list(Path('vision/background_image').iterdir()))
    background = cv2.imread(str(background_path))
    background = background_augmentations(image=background)

    annotations = []

    for _ in range(np.random.randint(1, 5)):
        background, annotation = add_target_to_image(background, target_augmentations)
        annotations.append(annotation)

    background = final_augmentations(image=background)

    return background, annotations


target_path = 'vision/dataset_2'


all_images = []
all_annotations = []

annotation_count = 0


for i in tqdm(range(30000)):
    image, annotations = generate_image(
        background_augmentations, target_augmentations, final_augmentations)

    file_name = f'{target_path}/{i}.png'
    assert cv2.imwrite(file_name, image)

    all_images.append({
        "id": i,
        "width": image.shape[1],
        "height": image.shape[0],
        "file_name": file_name,
    })

    for annotation in annotations:
        annotation['image_id'] = i
        annotation['id'] = annotation_count
        annotation_count += 1

    all_annotations.extend(annotations)


categories = []
for colour in range(4):
    for value in range(9):
        categories.append({
            'id': colour * 9 + value + 1,
            'name': str(colour) + '_' + str(value + 6)
        })


coco = {
    'images': all_images,
    'annotations': all_annotations,
    'categories': categories
}


with open(f'{target_path}/coco.json', 'w') as f:
    json.dump(coco, f)
