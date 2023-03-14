from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2

image = Image.open('vision/images/20230212_101725.jpg')

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)[0]

pred = cv2.imread('vision/images/20230212_101725.jpg')
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [int(i) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=3, lineType=cv2.LINE_8)


#cv2.imshow('imageRectangle', pred)
cv2.imwrite('vision/images/1.png', pred)
