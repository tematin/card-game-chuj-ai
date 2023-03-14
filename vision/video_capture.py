from cv2 import VideoCapture
from matplotlib import pyplot as plt

cam = VideoCapture(0)
result, image = cam.read()

plt.imshow(image[:, :, ::-1])

