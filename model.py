import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# print("OpenCV:", cv2.__version__)
# print("Torch:", torch.__version__)
# print("NumPy:", np.__version__)

img_path = "train/images/2_bmp.rf.620909ed186ccd7c289b1396cdd8b2f0.jpg"

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
print(img.shape)
plt.show()