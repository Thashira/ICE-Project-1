import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from step7_model import model, DEVICE
from step6_dataset import FabricDataset

TEST_IMG_DIR = "test/images"
TEST_LABEL_DIR = "test/labels"

test_dataset = FabricDataset(TEST_IMG_DIR, TEST_LABEL_DIR)
model.eval()

with torch.no_grad():
    for idx in range(len(test_dataset)):
        img, target = test_dataset[idx]
        prediction = model([img.to(DEVICE)])[0]

        boxes = prediction["boxes"].cpu()
        scores = prediction["scores"].cpu()

        # Convert PyTorch tensor to HWC uint8, contiguous
        img_disp = img.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
        img_disp = (img_disp * 255).astype(np.uint8)
        img_disp = np.ascontiguousarray(img_disp)  # ✅ ensure memory layout

        # Draw predicted boxes with score > 0.5
        for i, box in enumerate(boxes):
            if scores[i] > 0.5:
                x_min, y_min, x_max, y_max = box.int().tolist()  # Python ints
                cv2.rectangle(img_disp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Convert BGR -> RGB for matplotlib
        img_disp_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8,6))
        plt.imshow(img_disp_rgb)
        plt.axis('off')
        plt.title("Predicted Boxes")
        plt.show()

        break  # only first image
