import os
import cv2
import torch
from torchvision.transforms import functional as F

TRAIN_IMG_DIR = "train/images"
TRAIN_LABEL_DIR = "train/labels"

class FabricDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        boxes = []
        labels = []

        # Read label file if exists
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    class_id, x, y, bw, bh = map(float, line.strip().split())
                    # YOLO -> pixel coordinates
                    x_center = x * w
                    y_center = y * h
                    box_w = bw * w
                    box_h = bh * h
                    x_min = x_center - box_w / 2
                    y_min = y_center - box_h / 2
                    x_max = x_center + box_w / 2
                    y_max = y_center + box_h / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)  # defect class = 1

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        img = F.to_tensor(img)

        return img, target

# Test your dataset
dataset = FabricDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
img, target = dataset[0]
print("Image shape:", img.shape)
print("Target:", target)
