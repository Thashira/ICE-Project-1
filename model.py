# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import torchvision.transforms.functional as F
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torch.utils.data import DataLoader
# from step6_dataset import FabricDataset  # your dataset
# from step7_model import model, DEVICE, NUM_CLASSES

# # print("OpenCV:", cv2.__version__)
# # print("Torch:", torch.__version__)
# # print("NumPy:", np.__version__)

# # img_path = "train/images/2_bmp.rf.620909ed186ccd7c289b1396cdd8b2f0.jpg"
# # label_path = "train/labels/2_bmp.rf.620909ed186ccd7c289b1396cdd8b2f0.txt"

# TRAIN_IMG_DIR = "train/images"
# TRAIN_LABEL_DIR = "train/labels"

# DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("Using device:",DEVICE)

# model = fasterrcnn_resnet50_fpn(pretrained=True)

# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, NUM_CLASSES)
# model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, NUM_CLASSES * 4)

# model = model.to(DEVICE)
# print("Model loaded on:", DEVICE)
                                
# # img = cv2.imread(img_path)
# # h,w,_=img.shape
# # print(h,w)

# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # plt.imshow(img)
# # plt.axis('off')
# # print(img.shape)
# # plt.show()


# # with open(label_path, "r") as f:
# #     lines = f.readlines()

# # class_id,x,y,bw,bh=map(float,lines[0].split())
# # x1=int((x-bw/2)*w)
# # y1=int((y-bh/2)*h)
# # x2=int((x+bw/2)*w)
# # y2=int((y+bh/2)*h)
# # print(class_id,x1,y1,x2,y2)
# # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
# # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# # plt.imshow(img)
# # plt.axis('off')
# # plt.show()

# # class FabricDataset(torch.utils.data.Dataset):
# #     def __init__(self, img_dir, label_dir):
# #         self.img_dir = img_dir
# #         self.label_dir = label_dir
# #         self.images = os.listdir(img_dir)

# #     def __len__(self):
# #         return len(self.images)

# #     def __getitem__(self, idx):
# #         img_name = self.images[idx]
# #         img_path = os.path.join(self.img_dir, img_name)
# #         label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

# #         img = cv2.imread(img_path)
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         h, w, _ = img.shape

# #         boxes = []
# #         labels = []

# #         # Read label file if exists
# #         if os.path.exists(label_path):
# #             with open(label_path) as f:
# #                 for line in f:
# #                     class_id, x, y, bw, bh = map(float, line.strip().split())
# #                     # YOLO -> pixel coordinates
# #                     x_center = x * w
# #                     y_center = y * h
# #                     box_w = bw * w
# #                     box_h = bh * h
# #                     x_min = x_center - box_w / 2
# #                     y_min = y_center - box_h / 2
# #                     x_max = x_center + box_w / 2
# #                     y_max = y_center + box_h / 2
# #                     boxes.append([x_min, y_min, x_max, y_max])
# #                     labels.append(1)  # defect class = 1

# #         boxes = torch.tensor(boxes, dtype=torch.float32)
# #         labels = torch.tensor(labels, dtype=torch.int64)

# #         target = {"boxes": boxes, "labels": labels}
# #         img = F.to_tensor(img)

# #         return img, target

# # dataset = FabricDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
# # img, target = dataset[0]
# # print("Image shape:", img.shape)
# # print("Target:", target)

import torch
from torch.utils.data import DataLoader
from step6_dataset import FabricDataset  # your dataset
from step7_model import model, DEVICE, NUM_CLASSES

TRAIN_IMG_DIR = "train/images"
TRAIN_LABEL_DIR = "train/labels"

# Dataset & DataLoader
dataset = FabricDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
loader = DataLoader(
    dataset, batch_size=2, shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))  # Required for detection models
)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# Training loop
model.train()
for epoch in range(1):  # 1 epoch
    for imgs, targets in loader:
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print("Loss:", losses.item())

print("Training completed!")
