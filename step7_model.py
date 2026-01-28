import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # background + defect

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace classifier for our number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, NUM_CLASSES)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, NUM_CLASSES * 4)

model = model.to(DEVICE)
print("Model loaded on:", DEVICE)
