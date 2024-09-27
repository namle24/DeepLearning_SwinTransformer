import torch
import torch.nn as nn
import timm

class SwinTransform(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransform, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.backbone.head = nn.Identity()

        self.segmentation_head = nn.Conv2d(1024, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        features = features.permute(0, 3, 1, 2)
        segmentation_map = self.segmentation_head(features)
        return segmentation_map
