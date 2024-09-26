import torch
import torch.nn as nn
import timm

class SwinSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinSegmentationModel, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.backbone.head = nn.Identity()  # Remove classification head

        self.segmentation_head = nn.Conv2d(1024, num_classes, kernel_size=1)  # Adjust input channels

    def forward(self, x):
        features = self.backbone(x)
        features = features.permute(0, 3, 1, 2)  # Change shape to [N, C, H, W]
        segmentation_map = self.segmentation_head(features)
        return segmentation_map
