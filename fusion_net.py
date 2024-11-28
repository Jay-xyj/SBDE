import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        
        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(1)
        
        self.conv3x3 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(1)

    def forward(self, part_mask0, part_mask1, part_mask2):
        overall_mask = part_mask0 + part_mask1 + part_mask2

        overall_mask = self.bn1x1(self.conv1x1(overall_mask))
        
        overall_mask = self.bn3x3(self.conv3x3(overall_mask))
        
        return overall_mask
