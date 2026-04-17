import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import (
    create_directories, 
    IMAGES_DIR, 
    MASKS_DIR, 
    SEGMENTED_ANYTHING_DIR,
    SIFT_FEATURES_DIR
)

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv4_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), kernel_size=2))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)
    
def load_img(img_src):
    img = cv2.imread(img_src)
    return img