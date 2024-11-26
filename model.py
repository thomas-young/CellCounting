'''
File: model.py
Author: Abdurahman Mohammed
Co-Author: Thomas Young
Co-Author: GPT-o1
Date: 2024-11-14
Description: A Python script that defines a PyTorch model for the cell counting task.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CellCounter(nn.Module):
    '''
    A model that uses a pretrained VGG16 as a feature extractor and outputs a density map.
    '''

    def __init__(self, pretrained=True, freeze_features=True, unfreeze_from_layer=None):
        super(CellCounter, self).__init__()

        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for idx, layer in enumerate(self.vgg16.features):
            print(f"Layer {idx}: {layer}")
        # Optionally freeze all feature layers
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False

        # Unfreeze layers from a specific point
        if unfreeze_from_layer is not None:
            for layer in self.vgg16.features[unfreeze_from_layer:]:
                for param in layer.parameters():
                    param.requires_grad = True

        for idx, layer in enumerate(self.vgg16.features):
            for param in layer.parameters():
                print(f"Layer {idx}, requires_grad={param.requires_grad}")
        # Extract the features (convolutional layers only)
        self.features = self.vgg16.features

        # Define new layers for density map prediction (decoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Assuming the last conv layer outputs 512 channels
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # Ensure output is non-negative
        )

        # Initialize decoder weights
        self.decoder.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x