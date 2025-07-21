# model/plant_model.py

import torch
import torch.nn as nn

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=3):
        super(PlantDiseaseModel, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=2)
            )

        self.conv_block1 = conv_block(3, 64)
        self.conv_block2 = conv_block(64, 128)
        self.conv_block3 = conv_block(128, 256)
        self.conv_block4 = conv_block(256, 512)
        self.conv_block5 = conv_block(512, 512)

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),  # Adjust this based on your input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.fc_block(x)
        return x
