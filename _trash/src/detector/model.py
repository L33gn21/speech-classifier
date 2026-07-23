import torch.nn as nn
from torchvision.models import resnet18

class Detector(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)