import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

import numpy as np  

class Vgg19Embedding(nn.Module):
    """
    args: feature_layer: 22 conv2d-10, 35 last conv2d 38: after maxpool  42: 
    """
    def __init__(self, feature_layer):
        nn.Module.__init__(self)
        self.feature_layer= feature_layer 
        model = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:self.feature_layer])
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)


    def forward(self, x):
        # Assume input range is [0, 1]
        x = (x - self.mean) / self.std
        x = self.features(x)
        if self.feature_layer == 22:
            x = F.max_pool2d(x, 2, 2)
            x = F.max_pool2d(x, 2, 2)
            x = F.max_pool2d(x, 2, 2)
            x = F.max_pool2d(x, 2, 2)
        # 512 x 7 x 7 25088
        x = x.view(x.shape[0], -1)
        return x
