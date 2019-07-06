"""EfficientNet architecture.
See:
- https://arxiv.org/abs/1905.11946 - EfficientNet
- https://arxiv.org/abs/1801.04381 - MobileNet V2
- https://arxiv.org/abs/1905.02244 - MobileNet V3
- https://arxiv.org/abs/1709.01507 - Squeeze-and-Excitation
- https://arxiv.org/abs/1803.02579 - Concurrent spatial and channel squeeze-and-excitation
"""

import math
import torch
import torch.nn as nn
import numpy as np
import collections
import torch.nn.functional as F
import utils
from models.layers import DropConnect, SamePadConv2d, Attention, Swish, conv_bn_act


EfficientNetParam = collections.namedtuple("EfficientNetParam", [
"width", "depth", "resolution", "dropout"])

# just for reference
EfficientNetParams = {
  "B0": EfficientNetParam(1.0, 1.0, 224, 0.2),
  "B1": EfficientNetParam(1.0, 1.1, 240, 0.2),
  "B2": EfficientNetParam(1.1, 1.2, 260, 0.3),
  "B3": EfficientNetParam(1.2, 1.4, 300, 0.3),
  "B4": EfficientNetParam(1.4, 1.8, 380, 0.4),
  "B5": EfficientNetParam(1.6, 2.2, 456, 0.4),
  "B6": EfficientNetParam(1.8, 2.6, 528, 0.5),
  "B7": EfficientNetParam(2.0, 3.1, 600, 0.5)}


def efficientnet0(pretrained=False, num_classes=1000):
    model = EfficientNet(num_classes=num_classes, width_coef=1., depth_coef=1., scale=224., dropout_ratio=0.2, 
                         se_reduction=24, drop_connect_ratio=0.5)
    if pretrained:
        checkpoint = torch.load('/home/denys/DataScience/projects/owoschi/classification/models/weight/efficientnet-b0.pth')
        filtered_state_dict = {k:v for k, v in checkpoint.items() if 'features' in k}
        model.load_state_dict(filtered_state_dict, strict=False)
    return model

def efficientnet1(pretrained=False, num_classes=1000):
    model = EfficientNet(num_classes=num_classes, width_coef=1., depth_coef=1.1, scale=240., dropout_ratio=0.2, 
                         se_reduction=24, drop_connect_ratio=0.5)
    if pretrained:
        checkpoint = torch.load('/home/denys/DataScience/projects/owoschi/classification/models/weight/efficientnet-b1.pth')
        filtered_state_dict = {k:v for k, v in checkpoint.items() if 'features' in k}
        model.load_state_dict(filtered_state_dict, strict=False)
    return model

def efficientnet2(pretrained=False, num_classes=1000):
    model = EfficientNet(num_classes=num_classes, width_coef=1.1, depth_coef=1.2, scale=260., dropout_ratio=0.3, 
                         se_reduction=24, drop_connect_ratio=0.5)
    if pretrained:
        checkpoint = torch.load('/home/denys/DataScience/projects/owoschi/classification/models/weight/efficientnet-b2.pth')
        filtered_state_dict = {k:v for k, v in checkpoint.items() if 'features' in k}
        model.load_state_dict(filtered_state_dict, strict=False)
    return model

def efficientnet3(pretrained=False, num_classes=1000):
    model = EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=300., dropout_ratio=0.3, 
                         se_reduction=24, drop_connect_ratio=0.5)
    if pretrained:
        checkpoint = torch.load('/home/denys/DataScience/projects/owoschi/classification/models/weight/efficientnet-b3.pth')
        filtered_state_dict = {k:v for k, v in checkpoint.items() if 'features' in k}
        model.load_state_dict(filtered_state_dict, strict=False)
    return model
        

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, expand_ratio, kernel_size, stride, se_reduction, drop_connect_ratio=0.2):
        """Basic building block - Inverted Residual Convolution from MobileNet V2 
        architecture.

        Arguments:
            expand_ratio (int): ratio to expand convolution in width inside convolution.
                It's not the same as width_mult in MobileNet which is used to increase
                persistent input and output number of channels in layer. Which is not a
                projection of channels inside the conv. 
        """
        super().__init__()
        
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        if self.use_res_connect:
            self.dropconnect = DropConnect(drop_connect_ratio)
            
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depth-wise
                SamePadConv2d(inp=hidden_dim, oup=hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                Attention(channels=hidden_dim, reduction=4),  # somehow here reduction should be always 4
                Swish(), 
                
                # point-wise-linear
                SamePadConv2d(inp=hidden_dim, oup=oup, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # point-wise
                SamePadConv2d(inp, hidden_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish(), 
                                
                # depth-wise
                SamePadConv2d(hidden_dim, hidden_dim, kernel_size, stride, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Attention(channels=hidden_dim, reduction=se_reduction),  
                Swish(), 
                
                # point-wise-linear
                SamePadConv2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, inputs):
        if self.use_res_connect: 
            return self.dropconnect(inputs) + self.conv(inputs)
        else: 
            return self.conv(inputs)

        
def round_filters(filters, width_coef, depth_divisor=8, min_depth=None):
    """ Calculate and round number of filters based on depth multiplier. """
    if not width_coef:
        return filters
    filters *= width_coef
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coef):
    """ Round number of filters based on depth multiplier. """
    if not depth_coef:
        return repeats
    return int(math.ceil(depth_coef * repeats)) 
    
        
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coef=1., depth_coef=1., scale=1., dropout_ratio=0.2, 
                 se_reduction=24, drop_connect_ratio=0.5):
        super(EfficientNet, self).__init__()
        
        block = InvertedResidual
        input_channel     = round_filters(32, width_coef)   # input_channel = round(32*width_coef) 
        self.last_channel = round_filters(1280, width_coef)     # self.last_channel  = round(1280*width_coef)
        config = np.array([
            # stride only for first layer in group, all other always with stride 1
            # channel,expand,repeat,stride,kernel_size
            [16,  1, 1, 1, 3],
            [24,  6, 2, 2, 3],
            [40,  6, 2, 2, 5],
            [80,  6, 3, 2, 3],
            [112, 6, 3, 1, 5],
            [192, 6, 4, 2, 5],
            [320, 6, 1, 1, 3],
            ])

        # first steam layer - ordinar conv
        self.features = [conv_bn_act(3, input_channel, kernel_size=3, stride=2, bias=False)]

        # main 7 group of layers
        for c, t, n, s, k in config:
            output_channel = round_filters(c, width_coef)
            for i in range(round_repeats(n, depth_coef)):
                if i == 0:
                    self.features.append(block(inp=input_channel, oup=output_channel, expand_ratio=t, kernel_size=k, 
                                               stride=s, se_reduction=se_reduction, drop_connect_ratio=drop_connect_ratio))
                else:
                    # here stride is equal 1 because only first layer in group could have stride 2, 
                    self.features.append(block(inp=input_channel, oup=output_channel, expand_ratio=t, kernel_size=k, 
                                               stride=1, se_reduction=se_reduction, drop_connect_ratio=drop_connect_ratio))
                input_channel = output_channel
        
        # building last several layers
        self.features.append(conv_bn_act(input_channel, self.last_channel, kernel_size=1, bias=False))
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
            )
                  
        self._initialize_weights()

                      
    def _initialize_weights(self):
        flattener = utils.Flattener()
        flattened = flattener(self)
        for m in flattened:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x