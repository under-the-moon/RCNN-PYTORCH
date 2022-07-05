"""
@Time ：2022/6/23 8:28
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""

import os
import sys
import torch.nn as nn
from importlib import import_module
from torch.hub import load_state_dict_from_url

basedir = os.path.dirname(__file__)
sys.path.append(basedir)


class Model(nn.Module):

    def __init__(self, backbone, num_classes=4, pretrained=True, **kwargs):
        super(Model, self).__init__(**kwargs)
        x = import_module('net.backbone.' + backbone)
        backbone_net = x.get_backbone()
        if pretrained:
            backbone_net.load_state_dict(load_state_dict_from_url(x.model_urls[backbone], progress=True), strict=False)
        self.backbone = backbone_net
        self.fc = nn.Linear(1280, 256)
        self.head = nn.Linear(256, num_classes + 1)

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.fc(x)
        output = self.head(x)
        return output, x
