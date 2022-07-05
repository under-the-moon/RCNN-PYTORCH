"""
@Time ：2022/7/5 7:25
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
from net.backbone.efficientnet import _efficientnet, EfficientNet, model_urls
from typing import Any, Callable, Optional, List, Sequence


def get_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, pretrained, progress, **kwargs)
