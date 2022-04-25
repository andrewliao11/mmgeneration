# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.pipelines.formatting import Collect as DetectionCollect
from mmdet.datasets.pipelines.formatting import ImageToTensor as DetectionImageToTensor
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle as DetectionDefaultFormatBundle
from mmdet.datasets.pipelines.loading import LoadAnnotations as DetectionLoadAnnotations
from mmdet.datasets.pipelines.loading import LoadImageFromFile as DetectionLoadImageFromFile
from mmdet.datasets.pipelines.test_time_aug import MultiScaleFlipAug as DetectionMultiScaleFlipAug
from mmdet.datasets.pipelines.transforms import Pad as DetectionPad
from mmdet.datasets.pipelines.transforms import Normalize as DetectionNormalize
from mmdet.datasets.pipelines.transforms import Resize as DetectionResize
from mmdet.datasets.pipelines.transforms import RandomFlip as DetectionRandomFlip


from mmgen.datasets.builder import PIPELINES

PIPELINES.register_module(module=DetectionLoadImageFromFile, name='DetectionLoadImageFromFile')
PIPELINES.register_module(module=DetectionLoadAnnotations, name='DetectionLoadAnnotations')
PIPELINES.register_module(module=DetectionResize, name='DetectionResize')
PIPELINES.register_module(module=DetectionRandomFlip, name='DetectionRandomFlip')
PIPELINES.register_module(module=DetectionNormalize, name='DetectionNormalize')
PIPELINES.register_module(module=DetectionPad, name='DetectionPad')
PIPELINES.register_module(module=DetectionDefaultFormatBundle, name='DetectionDefaultFormatBundle')
PIPELINES.register_module(module=DetectionCollect, name='DetectionCollect')
PIPELINES.register_module(module=DetectionMultiScaleFlipAug, name='DetectionMultiScaleFlipAug')
PIPELINES.register_module(module=DetectionImageToTensor, name='DetectionImageToTensor')


__all__ = [
    'DetectionLoadImageFromFile',
    'DetectionLoadAnnotations',
    'DetectionResize',
    'DetectionRandomFlip',
    'DetectionNormalize',
    'DetectionPad',
    'DetectionDefaultFormatBundle',
    'DetectionCollect',
    'DetectionMultiScaleFlipAug', 
    'DetectionImageToTensor'
]
