
# Copyright (c) OpenMMLab. All rights reserved.
#from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from mmdet.datasets.cityscapes import CityscapesDataset as DetectionCityscapesDataset
from mmdet.datasets.coco import CocoDataset as DetectionCocoDataset
from mmdet.datasets.custom import CustomDataset as DetectionCustomDataset
from .dataset_wrapper import DetectionUnpairedDataset

from mmgen.datasets.builder import DATASETS

DATASETS.register_module(module=DetectionCustomDataset, name='DetectionCustomDataset')
DATASETS.register_module(module=DetectionCocoDataset, name='DetectionCocoDataset')
DATASETS.register_module(module=DetectionCityscapesDataset, name='DetectionCityscapesDataset')

__all__ = [
    'DetectionCustomDataset', 'DetectionCocoDataset', 'DetectionCityscapesDataset', 'DetectionUnpairedDataset'
]
