# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import collections
import copy
import math
from collections import defaultdict

import numpy as np

from mmgen.datasets.builder import DATASETS
from .coco import CocoDataset

import ipdb

@DATASETS.register_module()
class DetectionUnpairedDataset:

    def __init__(self, domain_a_dataset, domain_b_dataset):
        self.domain_a_dataset = domain_a_dataset
        self.domain_b_dataset = domain_b_dataset
        
    def __getitem__(self, idx):
        domain_a_data = self.domain_a_dataset[idx]
        domain_a_data.update({'idx': idx})
        idx_b = np.random.randint(0, len(self.domain_b_dataset))

        domain_b_data = self.domain_b_dataset[idx_b]
        domain_b_data.update({'idx': idx_b})

        return dict(
            domain_a_data=domain_a_data, 
            domain_b_data=domain_b_data
        )

    def get_ann_info(self, idx):
        ipdb.set_trace()
        pass

    def __len__(self):
        return len(self.domain_a_dataset)
