# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch
import monai

from monai import transforms
from monai.data import load_decathlon_datalist
from monai import data
import json
import pandas as pd
from monai.transforms import *
import pdb
from monai.transforms import MapTransform
from monai.config import KeysCollection
import random
from typing import Dict, List, Any

MONAI_MAX_SEED = int(np.iinfo(np.uint32).max)
monai.transforms.compose.MAX_SEED = min(monai.transforms.compose.MAX_SEED, MONAI_MAX_SEED)
monai.transforms.transform.MAX_SEED = min(monai.transforms.transform.MAX_SEED, MONAI_MAX_SEED)


ALL_MODALITIES = ['t1', 'flair', 't2', 't1c', 't1n', 't2f', 't2w','dwi', 'adc', 't2f','mra', 'pd']

def collate_fn(batch):
    return batch

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class GetName():
    def __call__(self,data):
        for key in data.keys():
            if key in ALL_MODALITIES:
                # print(data[key])
                data['meta_path'] = data[key]
            break
        return data

class Random2Modal():

    def select_random_images(self, data: Dict[str, Any], num_images: int) -> List[str]:
        images = data.get("image", [])
        if isinstance(images, str):
            images = [images]
        if len(images) == 0:
            return []
        if len(images) >= num_images:
            return random.sample(images, num_images)
        return images + random.choices(images, k=num_images - len(images))

    def __call__(self, data):
        selected_images = self.select_random_images(data, 2)
        if len(selected_images) < 2:
            raise ValueError("Sample does not contain any image paths.")
        return {
            "modal1": selected_images[0],
            "modal2": selected_images[1],
            "modal1_path": selected_images[0],
            "modal2_path": selected_images[1],
        }

class Random4Modal(Random2Modal):

    def __call__(self, data):
        selected_images = self.select_random_images(data, 4)
        if len(selected_images) < 4:
            raise ValueError("Sample does not contain any image paths.")
        return {
            "modal1": selected_images[0],
            "modal2": selected_images[1],
            "modal3": selected_images[2],
            "modal4": selected_images[3],
            "modal1_path": selected_images[0],
            "modal2_path": selected_images[1],
            "modal3_path": selected_images[2],
            "modal4_path": selected_images[3],
        }

class GetShape():
    def __call__(self,data):
        tensor_1 = data['image']
        tensor_2 = data['label']
        if tensor_1.shape != tensor_2.shape:
            tensor_1 = tensor_1.permute(0, 3, 2, 1) if tensor_1.shape[1:] != tensor_2.shape[1:] else tensor_1
            data['image'] = tensor_1
        return data

from collections import defaultdict
from monai.transforms import MapTransform
from monai.transforms import Transform
from typing import Dict, Any


def get_loader(args):
    data_dir = './'
    datalist_json = 'pretrain_data.json'
    modal_keys = ['t1', 'flair', 't2', 't1c', 't1n', 't2f', 't2w','dwi', 'adc', 't2f','mra', 'pd']
    load_keys = ['modal1','modal2']
    train_transforms_list = [ 
            Random2Modal(),
            LoadImaged(keys=load_keys,reader="NibabelReader", ensure_channel_first=False,allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=load_keys, allow_missing_keys=True, channel_dim="no_channel"),
            Orientationd(keys=load_keys, axcodes="RSA",allow_missing_keys=True),
            # transforms.ScaleIntensityRangePercentilesd(keys=load_keys,lower=1, upper=99, b_min=0, b_max=1, clip=True,allow_missing_keys=True, channel_wise=True),
            transforms.NormalizeIntensityd(keys=load_keys,nonzero=True, channel_wise=True),
            # transforms.SpatialPadd(keys=load_keys,spatial_size=(args.roi_large, args.roi_large, args.roi_large)),
            # 
            # transforms.RandSpatialCropd(keys=load_keys, roi_size=(args.roi_large, args.roi_large, args.roi_large), random_size=False),  
            CenterSpatialCropd(
            keys=load_keys,roi_size=(args.roi_large, args.roi_large, args.roi_z),allow_missing_keys=True
            ),
            Resized(keys=load_keys, mode="trilinear", align_corners=True,
                    spatial_size=(args.roi_large, args.roi_large, args.roi_large),allow_missing_keys=True),
            transforms.RandFlipd(keys=load_keys, prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=load_keys, prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=load_keys, prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=load_keys, prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys=load_keys, factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys=load_keys, offsets=0.1, prob=0.1)]

    train_transforms = Compose(train_transforms_list)

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    
    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)
    
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
        # collate_fn=collate_fn
    )
    return train_loader



def get_test_loader(args):
    data_dir = './'
    datalist_json = './test_data.json'
    modal_keys = ['t1', 'flair', 't2', 't1c', 't1n', 't2f', 't2w','dwi', 'adc', 't2f','mra', 'pd']
    # load_keys = modal_keys
    load_keys = ['modal1','modal2','modal3','modal4']
    train_transforms_list = [ 
            # GetName(),
            Random4Modal(),
            LoadImaged(keys=load_keys,reader="NibabelReader", ensure_channel_first=False,allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=load_keys, allow_missing_keys=True, channel_dim="no_channel"),
            Orientationd(keys=load_keys, axcodes="RSA",allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=load_keys,lower=1, upper=99, b_min=0, b_max=1, clip=True,allow_missing_keys=True, channel_wise=True),
            
            CenterSpatialCropd(
            keys=load_keys,roi_size=(args.roi_large, args.roi_large, args.roi_z),allow_missing_keys=True
            ),
            Resized(keys=load_keys, mode="trilinear", align_corners=True,
                    spatial_size=(args.roi_large, args.roi_large, args.roi_large),allow_missing_keys=True),]

    train_transforms = Compose(train_transforms_list)

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    
    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)
    
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
        # collate_fn=collate_fn
    )
    return train_loader


def get_visual_loader(args):
    data_dir = './'
    datalist_json = 'seg_visual.json'
    load_keys = ['image','label']
    train_transforms_list = [
            LoadImaged(keys=load_keys,reader="NibabelReader", ensure_channel_first=False,allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=load_keys, allow_missing_keys=True, channel_dim="no_channel"),
            Orientationd(keys=load_keys, axcodes="RSA",allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"],lower=1, upper=99, b_min=0, b_max=1, clip=True,allow_missing_keys=True, channel_wise=True),
            
            CenterSpatialCropd(
            keys=load_keys,roi_size=(args.roi_large, args.roi_large, args.roi_z),allow_missing_keys=True
            ),
            Resized(keys=load_keys, mode="trilinear", align_corners=True,
                    spatial_size=(args.roi_large, args.roi_large, args.roi_large),allow_missing_keys=True),]

    train_transforms = Compose(train_transforms_list)

    datalist = load_decathlon_datalist(datalist_json, True, "visual", base_dir=data_dir)
    
    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)
    
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
        # collate_fn=collate_fn
    )
    return train_loader
