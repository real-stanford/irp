import os
import copy
from pathlib import Path
import json

import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import torch
import torch.utils.data
import pytorch_lightning as pl
from skimage.io import imread
from torchvision import transforms
import imgaug.augmenters as iaa
from hydra.utils import to_absolute_path

from datasets.keypoint_augumentation import (
    build_augmentation_pipeline, get_gaussian_scoremap)


class KeypointImgaugDataset(torch.utils.data.Dataset):
    def __init__(self, 
        sample_df: pd.DataFrame,
        imgaug_cfg: dict,
        enable_aug: bool = True,
        sigma: float = 5,
        **kwargs):
        super().__init__()

        assert(len(sample_df) > 0)
        sample_item = sample_df.iloc[0]
        sample_img_path = sample_item['image_path']
        sample_img = imread(sample_img_path)
        img_shape = sample_img.shape[:2]
        height, width = img_shape

        imgaug_pipeline = iaa.Sequential([])
        if enable_aug:
            apply_prob = imgaug_cfg.get('apply_prob', 0.5)
            imgaug_pipeline = build_augmentation_pipeline(
                imgaug_cfg, 
                height=height, width=width, 
                apply_prob=apply_prob)
        
        def normalize(img):
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            result = (img - mean) / std
            return result

        self.image_transform = normalize
        self.sample_df = sample_df
        self.imgaug_pipeline = imgaug_pipeline
        self.sigma = sigma
        self.kwargs = kwargs
    
    def __len__(self):
        return len(self.sample_df)
    
    def __getitem__(self, idx: int) -> dict:
        sample_df = self.sample_df
        imgaug_pipeline = self.imgaug_pipeline
        sigma = self.sigma

        item = sample_df.iloc[idx]
        keypoint = item['keypoint']
        img_path = item['image_path']
        img = imread(img_path).astype(np.float32) / 255
        img_norm = self.image_transform(img)

        img_aug, kp_aug = imgaug_pipeline(image=img_norm, keypoints=keypoint)
        kp_aug = np.array(kp_aug, dtype=np.float32)
        target_scoremap = get_gaussian_scoremap(
            img_aug.shape[:2], 
            keypoint=kp_aug,
            sigma=sigma)

        data = {
            'input_image': np.moveaxis(img_aug, -1, 0),
            'target_scoremap': np.expand_dims(target_scoremap, axis=0),
            'target_keypoint': kp_aug
        }
        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = torch.from_numpy(value)
        return data_torch


def get_data_df(data_dir, json_name='labels.json', img_format='jpg'):
    rows_dict = dict()
    for json_path in Path(data_dir).glob("*/*"+json_name):
        dir_path = json_path.parent
        this_dir_name = dir_path.name
        # read images
        this_imgs_dict = dict()
        for img_path in dir_path.glob('*.'+img_format):
            this_imgs_dict[img_path.stem] = str(img_path.absolute())
        this_labels_dict = json.load(json_path.open('r'))
        # key intersection
        valid_keys = this_imgs_dict.keys() & this_labels_dict.keys()
        for key in valid_keys:
            key_tuple = (this_dir_name,) + tuple(key.split('_'))
            rows_dict[key_tuple] = {
                'keypoint': tuple(this_labels_dict[key]),
                'image_path': this_imgs_dict[key]
            }
    data_df = pd.DataFrame(
        data=rows_dict.values(), 
        index=rows_dict.keys())
    data_df.sort_index(inplace=True)
    return data_df


class KeypointImgaugDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.sample_dfs = None
    
    def prepare_data(self):
        kwargs = self.kwargs
        data_dir = os.path.expanduser(
            to_absolute_path(kwargs['data_dir']))
        data_df = get_data_df(data_dir)

        # split
        train_split = kwargs.get('train_split', 0.9)
        num_train = int(len(data_df) * train_split)

        split_seed = kwargs.get('split_seed', 0)
        rs = np.random.RandomState(seed=split_seed)
        all_idxs = rs.permutation(len(data_df))
        train_idxs = all_idxs[:num_train]
        val_idxs = all_idxs[num_train:]

        train_df = data_df.iloc[train_idxs]
        val_df = data_df.iloc[val_idxs]

        sample_dfs = {
            'train': train_df,
            'val': val_df
        }

        self.sample_dfs = sample_dfs

    def get_dataset(self, set_name: str, **kwargs):
        assert(set_name in ['train', 'val'])
        sample_dfs = self.sample_dfs
        sample_df = sample_dfs[set_name]

        enable_aug = (set_name == 'train')
        dataset_kwargs = copy.deepcopy(self.kwargs)
        dataset_kwargs['sample_df'] = sample_df
        dataset_kwargs['enable_aug'] = enable_aug
        dataset_kwargs.update(**kwargs)
        dataset = KeypointImgaugDataset(**dataset_kwargs)
        return dataset
    
    def get_dataloader(self, set_name: str):
        assert(set_name in ['train', 'val'])
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']

        is_train = (set_name == 'train')
        dataset = self.get_dataset(set_name)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=num_workers)
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')
