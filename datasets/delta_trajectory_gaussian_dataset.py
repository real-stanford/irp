import os
import copy

import numpy as np
from numpy.random.mtrand import random
from torch.utils.data.dataloader import DataLoader
import zarr
import torch
import torch.utils.data
import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from common.sample_util import VirtualSampleGrid, get_nd_index_volume
from common.async_dataloader import AsynchronousLoader
from common.cv_util import get_dist_function_precise
from common.zarr_util import open_cached

class DeltaTrajectoryGaussianDataset(torch.utils.data.Dataset):
    def __init__(self,
        # data
        zarr_path: str,
        name: str = 'traj_occu',
        zarr_cache_size: str = '4GB',
        # sample setup
        rope_coords = None,
        is_setup_valid = None,
        action_sigma = 0.125,
        random_init = True,
        size = 10000,
        # image setup
        distance = False,
        pix_per_m = 60.34,
        direction = False,
        # training params
        static_epoch_seed: bool = False,
        **kwargs):
        super().__init__()

        assert(not (distance and direction))
        path = os.path.expanduser(to_absolute_path(zarr_path))
        assert(os.path.isdir(path))
        root = open_cached(path, mode='r', cache_size=zarr_cache_size)
        data_array = root[name]

        assert(rope_coords is not None)
        assert(is_setup_valid is not None)
        
        self.data_array = data_array
        self.rope_coords = rope_coords
        self.is_setup_valid = is_setup_valid
        self.static_epoch_seed = static_epoch_seed
        self.action_sigma = action_sigma
        self.random_init = random_init
        self.size = size
        self.distance = distance
        self.pix_per_m = pix_per_m
        self.direction = direction
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> dict:
        data_array = self.data_array
        rope_coords = self.rope_coords
        is_setup_valid = self.is_setup_valid
        static_epoch_seed = self.static_epoch_seed
        action_sigma = self.action_sigma
        random_init = self.random_init

        action_shape = data_array.shape[2:5]
        n_init = data_array.shape[5]

        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)

        # select rope
        rope_coord = tuple(rope_coords[rs.choice(len(rope_coords))].tolist())

        # rejection sample base action
        n_action = np.prod(action_shape)
        base_rs = np.random.RandomState(seed=rs.randint(0,2**16))
        while True:
            base_action_idx = base_rs.randint(0, n_action)
            base_action_coord = np.unravel_index(
                base_action_idx, action_shape)
            coord = rope_coord + base_action_coord
            if is_setup_valid[coord]:
                break

        # rejection sample delta action
        action_scale = np.array(action_shape)
        delta_rs = np.random.RandomState(seed=rs.randint(0,2**16))
        while True:
            delta_action = delta_rs.normal(loc=0, scale=action_sigma, size=3)
            delta_coord = (action_scale * delta_action).round().astype(np.int64)
            next_coord = np.array(base_action_coord) + delta_coord
            if np.any(next_coord < 0) or np.any(
                next_coord >= np.array(action_shape)):
                continue
            next_action_coord = tuple(next_coord.tolist())
            coord = rope_coord + next_action_coord
            if is_setup_valid[coord]:
                break

        # get init
        init_coord = (n_init // 2,)
        if random_init:
            init_coord = (int(rs.randint(0, n_init)),)

        # get data
        input_trajectory = data_array[
            rope_coord + base_action_coord + init_coord]
        delta_trajectory = data_array[
            rope_coord + next_action_coord + init_coord]

        # convert data
        if self.distance:
            pix_per_m = self.pix_per_m
            input_df = get_dist_function_precise(input_trajectory) / pix_per_m
            delta_df = get_dist_function_precise(delta_trajectory) / pix_per_m
            input_trajectory = input_df
            delta_trajectory = delta_df
        elif self.direction:
            input_df = get_dist_function_precise(input_trajectory)
            delta_df = get_dist_function_precise(delta_trajectory)
            is_closer = delta_df < input_df
            delta_trajectory = is_closer
        
        action_delta = delta_coord / action_scale
        data = {
            'input_trajectory': np.expand_dims(input_trajectory, axis=0).astype(np.float32),
            'action_delta': action_delta.astype(np.float32),
            'target_trajectory': np.expand_dims(delta_trajectory, axis=0).astype(np.float32)
        }

        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = torch.from_numpy(value)
        return data_torch


class DeltaTrajectoryGaussianDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.set_rope_coords = None
        self.is_setup_valid = None

    def prepare_data(self):
        kwargs = self.kwargs
        zarr_path = os.path.expanduser(
            to_absolute_path(kwargs['zarr_path']))
        root = zarr.open(zarr_path, 'r')

        is_setup_valid = root['is_valid'][:]
        self.is_setup_valid = is_setup_valid

        is_rope_train = root['split/is_train'][:]
        is_rope_val = root['split/is_val'][:]
        is_rope_test = root['split/is_test'][:]
        
        rope_shape = is_setup_valid.shape[:2]
        rope_coords_volume = get_nd_index_volume(rope_shape)
        set_rope_coords = {
            'train': rope_coords_volume[is_rope_train],
            'val': rope_coords_volume[is_rope_val],
            'test': rope_coords_volume[is_rope_test]
        }
        self.set_rope_coords = set_rope_coords
    
    def get_dataset(self, set_name: str):
        assert(set_name in ['train', 'val', 'test'])
        kwargs = self.kwargs

        static_epoch_seed = (set_name != 'train')
        dataset_kwargs = copy.deepcopy(kwargs)
        dataset_kwargs['rope_coords'] = self.set_rope_coords[set_name]
        dataset_kwargs['is_setup_valid'] = self.is_setup_valid
        dataset_kwargs['size'] = kwargs['size'][set_name]
        dataset_kwargs['static_epoch_seed'] = static_epoch_seed
        dataset = DeltaTrajectoryGaussianDataset(**dataset_kwargs)
        return dataset
    
    def get_dataloader(self, set_name: str):
        assert(set_name in ['train', 'val', 'test'])
        kwargs = self.kwargs
        dataloader_params = kwargs['dataloader_params']

        is_train = (set_name == 'train')
        dataset = self.get_dataset(set_name)

        dataloader = DataLoader(
            dataset,
            shuffle=is_train,
            drop_last=is_train,
            **dataloader_params)

        if kwargs['async_device'] is not None:
            device_id = kwargs['async_device']
            device = torch.device('cuda', device_id)
            dataloader = AsynchronousLoader(
                data=dataloader, device=device)

        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')
    
    def test_dataloader(self):
        return self.get_dataloader('test')
