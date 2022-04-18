from typing import Tuple, Optional

import numpy as np
from hydra.utils import to_absolute_path

from common.sample_util import GridCoordTransformer, get_nd_index_volume
from common.zarr_util import open_cached

class DatasetEnvironment:
    def __init__(self, 
            zarr_path: str,
            name: str,
            rope_id: Tuple[int, int], 
            transformer: GridCoordTransformer,
            random_init: bool=False, 
            seed: Optional[bool]=None,
            cache_size: str='10GB',
            raw_data_name: str=None):
        rope_id = tuple(rope_id)
        
        zarr_root = open_cached(
            to_absolute_path(zarr_path), 'r', 
            cache_size=cache_size)
        data_array = zarr_root[name]
        raw_data_array = None
        if raw_data_name is not None:
            raw_data_array = zarr_root[raw_data_name]
        nn_valid_action = zarr_root['fill_invalid/nn_valid_action'][rope_id]
        nn_dist = zarr_root['fill_invalid/nn_dist'][rope_id]

        self.data_array = data_array
        self.raw_data_array = raw_data_array
        self.nn_valid_action = nn_valid_action
        self.nn_dist = nn_dist
        self.rope_id = rope_id
        self.goal_pix = None
        self.random_init = random_init
        self.rs = np.random.RandomState(seed=seed)
        self.transformer = transformer
    
    def set_goal(self, goal: Tuple[float, float]):
        self.goal_pix = tuple(self.transformer.to_grid([goal], clip=True)[0])

    def set_goal_pix(self, goal_pix: Tuple[int, int]):
        self.goal_pix = tuple(goal_pix)
    
    def step(self, action: np.ndarray
            ) -> Tuple[np.ndarray, float, bool, dict]:
        if self.goal_pix is None:
            raise RuntimeError('Please call set_goal before step.')

        eps = 1e-7
        action = np.clip(action, 0, 1-eps)
        action_scale = np.array(
            self.nn_dist.shape, 
            dtype=action.dtype)
        raw_action_coord = tuple((action * action_scale
            ).astype(np.int64).tolist())
        # avoid potential invalid cell
        action_nn_dist = self.nn_dist[raw_action_coord]
        action_coord = tuple(self.nn_valid_action[raw_action_coord].tolist())

        n_inits = self.data_array.shape[-3]
        init_id = n_inits // 2
        if self.random_init:
            init_id = self.rs.choice(init_id, size=1)[0]

        coord = self.rope_id + action_coord + (init_id,)
        traj_img = self.data_array[coord]

        # compute error
        img_coords = get_nd_index_volume(traj_img.shape)
        traj_coords = img_coords[traj_img]
        dists_pix = np.linalg.norm(traj_coords - self.goal_pix, axis=-1)
        dist_pix = np.min(dists_pix)
        dist_m = dist_pix / self.transformer.pix_per_m

        observation = traj_img
        loss = dist_m
        done = False
        info = {
            'action': np.array(action_coord) / action_scale,
            'action_coord': action_coord,
            'action_nn_dist': action_nn_dist,
            'init_id': init_id
        }
        # add trajectory
        if self.raw_data_array is not None:
            data = self.raw_data_array[coord]
            has_data = np.all(np.isfinite(data),axis=-1)
            trajectory = data[has_data].astype(np.float32)
            trajectory_pix = self.transformer.to_grid(trajectory[:,[0,2]])
            info['trajectory'] = trajectory
            info['trajectory_pix'] = trajectory_pix

        return observation, loss, done, info
