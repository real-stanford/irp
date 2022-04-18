import numpy as np
from common.sample_util import get_nd_index_volume


def get_distance(traj_img, goal_pix):
    coord_img = get_nd_index_volume(traj_img.shape)
    coords = coord_img[traj_img]
    dists = np.linalg.norm(coords - goal_pix, axis=-1)
    min_idx = np.argmin(dists)
    dist = dists[min_idx]
    coord = coords[min_idx]
    return dist, coord


class DeltaActionGridSampler:
    def __init__(self, delta=3, grid_shape=(32,32,32)):
        self.grid_shape = grid_shape
        self.delta = delta
    
    def get_delta_action_samples(self, action: np.ndarray):
        """
        Assume action is 0-1 3d array float32
        """
        # conver to grid coord
        delta = self.delta
        grid_shape = self.grid_shape
        # corner aligned
        corners = np.array(grid_shape, dtype=np.float32) - 1
        action_coord = (action * corners
            ).round().astype(np.int64)
        
        # rejection sample
        delta_coord = get_nd_index_volume((delta*2+1,)*3) - delta
        next_action_coord = delta_coord + action_coord
        is_low = np.all(next_action_coord >= np.array([0,0,0]), axis=-1)
        is_high = np.all(next_action_coord < np.array(grid_shape), axis=-1)
        valid_next_action_coord = next_action_coord[is_low & is_high]
        
        # convert to float
        next_action_samples = valid_next_action_coord.astype(
            np.float32) / corners
        delta_action_samples = next_action_samples - action
        return delta_action_samples


class DeltaActionGaussianSampler:
    def __init__(self, num_samples=128, seed=0, dim=3):
        self.num_samples = num_samples
        self.rs = np.random.RandomState(seed=seed)
        self.dim = dim
    
    def get_delta_action_samples(self, 
            action: np.ndarray, sigma: float=1/32):
        
        delta_actions = list()
        while len(delta_actions) < self.num_samples:
            n_needed = self.num_samples - len(delta_actions)
            delta_samples = self.rs.normal(
                loc=0, scale=sigma, 
                size=(n_needed,self.dim))
            action_samples = delta_samples + action
            is_valid = np.all(
                action_samples >= 0, axis=-1) & np.all(
                    action_samples < 1, axis=-1)
            valid_delta_samples = delta_samples[is_valid]
            delta_actions.extend(valid_delta_samples)
        delta_actions = np.array(delta_actions)
        return delta_actions
