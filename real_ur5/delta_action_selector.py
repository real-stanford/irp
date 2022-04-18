from typing import Callable
import numpy as np
import pytorch_lightning as pl
import torch

from common.torch_util import to_numpy
from common.sample_util import ArraySlicer, get_nd_index_volume


def dict_expand_batch(input, batch_size):
    output = dict()
    for key, value in input.items():
        output[key] = value.expand(batch_size, *value.shape[1:])
    return output


class DeltaActionSelector:
    def __init__(self, 
            model: pl.LightningModule, 
            batch_size=32):
        self.model = model
        self.batch_size = batch_size

    def get_dist_img(self, goal_pix, img_shape):
        """
        Unit in pixels
        """
        img_coords = get_nd_index_volume(img_shape)
        dist_img = np.linalg.norm(
            img_coords - goal_pix, axis=-1)
        return dist_img

    def get_delta_action(self, 
            traj_img:np.ndarray, 
            goal_pix:np.ndarray,
            delta_action_samples:np.ndarray,
            threshold:float):
        with torch.no_grad():
            device = self.model.device
            dtype = self.model.dtype
            input_trajectory = torch.from_numpy(
                traj_img.reshape(1,1,*traj_img.shape)).to(
                    device=device, dtype=dtype, non_blocking=False)


            num_samples = delta_action_samples.shape[0]
            batch_size = self.batch_size
            slicer = ArraySlicer((num_samples,), (batch_size,))
            all_probs = list()
            for i in range(len(slicer)):
                this_slice = slicer[i][0]
                this_samples = delta_action_samples[this_slice]
                this_batch_size = len(this_samples)
                this_input_trajectory = input_trajectory.expand(
                    this_batch_size, *input_trajectory.shape[1:])
                x = torch.from_numpy(this_samples).to(
                    device=device, dtype=dtype, non_blocking=True)
                x = torch.sigmoid(self.model(this_input_trajectory, x))
                all_probs.append(x)
            probs = torch.cat(all_probs, dim=0)
            probs = to_numpy(probs[:,0,:,:]).astype(np.float32)
            can_hit_mask = probs > threshold

            # action selection
            dist_img = self.get_dist_img(goal_pix, traj_img.shape)
            min_dists = list()
            for i in range(len(can_hit_mask)):
                dists = dist_img[can_hit_mask[i]]
                if len(dists) == 0:
                    min_dists.append(np.inf)
                else:
                    min_dists.append(dists.min())
            min_dists = np.array(min_dists)
            best_action_idx = np.argmin(min_dists)
            best_delta_action = delta_action_samples[best_action_idx]

            result = {
                'best_delta_action': best_delta_action,
                'best_action_idx': best_action_idx,
                'best_mask': can_hit_mask[best_action_idx],
                'best_prob': probs[best_action_idx],
                'probs': probs,
                'distances': min_dists
            }
            return result


class DeltaActionLossSelector:
    def __init__(self, 
            model: pl.LightningModule, 
            batch_size=32):
        self.model = model
        self.batch_size = batch_size

    def get_dist_img(self, goal_pix, img_shape):
        """
        Unit in pixels
        """
        img_coords = get_nd_index_volume(img_shape)
        dist_img = np.linalg.norm(
            img_coords - goal_pix, axis=-1)
        return dist_img

    def get_delta_action(self, 
            traj_img:np.ndarray, 
            delta_action_samples:np.ndarray,
            loss_func: Callable[[np.ndarray], float],
            threshold:float):
        with torch.no_grad():
            device = self.model.device
            dtype = self.model.dtype
            input_trajectory = torch.from_numpy(
                traj_img.reshape(1,*traj_img.shape)).to(
                    device=device, dtype=dtype, non_blocking=False)

            num_samples = delta_action_samples.shape[0]
            batch_size = self.batch_size
            slicer = ArraySlicer((num_samples,), (batch_size,))
            all_probs = list()
            for i in range(len(slicer)):
                this_slice = slicer[i][0]
                this_samples = delta_action_samples[this_slice]
                this_batch_size = len(this_samples)
                this_input_trajectory = input_trajectory.expand(
                    this_batch_size, *input_trajectory.shape[1:])
                x = torch.from_numpy(this_samples).to(
                    device=device, dtype=dtype, non_blocking=True)
                x = torch.sigmoid(self.model(this_input_trajectory, x))
                all_probs.append(x)
            probs = to_numpy(torch.cat(all_probs, dim=0)).astype(np.float32)
            can_hit_mask = probs > threshold

            # action selection
            loss_arr = list()
            for i in range(len(can_hit_mask)):
                loss_arr.append(loss_func(can_hit_mask[i]))
            best_action_idx = np.argmin(loss_arr)
            best_delta_action = delta_action_samples[best_action_idx]

            result = {
                'best_delta_action': best_delta_action,
                'best_action_idx': best_action_idx,
                'best_mask': can_hit_mask[best_action_idx],
                'best_prob': probs[best_action_idx],
                'probs': probs
            }
            return result
