from typing import Dict, Tuple, Optional
import time
from threading import Thread
import pickle
from hydra.utils import to_absolute_path

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from matplotlib import cm
import cv2 as cv
from scipy.interpolate import interp1d

from real_ur5.zed_camera import ZedCamera
from real_ur5.swing_actor import SwingActor
from real_ur5.zed_ram_recorder import ZedRamRecorder
from real_ur5.trajectory_projector import (
    TrajectoryProjector, GridCoordTransformer)
from networks.keypoint_deeplab import KeypointDeeplab
from components.tracking import KeypointTracker
from real_ur5.buffered_tracker import BufferedTracker
from real_ur5.delta_action_sampler import get_distance


class Ur5WhipEnvironment:
    def __init__(self, env_cfg: DictConfig, camera: ZedCamera):
        self.env_cfg = env_cfg
        # init action
        self.actor = SwingActor(**env_cfg.action.swing_actor)
        # init tracking
        vc = env_cfg.vision
        self.transformer = GridCoordTransformer.from_zarr(
            to_absolute_path(vc.transformer.zarr_path))
        
        cal_data = pickle.load(open(
            to_absolute_path(vc.projector.calib_path), 'rb'))
        tx_img_robot = cal_data['tx_img_robot']
        self.projector = TrajectoryProjector(
            tx_img_robot=tx_img_robot,
            transformer=self.transformer,
            **vc.projector)

        tc = vc.tracker
        device = torch.device('cuda', tc.gpu_id)
        model = KeypointDeeplab.load_from_checkpoint(
            to_absolute_path(tc.ckpt_path))
        tracker = KeypointTracker(model)
        tracker.to(device=device, dtype=torch.float16)
        self.tracker = tracker
        self.device = device

        self.camera = camera
        self.goal_pix = None

    def __enter__(self):
        self.actor.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.actor.__exit__(exc_type, exc_value, exc_traceback)
        
    def set_goal(self, goal: Tuple[float, float]):
        self.goal_pix = tuple(self.transformer.to_grid([goal], clip=True)[0])

    def set_goal_pix(self, goal_pix: Tuple[int, int]):
        self.goal_pix = tuple(goal_pix)
    
    def reset(self):
        self.actor.reset(blocking=True)
        time.sleep(0.5)
    
    def step(self, action: np.ndarray
            ) -> Tuple[np.ndarray, float, bool, dict]:
        if self.goal_pix is None:
            raise RuntimeError('Please call set_goal before step.')

        # prep
        eps = 1e-7
        action = np.clip(action, 0, 1-eps)
        vc = self.env_cfg.vision
        recorder = ZedRamRecorder(self.camera,
            **vc.ram_recorder)
        buff_tracker = BufferedTracker(
            recorder=recorder,
            tracker=self.tracker,
            **vc.buff_tracker)
        
        # wait
        if 'pre_wait' in self.env_cfg.action:
            time.sleep(self.env_cfg.action.pre_wait)

        # action
        ac = self.env_cfg.action
        buff_tracker.start()
        self.actor.swing(*action.tolist(), blocking=True)
        time.sleep(ac.action_duration)
        buff_tracker.stop()

        # async reset
        def reset_func(actor, action, duration, speed):
            actor.swing(*action, blocking=True)
            time.sleep(duration)
            actor.reset(speed=speed, blocking=True)
        reset_t = Thread(target=reset_func, 
            args=(self.actor,),
            kwargs=ac.reset)
        reset_t.start()
        
        # get tracking
        tracking_result = buff_tracker.get_tracking()
        traj_img = self.projector.get_sim_traj_img(tracking_result)

        # compute loss
        dist_to_goal_pix = get_distance(traj_img, self.goal_pix)[0]
        pix_per_m = self.transformer.pix_per_m
        dist_to_goal_m = dist_to_goal_pix / pix_per_m

        # compute trajectory
        keypoints = tracking_result['keypoints'][:,0,:]
        confidence = tracking_result['confidence'][:,0]
        mask = confidence > self.projector.confidence_threshold
        # if mask.sum() < 10:
        #     raise RuntimeError("Invalid tracking")
        raw_x = np.linspace(0,1,len(keypoints))
        valid_keypoints = keypoints[mask]
        valid_x = raw_x[mask]

        def nearest_interp1d(x, y, **kwargs):
            return interp1d(x, y, bounds_error=False, fill_value=(y[0],y[-1]),**kwargs)
        raw_traj = nearest_interp1d(valid_x, valid_keypoints, axis=0)(raw_x)
        robot_traj = self.projector.to_robot_frame(raw_traj)
        if self.projector.flip_x:
            robot_traj[:,0] *= -1
        pix_traj = self.projector.transformer.to_grid(robot_traj)

        observation = traj_img
        loss = dist_to_goal_m
        done = False
        info = {
            'goal_pix': self.goal_pix,
            'action': action,
            'images': np.concatenate(buff_tracker.imgs_list, axis=0),
            'tracking_result': tracking_result,
            'trajectory': robot_traj,
            'trajectory_pix': pix_traj
        }
        torch.cuda.synchronize(self.device)
        return observation, loss, done, info
