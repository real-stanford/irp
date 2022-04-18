# %%
# import
import os
import pathlib
import yaml
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import zarr
import numpy as np
from tqdm import tqdm
import torch
import wandb
from scipy.interpolate import interp1d
import cv2 as cv
from matplotlib import cm
import pickle
import skvideo.io
from threading import Thread
import shutil

from real_ur5.zed_camera import ZedCamera
from environments.ur5_whip_environment import Ur5WhipEnvironment
from common.sample_util import transpose_data_dict
from common.wandb_util import get_error_plots_log

from networks.delta_trajectory_deeplab import DeltaTrajectoryDeeplab
from real_ur5.delta_action_sampler import DeltaActionGaussianSampler
from real_ur5.delta_action_selector import DeltaActionSelector

# %%
def vis_tracking_goal(images, keypoints, confidence, goal_pix, 
        projector, threshold=0.3):
    vis_img = images.min(axis=0)[:,:,[2,1,0]].copy()
    mask = confidence > threshold
    valid_keypoints = keypoints[mask]
    invalid_keypoints = keypoints[~mask]
    for kp in invalid_keypoints:
        cv.drawMarker(vis_img, tuple(kp), (0,0,255), 
            markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1)
    cv.polylines(vis_img, [valid_keypoints], False, color=(0,255,0))
    goal_kp = projector.grid_to_image([goal_pix])[0].astype(np.int32)
    cv.drawMarker(vis_img, goal_kp, 
            color=(255,0,0), markerType=cv.MARKER_CROSS,
            markerSize=50, thickness=3)
    return vis_img


def vis_action(best_prob, best_mask, observation, goal_pix):
    def draw(input, observation, goal_pix):
        cmap = cm.get_cmap('viridis')
        left_img = cmap(input)[:,:,:3].copy()
        if observation is not None:
            left_img[observation] = (np.array([1,0,0]) + left_img[observation]) / 2
        cv.drawMarker(left_img, goal_pix[::-1], 
                color=(1,0,0), markerType=cv.MARKER_CROSS,
                markerSize=20, thickness=1)
        return np.moveaxis(left_img,0,1)[::-1,::-1]

    left_img = draw(best_prob, None, goal_pix)
    right_img = draw(best_mask.astype(np.float32), observation, goal_pix)
    img = np.concatenate([left_img, right_img], axis=1)
    return img


def save_video(out_fname, images):
    pathlib.Path(out_fname).resolve().parent.mkdir(parents=True, exist_ok=True)
    writer = skvideo.io.FFmpegWriter(
        out_fname, 
        inputdict={
            '-r': '60'
        },
        outputdict={
            '-r': '60',
            '-c:v': 'libx264',
            '-crf': '18',
            '-preset': 'medium',
            '-pix_fmt': 'yuv420p'
            # '-profile:v': 'high'
        })
    for img in images:
        writer.writeFrame(img[:,:,[2,1,0]])
    writer.close()


# %%
@hydra.main(config_path="config", config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig) -> None:
    output_dir= os.getcwd()
    print(output_dir)
    if not cfg.offline:
        wandb.init(**cfg.wandb)
    config = OmegaConf.to_container(cfg, resolve=True)
    config['output_dir'] = output_dir
    yaml.dump(config, open('config.yaml', 'w'), default_flow_style=False)
    if not cfg.offline:
        wandb.config.update(config)
    shutil.copyfile(
        to_absolute_path(cfg.env.vision.projector.calib_path), 
        'calibration.pkl')

    # load setup
    goal_pixs = list(cfg.setup.goal_pixs)

    # load enviroment
    with ZedCamera() as camera:
        env = Ur5WhipEnvironment(env_cfg=cfg.env, camera=camera)
        # load action model
        device = torch.device('cuda', cfg.action.gpu_id)
        dtype = torch.float16 if cfg.action.use_fp16 else torch.float32
        sampler = DeltaActionGaussianSampler(**cfg.action.sampler)
        action_model = DeltaTrajectoryDeeplab.load_from_checkpoint(
            to_absolute_path(cfg.action.ckpt_path))
        action_model_gpu = action_model.to(
            device, dtype=dtype).eval()
        selector = DeltaActionSelector(
            model=action_model_gpu, **cfg.action.selector)

        # load init action
        ic = cfg.action.init_action
        root = zarr.open(to_absolute_path(ic.zarr_path), 'r')
        init_action_arr = root[ic.name][:] / ic.action_scale
        video_threads = list()
        with env:
            env.reset()
            goals_log = list()
            for goal_pix in tqdm(goal_pixs):
                env.set_goal_pix(goal_pix)
                # experiment
                init_action = init_action_arr[tuple(goal_pix)]
                const_action = ic.get('const_action', None)
                if const_action is not None:
                    init_action = np.array(const_action)
                action = init_action
                steps_log = list()
                # obs_log = list()
                min_dist = float('inf')
                for step_id in tqdm(range(cfg.setup.n_steps)):
                    print('action', action)
                    observation, dist_to_goal_m, _, info = env.step(
                        action)

                    # log video
                    if cfg.setup.save_video:
                        fname = 'videos/' + '_'.join(str(x) for x in goal_pix) + '_' + str(step_id) + '.mp4'
                        vt = Thread(target=save_video, args=(fname, info['images']))
                        vt.start()
                        video_threads.append(vt)

                    min_dist = min(min_dist, dist_to_goal_m)
                    # action = info['action']
                    sigma = cfg.action.constant_sigma
                    if sigma is None:
                        sigma = min(dist_to_goal_m * cfg.action.gain, cfg.action.sigma_max)
                    
                    ts = cfg.action.threshold
                    threshold_interp = interp1d(
                        x=[0, ts.dist_max], y=[ts.max, ts.min], 
                        kind='linear',
                        bounds_error=False,
                        fill_value=(ts.max, ts.min))
                    threshold = threshold_interp(min(dist_to_goal_m, ts.dist_max))
                    
                    delta_action_samples = sampler.get_delta_action_samples(
                        action, sigma=sigma)
                    selection_result = selector.get_delta_action(
                        traj_img=observation, goal_pix=goal_pix, 
                        delta_action_samples=delta_action_samples,
                        threshold=threshold)
                    best_delta_action = selection_result['best_delta_action']

                    # logging
                    # vis trajectory and goal
                    images = info['images']
                    tracking_result = info['tracking_result']
                    projector = env.projector
                    tracking_vis = vis_tracking_goal(
                        images, tracking_result['keypoints'],
                        confidence=tracking_result['confidence'],
                        projector=projector,
                        goal_pix=goal_pix,
                        threshold=cfg.env.vision.projector.confidence_threshold
                    )
                    # vis action
                    best_prob = selection_result['best_prob']
                    best_mask = selection_result['best_mask']
                    action_vis = vis_action(best_prob, best_mask, observation, goal_pix)
                    
                    log = {
                        'tracking_vis': tracking_vis,
                        'action_vis': action_vis,
                        'error': dist_to_goal_m,
                        'k_min_error': min_dist,
                        'step_id': step_id
                    }
                    other_log = {
                        'action': action,
                        'observation': observation,
                        'keypoints': tracking_result['keypoints'],
                        'confidence': tracking_result['confidence'],
                        'goal_pix': np.array(goal_pix),
                        'sigma': sigma,
                        'threshold': threshold
                    }
                    if not cfg.offline:
                        wandb_log = dict()
                        this_key = 'goal_'+'_'.join(str(x) for x in goal_pix)
                        for key, value in log.items():
                            if key.endswith('vis'):
                                value = wandb.Image(value)
                            elif key.endswith('error'):
                                value = value * 100
                            wandb_log[this_key + '/' + key] = value
                        wandb.log(wandb_log)
                    log.update(other_log)
                    steps_log.append(log)
                    # next
                    action = best_delta_action + action
                # aggregate
                steps_log = transpose_data_dict(steps_log)
                goals_log.append(steps_log)
            rope_log = transpose_data_dict(goals_log)
            pickle.dump(rope_log, open('rope_log.pkl', 'wb'))
            if not cfg.offline:
                log = get_error_plots_log('all', rope_log['error'])
                wandb.log(log)
        for vt in video_threads:
            vt.join()

# %%
if __name__ == '__main__':
    main()

# %%
