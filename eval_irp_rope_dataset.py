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

from common.sample_util import GridCoordTransformer, transpose_data_dict
from environments.dataset_environment import DatasetEnvironment
from environments.goal_selection import select_rope_and_goals
from common.wandb_util import get_error_plots_log


from networks.delta_trajectory_deeplab import DeltaTrajectoryDeeplab
from real_ur5.delta_action_sampler import DeltaActionGaussianSampler
from real_ur5.delta_action_selector import DeltaActionSelector


# %%
@hydra.main(config_path="config", config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig) -> None:
    if not cfg.offline:
        wandb.init(**cfg.wandb)
    abs_zarr_path = to_absolute_path(cfg.setup.zarr_path)
    rope_goal_dict = select_rope_and_goals(
        zarr_path=abs_zarr_path, 
        **cfg.setup.selection)
    config = OmegaConf.to_container(cfg, resolve=True)
    output_dir = os.getcwd()
    config['output_dir'] = output_dir
    yaml.dump(config, open('config.yaml', 'w'), default_flow_style=False)
    if not cfg.offline:
        wandb.config.update(config)

    root = zarr.open(abs_zarr_path, 'r')
    init_action_array = root['train_rope/best_action_coord']
    action_scale = np.array(root[cfg.setup.name].shape[2:5])

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

    transformer = GridCoordTransformer.from_zarr(abs_zarr_path)
    # load action model
    ropes_log = list()
    for rope_id, goals in rope_goal_dict.items():
        env = DatasetEnvironment(
            zarr_path=abs_zarr_path,
            name=cfg.setup.name,
            rope_id=rope_id,
            transformer=transformer,
            random_init=cfg.setup.random_init)
        goals_log = list()
        for goal_pix in tqdm(goals):
            env.set_goal_pix(goal_pix)
            # experiment
            init_action = init_action_array[tuple(goal_pix)] / action_scale
            action = init_action
            # action = np.random.uniform(0,1,3)
            steps_log = list()
            # obs_log = list()
            for step_id in range(cfg.setup.n_steps):
                observation, dist_to_goal_m, _, info = env.step(
                    action)
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
                steps_log.append({
                    'error': dist_to_goal_m,
                    'action': action
                })
                # next
                action = best_delta_action + action
                # action = np.random.uniform(0,1,3)
                # obs_log.append(observation)
            
            # aggregate
            steps_log = transpose_data_dict(steps_log)
            goals_log.append(steps_log)
        
        # aggregate data
        rope_log = transpose_data_dict(goals_log)
        rope_key = 'rope_' + '_'.join(str(x) for x in rope_id)
        log = get_error_plots_log(rope_key, rope_log['error'])
        if not cfg.offline:
            wandb.log(log)
        ropes_log.append(rope_log)
    all_logs = transpose_data_dict(ropes_log)
    errors = all_logs['error'].reshape(-1, all_logs['error'].shape[-1])
    log = get_error_plots_log('all', errors)
    if not cfg.offline:
        wandb.log(log)
    import pickle
    pickle.dump(all_logs, open('log.pkl', 'wb'))

# %%
if __name__ == '__main__':
    main()

# %%
