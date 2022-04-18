# %%
# import
import os
import pathlib
import collections
import json

import numpy as np
from scipy.spatial.ckdtree import cKDTree

import yaml
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import wandb
from tqdm import tqdm

from networks.delta_trajectory_deeplab import DeltaTrajectoryDeeplab
from datasets.delta_trajectory_gaussian_dataset import DeltaTrajectoryGaussianDataModule
from common.sample_util import GridCoordTransformer

# %%
@hydra.main(config_path="config", config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig) -> None:
    if cfg.wandb.project is None:
        cfg.wandb.project = os.path.basename(__file__)
    wandb.init(**cfg.wandb)
    config = OmegaConf.to_container(cfg, resolve=True)
    output_dir = os.getcwd()
    config['output_dir'] = output_dir
    yaml.dump(config, open('config.yaml', 'w'), default_flow_style=False)
    wandb.config.update(config)

    datamodule = DeltaTrajectoryGaussianDataModule(**cfg.datamodule)
    datamodule.prepare_data()
    test_dataloader = datamodule.test_dataloader()
    transformer = GridCoordTransformer.from_zarr(
        to_absolute_path(cfg.datamodule.zarr_path))

    device = torch.device('cuda', cfg.model.gpu_id)
    model = DeltaTrajectoryDeeplab.load_from_checkpoint(
        to_absolute_path(cfg.model.ckpt_path))
    model = model.eval().to(device)

    coord_img = np.moveaxis(np.indices((256,256)),0,-1)
    coord_img = transformer.from_grid(coord_img)
    raw_metrics_dict = collections.defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_trajectory = batch['input_trajectory'].to(device)
            action_delta = batch['action_delta'].to(device)
            logits = model.forward(input_trajectory, action_delta)
            probs = torch.sigmoid(logits)

            gt_imgs = batch['target_trajectory'].numpy()
            pred_imgs = probs.detach().to('cpu').numpy()

            for i in range(gt_imgs.shape[0]):
                gt_points = coord_img[gt_imgs[i,0] > 0.5]
                pred_points = coord_img[pred_imgs[i,0] > cfg.model.threshold]
                gt_tree = cKDTree(gt_points)
                pred_tree = cKDTree(pred_points)

                gt_dists, _ = pred_tree.query(gt_points)
                pred_dists, _ = gt_tree.query(pred_points)

                metric = {
                    'gt_chamfer': np.mean(gt_dists),
                    'pred_chamfer': np.mean(pred_dists)
                }
                for key, value in metric.items():
                    raw_metrics_dict[key].append(value)
                    wandb.log(metric)
    
    # aggregate
    agg_metrics_dict = dict()
    for key, value in raw_metrics_dict.items():
        agg_metrics_dict[key] = np.mean(value)

    # save data
    json.dump(agg_metrics_dict, open('metrics.json','w'), indent=4)

# %%
if __name__ == '__main__':
    main()
