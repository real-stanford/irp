import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import Subset, DataLoader
import wandb
import numpy as np
from common.torch_util import dict_to


def stack_to_grid(stack, grid_shape):
    rows = list()
    for i in range(grid_shape[0]):
        row = list()
        for j in range(grid_shape[1]):
            idx = i * grid_shape[1] + j
            img = stack[idx]
            row.append(img)
        rows.append(torch.cat(row, axis=1))
    img = torch.cat(rows, axis=0)
    return img


class ImageGridCallback(pl.Callback):
    def __init__(self, 
            val_dataset, 
            input_key='occupancy', 
            output_key=None,
            grid_shape=(3,3),
            num_samples=16, 
            seed=0,
            sigmoid=True):
        super().__init__()
        rs = np.random.RandomState(seed=seed)
        vis_idxs = rs.choice(len(val_dataset), num_samples)
        vis_subset = Subset(val_dataset, vis_idxs)
        vis_dataloader = DataLoader(
            vis_subset, batch_size=num_samples)
        vis_batch = next(iter(vis_dataloader))
        self.vis_batch = vis_batch
        self.vis_idxs = vis_idxs
        self.input_key = input_key
        self.output_key = output_key
        self.sigmoid = sigmoid
        self.grid_shape = grid_shape
        
    def on_validation_epoch_end(self, 
            trainer: pl.Trainer, 
            pl_module: pl.LightningModule) -> None:
        vis_batch_device = dict_to(
            self.vis_batch,
            device=pl_module.device)        
        result = pl_module(**vis_batch_device)
        logits = result
        if self.output_key is not None:
            logits = result[self.output_key]
        probs = logits
        if self.sigmoid:
            probs = torch.sigmoid(logits)        
        probs_cpu = probs.detach().to('cpu')
        
        imgs = list()
        for batch_idx in range(len(self.vis_idxs)):
            vis_idx = self.vis_idxs[batch_idx]
            gt_img = stack_to_grid(self.vis_batch[self.input_key][batch_idx], self.grid_shape)
            pred_img = stack_to_grid(probs_cpu[batch_idx], self.grid_shape)
            img_pair = torch.cat([gt_img, pred_img], dim=1)
            imgs.append(wandb.Image(
                img_pair, caption=f"val-{vis_idx}"
            ))
        
        trainer.logger.experiment.log({
            "val/vis": imgs,
            "global_step": trainer.global_step
        })
