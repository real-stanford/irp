import inspect
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import Subset, DataLoader
import wandb
import numpy as np
from common.torch_util import dict_to


class ImagePairCallback(pl.Callback):
    def __init__(self, 
            val_dataset, input_key='occupancy', 
            output_key=None,
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
        
    def on_validation_epoch_end(self, 
            trainer: pl.Trainer, 
            pl_module: pl.LightningModule) -> None:
        vis_batch_device = dict_to(
            self.vis_batch,
            device=pl_module.device)
        func_args = set(inspect.signature(pl_module.forward).parameters.keys())
        kwargs = dict([(x, vis_batch_device[x]) for x in func_args])
        result = pl_module(**kwargs)
        logits = result
        if self.output_key is not None:
            logits = result[self.output_key]
        probs = logits
        if self.sigmoid:
            probs = torch.sigmoid(logits)        
        probs_cpu = probs.detach().to('cpu')
        
        imgs = list()
        for i in range(len(self.vis_idxs)):
            vis_idx = self.vis_idxs[i]
            gt_img = self.vis_batch[self.input_key][i]
            pred_img = probs_cpu[i]
            img_pair = torch.cat([gt_img, pred_img], dim=1)
            imgs.append(wandb.Image(
                img_pair, caption=f"val-{vis_idx}"
            ))
        
        trainer.logger.experiment.log({
            "val/vis": imgs,
            "global_step": trainer.global_step
        })
