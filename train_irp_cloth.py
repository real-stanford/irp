# %%
# import
import os
import pathlib
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from datasets.cloth_delta_gaussian_dataset import ClothDeltaGaussianDataModule
from networks.cloth_delta_deeplab import ClothDeltaDeeplab
from pl_vis.image_grid_callback import ImageGridCallback


# %%
# main script
@hydra.main(config_path="config", config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    print(os.getcwd())
    os.mkdir("checkpoints")

    datamodule = ClothDeltaGaussianDataModule(**cfg.datamodule)
    cfg.model.action_sigma = cfg.datamodule.action_sigma
    model = ClothDeltaDeeplab(**cfg.model)

    logger = pl.loggers.WandbLogger(
        project=os.path.basename(__file__),
        **cfg.logger)
    wandb_run = logger.experiment
    wandb_meta = {
        'run_name': wandb_run.name,
        'run_id': wandb_run.id
    }
    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
        'wandb': wandb_meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)

    datamodule.prepare_data()
    val_dataset = datamodule.get_dataset('val')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=5,
        mode='min', 
        save_weights_only=False, 
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    vis_callback = ImageGridCallback(
        val_dataset,
        **cfg.vis_callback
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, vis_callback],
        checkpoint_callback=True,
        logger=logger, 
        **cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)

# %%
# driver
if __name__ == "__main__":
    main()
