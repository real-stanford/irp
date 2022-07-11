import torch
import torch.nn as nn
import pytorch_lightning as pl

from components.deeplab_v3_plus import DeepLabv3_feature, OutConv

class ClothDeltaDeeplab(pl.LightningModule):
    def __init__(self,
            action_sigma,
            n_channels=9,
            action_dim=4,
            n_features=64,
            learning_rate=1e-3,
            weight_decay=None,
            loss='bce',
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.net = DeepLabv3_feature(
            n_channels=n_channels+action_dim, 
            n_features=n_features, os=8)
        self.outc = OutConv(
            in_channels=n_features, 
            out_channels=n_channels)

        self.criterion = None
        if loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise RuntimeError("Invalid loss type")

        self.action_sigma = action_sigma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        learning_rate = self.learning_rate
        weight_decay = self.weight_decay
        optimizer = None
        if weight_decay is None:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=learning_rate, 
                weight_decay=weight_decay)
        return optimizer

    def forward(self, input_trajectory, action_delta, **kwargs):
        action_delta_ = action_delta / self.action_sigma
        exploded_action_delta = action_delta_.reshape(
            *action_delta_.shape,1,1).expand(
                -1,-1,*input_trajectory.shape[2:])
        x = torch.cat([input_trajectory, exploded_action_delta], dim=1)
        features = self.net(x)
        logits = self.outc(features)
        return logits
    
    def step(self, batch, batch_idx, step_type='train'):
        input_trajectory = batch['input_trajectory']
        action_delta = batch['action_delta']
        target_trajectory = batch['target_trajectory']
        logits = self.forward(input_trajectory, action_delta)
        loss = self.criterion(logits, target_trajectory)

        self.log(f"{step_type}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
