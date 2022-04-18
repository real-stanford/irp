import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import torchvision.models.segmentation.segmentation as tvs

from common.torch_util import to_numpy

class KeypointDeeplab(pl.LightningModule):
    def __init__(self, 
            learning_rate=1e-3,
            weight_decay=None, 
            upsample = False,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.deeplab_model = tvs.deeplabv3_resnet50(
            pretrained=False, 
            progress=False, 
            num_classes=1, 
            aux_loss=None)
        # delete last layer
        self.deeplab_model.classifier = nn.Sequential(*list(
            self.deeplab_model.classifier)[:-1])
        
        # deeplab v3 plus
        self.mid_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.last_conv = None
        if upsample:
            self.last_conv = nn.Sequential(
                nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        else:
            self.last_conv = nn.Sequential(
                nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.upsample = upsample
        self.criterion = nn.BCEWithLogitsLoss()
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
    
    def load_pretrained_weight(self):
        model_url = tvs.model_urls['deeplabv3_resnet50_coco']
        state_dict = tvs.load_state_dict_from_url(model_url, progress=True)
        redundent_keys = ['classifier.4.weight', 'classifier.4.bias']
        for key in redundent_keys:
            del state_dict[key]

        result = self.deeplab_model.load_state_dict(
            state_dict, strict=False)
        return result
    
    def backbone_forward(self, x):
        # hack to be able to use pre-trained model
        self = self.deeplab_model.backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        l1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        l4 = x

        result = {
            'l1': l1,
            'l4': l4
        }
        return result
    
    def forward(self, batch):
        input_shape = batch.shape[-2:]
        features = self.backbone_forward(batch)

        # 8x down
        features_out = features['l4']
        x = self.deeplab_model.classifier(features_out)

        # 4x down
        low_level_features = features['l1']
        low_level_features = self.mid_conv(low_level_features)

        x = F.interpolate(
            x, size=low_level_features.shape[2:], 
            mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        scoremap_raw = x
        
        if self.upsample:
            # scoremap_raw is 4x down
            scoremap = F.interpolate(
                scoremap_raw, size=batch.shape[2:], 
                mode='bilinear', align_corners=True)
        else:
            # scoremap_raw is 1x
            scoremap = scoremap_raw
        
        result = {
            'features': features,
            'scoremap_raw': scoremap_raw,
            'scoremap': scoremap
        }
        return result

    def step(self, batch, batch_idx, step_type='train'):
        input_img = batch['input_image']
        target_scoremap = batch['target_scoremap'] # (N,3,H,W)
        target_keypoint = batch['target_keypoint'] # (N,2) float32

        result = self.forward(input_img)

        scoremap = result['scoremap']
        loss = self.criterion(scoremap, target_scoremap)

        # compute keypoint distance
        target_keypoint_np = to_numpy(target_keypoint)
        pred_idx_np = to_numpy(torch.argmax(
            scoremap.reshape(scoremap.shape[0], -1), 
            dim=-1, keepdim=False))
        pred_keypoint_np = np.stack(np.unravel_index(
            pred_idx_np, shape=input_img.shape[2:])).T.astype(np.float32)[:,::-1]
        keypoint_dist = np.linalg.norm(pred_keypoint_np - target_keypoint_np, axis=-1)
        avg_keypoint_dist = np.mean(keypoint_dist)

        # logging
        self.log(f"{step_type}_loss", loss)
        self.log(f"{step_type}_keypoint_dist", avg_keypoint_dist)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')


