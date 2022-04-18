import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms


class KeypointTracker(pl.LightningModule):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.transform = transforms.Normalize(
            (0.485, 0.456, 0.406), 
            (0.229, 0.224, 0.225))
        self.predictor.eval()
    
    def process_input(self, input: np.ndarray, non_blocking=True):
        """
        Assuming input is a uint8 ndarray 0-255 with bgra format
        N,H,W,C
        """
        with torch.no_grad():
            return torch.from_numpy(input)[:,:,:,[2,1,0]].to(
                    device=self.device, dtype=self.dtype, 
                    non_blocking=non_blocking
                    ).moveaxis(-1,1).contiguous() / 255

    def forward(self, x: torch.Tensor):
        """
        Return coordinates, confidence
        """
        with torch.no_grad():
            N,_,H,W = x.shape
            x = self.predictor(self.transform(x))['scoremap']
            C = x.shape[1]
            confidence, x = torch.max(x.reshape(N,C,-1), axis=-1)
            confidence = torch.sigmoid(confidence)
            keypoints = torch.zeros((N,C,2), device=self.device, dtype=torch.int64)
            keypoints[:,:,1] = torch.divide(x, W, rounding_mode='floor')
            keypoints[:,:,0] = x % W
            return keypoints, confidence
