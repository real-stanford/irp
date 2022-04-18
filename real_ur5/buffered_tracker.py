import time
import numpy as np
from threading import Thread

import torch
from real_ur5.zed_ram_recorder import ZedRamRecorder
from components.tracking import KeypointTracker
from common.torch_util import to_numpy

class BufferedTracker(Thread):
    def __init__(self, 
            recorder: ZedRamRecorder, 
            tracker: KeypointTracker, 
            max_batch_size=4):
        super().__init__()
        assert(max_batch_size > 0)

        self.recorder = recorder
        self.tracker = tracker
        self.max_batch_size = max_batch_size
        # external facing
        self.imgs_list = list()
        self.keypoints_list = list()
        self.confidence_list = list()
    
    def is_done(self):
        return self.recorder.is_done() and (len(self.recorder.deque) == 0)
    
    def run(self):
        self.recorder.start()
        while not self.is_done():
            batch_size = min(self.max_batch_size, len(self.recorder.deque))
            if batch_size > 0:
                batch_list = list()
                for i in range(batch_size):
                    batch_list.append(self.recorder.deque.popleft())
                batch = np.concatenate(batch_list, axis=0)
                keypoints, confidence = self.tracker(self.tracker.process_input(batch))
                self.imgs_list.append(batch)
                self.keypoints_list.append(keypoints)
                self.confidence_list.append(confidence)
            time.sleep(0)
        # wait for kernels to finish
        # important to get correct output
        torch.cuda.synchronize(self.tracker.device)

    def stop(self):
        self.recorder.stop(blocking=False)

    def get_images(self):
        self.join()
        return np.concatenate(self.imgs_list, axis=0)
    
    def get_tracking(self):
        self.join()
        result = {
            'keypoints': to_numpy(torch.cat(self.keypoints_list, dim=0)),
            'confidence': to_numpy(torch.cat(self.confidence_list, dim=0))
        }
        return result
