import time
import numpy as np
from real_ur5.zed_camera import ZedCamera
import pyzed.sl as sl
from threading import Thread, Event
import collections


class ZedRamRecorder(Thread):
    def __init__(self, camera: ZedCamera, maxlen=6000):
        super().__init__()
        assert(camera.is_opened())
        self.camera = camera
        self.deque = collections.deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.counter = 0
        self.stop_event = Event()
    
    def is_done(self):
        return not((not self.stop_event.is_set()) and (self.counter < self.maxlen))
    
    def run(self):
        while not self.is_done():
            img = self.camera.get_image()
            if img is not None:
                self.deque.append(img)
                self.counter += 1
            time.sleep(0)

    def stop(self, blocking=True):
        self.stop_event.set()
        if blocking:
            self.join()
    
    def __del__(self):
        self.stop(blocking=False)


class SVORamRecorder(Thread):
    def __init__(self, 
            svo_path,
            left=True,
            right=False):
        super().__init__()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        camera = sl.Camera()
        assert(camera.open(init_params) == sl.ERROR_CODE.SUCCESS)
        self.camera = camera
        self.deque = collections.deque()
        self.stop_event = Event()
        self.left = left
        self.right = right
        self.mat = sl.Mat()
    
    def is_done(self):
        return self.stop_event.is_set()
    
    def run(self):
        camera = self.camera
        while not self.is_done():        
            err = camera.grab()
            if err != sl.ERROR_CODE.SUCCESS:
                self.stop_event.set()
                break
            imgs = list()
            if self.left:
                self.camera.retrieve_image(self.mat, sl.VIEW.LEFT)
                imgs.append(self.mat.get_data())
            if self.right:
                self.camera.retrieve_image(self.mat, sl.VIEW.RIGHT)
                imgs.append(self.mat.get_data())
            self.deque.append(np.array(imgs))
        
    def stop(self, blocking=True):
        self.stop_event.set()
        if blocking:
            self.join()

    def __del__(self):
        self.stop(blocking=False)
