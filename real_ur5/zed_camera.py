import numpy as np
import pyzed.sl as sl
import time

class ZedCamera:    
    def __init__(self, 
            init_params=None, 
            video_settings=None, 
            left=True,
            right=False):
        if init_params is None:
            init_params = self.get_default_init_params()
        if video_settings is None:
            video_settings = self.get_default_video_settings()
        assert(left or right)

        self.init_params = init_params
        self.video_settings = video_settings
        self.left = left
        self.right = right
        self.camera = None
        self.mat = sl.Mat()

    @staticmethod
    def get_default_init_params():
        return sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            camera_fps=60,
            depth_mode=sl.DEPTH_MODE.NONE,
            sdk_verbose=True,
            sdk_gpu_id=0,
            enable_image_enhancement=True
        )
    
    @staticmethod
    def get_default_video_settings():
        # return {
        #     'BRIGHTNESS': 4,
        #     'CONTRAST': 4,
        #     'HUE': 0,
        #     'SATURATION': 4,
        #     'SHARPNESS': 4,
        #     'GAMMA': 9,
        #     'GAIN': 100,
        #     'EXPOSURE': 6,
        #     'WHITEBALANCE_TEMPERATURE': 3200,
        #     'LED_STATUS': 1
        # }
        return {
            'BRIGHTNESS': 4,
            'CONTRAST': 4,
            'HUE': 0,
            'SATURATION': 4,
            'SHARPNESS': 4,
            'GAMMA': 6,
            'GAIN': 100,
            'EXPOSURE': 9,
            'WHITEBALANCE_TEMPERATURE': 4200,
            'LED_STATUS': 1
        }
    
    def is_opened(self):
        return self.camera.is_opened()
    
    def __enter__(self):
        zed = sl.Camera()
        err = zed.open(self.init_params)
        if (err != sl.ERROR_CODE.SUCCESS):
            print("Unable to start camera.")
            return

        time.sleep(0.5)
        for key, value in self.video_settings.items():
            zed.set_camera_settings(getattr(sl.VIDEO_SETTINGS, key), value)

        self.camera = zed
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.camera.close()
    
    def get_image(self) -> np.ndarray:
        """
        Return ndarray
        (N,H,W,4) uint8 bgra
        """
        err = self.camera.grab()
        if (err == sl.ERROR_CODE.SUCCESS):
            imgs = list()
            if self.left:
                self.camera.retrieve_image(self.mat, sl.VIEW.LEFT)
                imgs.append(self.mat.get_data())
            if self.right:
                self.camera.retrieve_image(self.mat, sl.VIEW.RIGHT)
                imgs.append(self.mat.get_data())
            return np.array(imgs)
        return None
