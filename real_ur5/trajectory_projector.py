import numpy as np

from common.sample_util import GridCoordTransformer
from common.cv_util import get_traj_occupancy
from common.geometry_util import homo_transform

class TrajectoryProjector:
    def __init__(self, 
            tx_img_robot,
            transformer=None, 
            flip_x=True,
            confidence_threshold=0.3,
            **kwargs):
        if transformer is None:
            transformer = self.get_default_transformer()
        self.flip_x=flip_x
        self.transformer = transformer
        self.tx_img_robot = tx_img_robot
        self.tx_robot_img = np.linalg.inv(tx_img_robot)
        self.confidence_threshold = confidence_threshold
    
    @staticmethod
    def get_default_transformer():
        img_shape = (256, 256)
        transformer = GridCoordTransformer(
            (-3.0,-3.0),(3.0,3.0),img_shape)
        return transformer
    
    def to_robot_frame(self, keypoints):
        return homo_transform(self.tx_robot_img, keypoints)
    
    def robot_to_image(self, keypoints):
        return homo_transform(self.tx_img_robot, keypoints)
    
    def grid_to_image(self, grid_points):
        sim_points = self.transformer.from_grid(grid_points)
        if self.flip_x:
            sim_points[:,0] *= -1
        return homo_transform(self.tx_img_robot, sim_points)
    
    def get_sim_traj_img(self, tracking_result):
        keypoints = tracking_result['keypoints'][:,0,:]
        confidence = tracking_result['confidence'][:,0]
        mask = confidence > self.confidence_threshold
        valid_keypoints = keypoints[mask]

        sim_points = self.to_robot_frame(valid_keypoints)
        if self.flip_x:
            # robot setup is flipped with simulation
            sim_points[:,0] *= -1
        traj_img = get_traj_occupancy(sim_points, self.transformer)
        return traj_img
