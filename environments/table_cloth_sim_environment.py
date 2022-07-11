# %%
from typing import Tuple
import pathlib
import numpy as np
import mujoco_py as mj
from scipy.interpolate import CubicSpline, interp1d
from jinja2 import Template

from common.cv_util import get_traj_occupancy
from common.sample_util import GridCoordTransformer, get_nd_index_volume
from common.mujoco_util import MujocoCompensatedPDController
from abr_control_mod.mujoco_utils import get_cloth_body_ids, get_body_center_of_mass

# %%
def get_cubic_control(t, q, dt) -> Tuple:
    """
    t: input time steps (T)
    q: input joint pos (T,Q)

    qs: output joint pos steps (N,Q)
    dqs: output joint vel steps (N,Q)
    ts: output time steps (N)
    """

    duration = t[-1]
    n_steps = int(duration / dt)
    q_interp = CubicSpline(t, q, bc_type='clamped')
    dq_interp = q_interp.derivative()
    ts = np.arange(n_steps) * dt
    qs = q_interp(ts)
    dqs = dq_interp(ts)
    return qs, dqs, ts

def nearest_interp1d(x, y):
    return interp1d(x, y, bounds_error=False, fill_value=(y[0],y[-1]))

class ActionMapper:
    def __init__(self, center_height=0.05):
        # Action parameters: (duration, gy1, gz1, gy2)
        # center_height is half the thickness of capsuls
        interps = [
            nearest_interp1d([0,1],[3,1.2]),
            nearest_interp1d([0,1],[0.3,1]),
            nearest_interp1d([0,1],[center_height,1]),
            nearest_interp1d([0,1],[0.3,1])
        ]
        self.interps = interps
    
    def __call__(self, action):
        out_action = np.array([interp(action[i]) 
            for i, interp in enumerate(self.interps)], 
            dtype=np.float32)
        return out_action

# %%
class TableClothSimEnvironment:
    def __init__(self, rope_config, controller_config, dt=0.01, max_steps=400, show_vis=False, obs_topdown=False):
        # init transformer
        tf = GridCoordTransformer((-0.1,-0.7),(1.8,1.1),(256,256))
        if obs_topdown:
            tf = GridCoordTransformer((-0.1,-0.7),(1.7,1.1),(256,256))
        topdown_tf = GridCoordTransformer((-0.9,-0.1),(0.9,1.7),(256,256))

        # load mujoco model
        xml_path = pathlib.Path(__file__).parent.parent.joinpath(
            'assets','mujoco','cloth','table_cloth_template.xml.jinja2')
        template = Template(xml_path.open('r').read())
        xml = template.render(**rope_config)
        model = mj.load_model_from_xml(xml)
        model.opt.timestep = dt

        # load body ids
        cloth_body_ids = get_cloth_body_ids(model)
        # pick 9 points
        coords = tuple([c.astype(np.int64) for c in np.meshgrid(*(np.linspace(0,x-1,3) 
            for x in cloth_body_ids.shape))])
        kp_ids = cloth_body_ids[coords].T.flatten()
        # 0 3 6
        # 1 4 7
        # 2 5 8

        # load simulation
        sim = mj.MjSim(model)
        sim.forward()
        # load controller
        ctrl = MujocoCompensatedPDController(
            sim=sim, **controller_config)
        init_state = ctrl._load_state()

        viewer = None
        if show_vis:
            viewer = mj.MjViewer(sim)

        self.sim = sim
        self.ctrl = ctrl
        self.kp_ids = kp_ids
        self.action_mapper = ActionMapper()
        self.dt = dt
        self.max_steps = max_steps
        self.init_state = init_state
        self.show_vis = show_vis
        self.viewer = viewer
        self.transformer = tf
        self.topdown_transformer = topdown_tf
        self.rope_config = rope_config
        self.loss_func = None
        self.obs_topdown = obs_topdown
    
    def get_cloth_goal(self, alpha: float):
        """
        alpha in [0,1]
        """
        rope_config = self.rope_config
        cloth_size = rope_config['cloth_spacing'] * 12
        table_size = rope_config['table_size']
        table_height = 0
        table_y_start = rope_config['table_y'] - table_size / 2
        capsule_r = rope_config['cloth_spacing'] * 0.2
        max_reach = 1
        offset_range = min(table_size - cloth_size, max_reach - table_y_start)
        offset = offset_range * np.clip(alpha, 0, 1)

        base_coords = get_nd_index_volume((3,3,1))[:,:,0,:].astype(np.float32) / 2 * cloth_size
        rope_coords = base_coords.copy()
        rope_coords[:,:,0] -= cloth_size / 2
        rope_coords[:,:,1] += offset + table_y_start
        rope_coords[:,:,2] = table_height + capsule_r
        goal_coords = rope_coords.reshape(-1,3)
        return goal_coords
    
    def get_traj_loss_func(self, goal, measure_dims=[0,1,2], **kwargs):
        def loss_func(traj):
            traj_points = traj[-1]
            diff = traj_points - goal
            dists = np.linalg.norm(diff[...,measure_dims], axis=-1)
            mean_dist = np.mean(dists)
            return mean_dist
        return loss_func
    
    def get_img_loss_func(self, goal, measure_dims=[0,1], **kwargs):
        def loss_func(traj_imgs):
            transformer = self.transformer
            pix_per_m = transformer.pix_per_m
            goal_pixs = transformer.to_grid(goal[:,[1,2]])
            dists = list()
            for i, goal_pix in enumerate(goal_pixs):
                mask = traj_imgs[i]
                dist = float('inf')
                if mask.sum() > 0:
                    img_coords = get_nd_index_volume(mask.shape)
                    diff = img_coords - goal_pix
                    dist_img = np.linalg.norm(
                        diff[...,measure_dims], axis=-1)
                    dist = dist_img[mask].min()
                dists.append(dist)
            mean_dist = np.mean(dists) / pix_per_m
            return mean_dist
        return loss_func

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

    def step(self, action: np.ndarray, wait: int=0) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Action parameters: (duration, gy1, gz1, gy2)
        """
        # generate control
        dt = self.dt
        raw_action = self.action_mapper(action)
        duration, gy1, gz1, gy2 = raw_action
        gz2 = 0.05
        t_in = np.linspace(0, duration, 3)
        q_in = np.array([
            [0, 0],
            [gy1, gz1],
            [gy2, gz2]
        ])
        qs, dqs, ts = get_cubic_control(t_in, q_in, dt)
        pad_steps = int(0.2 / dt)

        # simulate 
        self.ctrl._load_state(*self.init_state)
        n_steps = min(self.max_steps, len(qs) + pad_steps + 20)
        hist = list()
        for i in range(n_steps + wait):
            ii = max(min(i - wait, (len(qs)-1)), 0)
            q = qs[ii]
            dq = dqs[ii]
            u = self.ctrl.generate(q, dq)
            self.ctrl.send_forces(u)
            self.sim.step()
            if self.show_vis:
                self.viewer.render()
            # record
            kp_com = get_body_center_of_mass(self.sim.data, self.kp_ids)
            hist.append(kp_com)
        hist = np.array(hist)

        # save vis
        imgs = list()
        for i in range(hist.shape[1]):
            traj = hist[:,i,[1,2]]
            img = get_traj_occupancy(traj, 
                transformer=self.transformer)
            imgs.append(img)
        if self.obs_topdown:
            for i in range(hist.shape[1]):
                traj = hist[:,i,[0,1]]
                img = get_traj_occupancy(traj, 
                    transformer=self.topdown_transformer)
                imgs.append(img)

        obs = np.array(imgs)
        hist_pix = self.transformer.to_grid(hist[...,[1,2]])

        # compute error
        loss = None
        if self.loss_func is not None:
            loss = self.loss_func(hist)

        # return
        observation = obs
        # loss = None
        done = False
        info = {
            'raw_action': raw_action,
            'trajectory': hist,
            'trajectory_pix': hist_pix
        }
        return observation, loss, done, info

