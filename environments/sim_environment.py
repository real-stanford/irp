from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import interp1d
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.controllers import Joint
from abr_control.interfaces.mujoco import Mujoco

from abr_control_mod.mujoco_utils import (
    get_rope_body_ids, get_body_center_of_mass,
    apply_impulse_com_batch, get_mujoco_state, set_mujoco_state)
from common.urscript_control_util import get_movej_trajectory
from common.template_util import require_xml
from common.cv_util import get_traj_occupancy
from common.sample_util import (GridCoordTransformer, get_nd_index_volume)


def deg_to_rad(deg):
    return deg / 180 * np.pi

def get_param_dict(length, density, 
        width=0.02, stick_length=0.48, 
        num_nodes=25, **kwargs):
    stiffness = 0.01 / 0.015 * density
    damping = 0.005 / 0.015 * density

    link_size = length / (num_nodes - 1) / 2
    param_dict = {
        'count': num_nodes,
        'spacing': link_size * 2,
        'link_size': link_size,
        'link_width': width / 2,
        'link_mass': density * length / num_nodes,
        'stiffness': stiffness,
        'damping': damping,
        'stick_size': (stick_length - link_size) / 2,
        'ee_offset': stick_length
    }
    return param_dict

class SimEnvironment:
    def __init__(self, env_cfg: DictConfig, rope_cfg: DictConfig):
        self.env_cfg = env_cfg

        # create transformer
        transformer = GridCoordTransformer(**env_cfg.transformer)

        # build simulation xml
        xml_dir = to_absolute_path(env_cfg.xml_dir)
        rope_param_dict = get_param_dict(**rope_cfg)
        xml_fname = require_xml(
            xml_dir, rope_param_dict, 
            to_absolute_path(env_cfg.template_path), force=True)
        
        # load mujoco environment
        robot_config = MujocoConfig(
            xml_file=xml_fname, 
            folder=xml_dir)
        interface = Mujoco(robot_config, 
            dt=env_cfg.sim.dt, 
            visualize=env_cfg.sim.visualize)
        interface.connect()
        ctrlr = Joint(robot_config, kp=env_cfg.sim.kp)

        j_init = deg_to_rad(np.array(env_cfg.sim.j_init_deg))
        interface.set_joint_state(q=j_init, dq=np.zeros(6))
        init_state = get_mujoco_state(interface.sim)
        rope_body_ids = get_rope_body_ids(interface.sim.model)

        self.transformer = transformer
        self.interface = interface
        self.ctrlr = ctrlr
        self.init_state = init_state
        self.rope_body_ids = rope_body_ids
        self.rs = np.random.RandomState(env_cfg.seed)

    def set_goal(self, goal: Tuple[float, float]):
        self.goal_pix = tuple(self.transformer.to_grid([goal], clip=True)[0])

    def set_goal_pix(self, goal_pix: Tuple[int, int]):
        self.goal_pix = tuple(goal_pix)

    def step(self, action: np.ndarray
            ) -> Tuple[np.ndarray, float, bool, dict]:
        interface = self.interface
        rope_body_ids = self.rope_body_ids
        ctrlr = self.ctrlr
        init_state = self.init_state

        if self.goal_pix is None:
            raise RuntimeError('Please call set_goal before step.')

        eps = 1e-7
        action = np.clip(action, 0, 1-eps)
        # compute action
        ac = self.env_cfg.action
        speed_interp = interp1d([0,1], ac.speed_range)
        j2_interp = interp1d([0,1], ac.j2_delta_range)
        j3_interp = interp1d([0,1], ac.j3_delta_range)
        speed = speed_interp(action[0])
        j2_delta = j2_interp(action[1])
        j3_delta = j3_interp(action[2])
        impulse = 0
        if self.env_cfg.random_init:
            impulse = self.rs.uniform(*ac.impulse_range)

        # generate target
        sc = self.env_cfg.sim
        j_init = deg_to_rad(np.array(sc.j_init_deg))
        j_start = j_init
        j_end = j_init.copy()
        j_end[2] += j2_delta
        j_end[3] += j3_delta

        q_target = get_movej_trajectory(
            j_start=j_start, j_end=j_end, 
            acceleration=ac.acceleration, speed=speed, dt=sc.dt)
        qdot_target = np.gradient(q_target, sc.dt, axis=0)

        # run simulation
        set_mujoco_state(interface.sim, init_state)
        
        impulses = np.multiply.outer(
            np.linspace(0, 1, len(rope_body_ids)),
            np.array([impulse, 0, 0]))
        apply_impulse_com_batch(
            sim=interface.sim, 
            body_ids=rope_body_ids, 
            impulses=impulses)

        num_sim_steps = int(sc.sim_duration / sc.dt)
        rope_history = list()
        n_contact_buffer = [0]
        for i in range(num_sim_steps):
            feedback = interface.get_feedback()

            idx = min(i, len(q_target)-1)
            u = ctrlr.generate(
                q=feedback['q'],
                dq=feedback['dq'],
                target=q_target[idx],
                target_velocity=qdot_target[idx]
            )
            n_contact = interface.sim.data.ncon
            n_contact_buffer.append(n_contact)
            if i % sc.subsample_rate == 0:
                nc = max(n_contact_buffer)
                if nc > 0:
                    break
                rope_body_com = get_body_center_of_mass(
                    interface.sim.data, rope_body_ids)
                rope_history.append(rope_body_com[-1])
                n_contact_buffer = [0]
            interface.send_forces(u)

        this_data = np.array(rope_history, dtype=np.float32)
        traj_img = get_traj_occupancy(this_data[:,[0,2]], self.transformer)

        img_coords = get_nd_index_volume(traj_img.shape)
        traj_coords = img_coords[traj_img]
        dists_pix = np.linalg.norm(traj_coords - self.goal_pix, axis=-1)
        dist_pix = np.min(dists_pix)
        dist_m = dist_pix / self.transformer.pix_per_m

        # return
        observation = traj_img
        loss = dist_m
        done = False
        info = {
            'action': action
        }
        return observation, loss, done, info
