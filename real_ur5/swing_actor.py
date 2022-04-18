import time
import numpy as np
import scipy.interpolate as si
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


def deg_to_rad(deg):
    return deg / 180 * np.pi


class SwingActor:
    def __init__(self, 
            robot_ip='192.168.0.139',
            q_init=(-90,-70,150,-170,-90,-90), 
            speed_range=(1.0,3.14),
            j2_delta_range=(-60,-180),
            j3_delta_range=(60,-120),
            tcp_offset=0.5) -> None:
        self.q_init = deg_to_rad(np.array(q_init))
        self.speed_func = si.interp1d((0,1),speed_range)
        self.j2_delta_func = si.interp1d(
            (0,1), deg_to_rad(np.array(j2_delta_range)))
        self.j3_delta_func = si.interp1d(
            (0,1), deg_to_rad(np.array(j3_delta_range)))

        tcp_offset_pose = [0.0]*6
        tcp_offset_pose[2] = tcp_offset
        self.tcp_offset_pose = tcp_offset_pose
        self.robot_ip = robot_ip
        self.rtde_c = None
        self.rtde_r = None
    
    def __enter__(self):
        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.rtde_c.setTcp(self.tcp_offset_pose)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
    
    def get_action(self, spd, j2, j3):
        this_speed = self.speed_func(spd)
        this_q_goal = self.q_init.copy()
        this_q_goal[2] += self.j2_delta_func(j2)
        this_q_goal[3] += self.j3_delta_func(j3)
        return this_speed, this_q_goal
    
    def reset(self, acc=10.0, speed=3.0, blocking=True):
        self.rtde_c.moveJ(
            self.q_init.tolist(), 
            speed, acc, not blocking)
    
    def swing(self, spd, j2, j3, acc=10.0, blocking=True):
        speed, q_goal = self.get_action(spd, j2, j3)
        self.rtde_c.moveJ(
            q_goal.tolist(), 
            speed, acc, not blocking)

