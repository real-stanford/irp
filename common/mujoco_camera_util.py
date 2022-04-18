"""
Make sure to run render once before executing following functions, so that 
offscreen rendering context is initialized
"""

import math
from mujoco_py.builder import cymj
import numpy as np
from scipy.spatial.transform import Rotation

def get_render_context(sim: cymj.MjSim) -> cymj.MjRenderContext:
    return sim._render_context_offscreen

def set_camera_pose(sim: cymj.MjSim, 
    lookat: np.ndarray, distance: float, 
    azimuth: float, elevation: float):
    ctx = get_render_context(sim)
    cam = ctx.cam
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

def normalize(v):
    return v / np.linalg.norm(v)

def lookAt(eye, center, up):
    ''' Matrix 4x4 lookAt function.'''
    f = normalize(center - eye)
    u = normalize(up)
    s = normalize(np.cross(f, u))
    u = np.cross(s, f)

    output = [[s[0], u[0], -f[0], 0.0],
              [s[1], u[1], -f[1], 0.0],
              [s[2], u[2], -f[2], 0.0],
              [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0]]
    return np.array(output).T

def azel_to_enu(azimuth, elevation):
    """
    Mujoco's Azimuth/Elevation is non-standard
    Def: ray from the camera
    Azimuth, angle from pos x, counter clockwise
    Elevation, angle from xy plane
    """
    # az_rad = azimuth/180*np.pi
    xy_rad = azimuth/180*np.pi
    ele_rad = elevation/180*np.pi
    z = np.sin(ele_rad)
    xy_norm = np.cos(ele_rad)
    x = np.cos(xy_rad) * xy_norm
    y = np.sin(xy_rad) * xy_norm
    return np.array([x,y,z])

def get_extrinsic_mat(sim: cymj.MjSim):
    """
    Mujoco uses OpenGL convesion for camera coords.
    -Z front, X right, Y up
    """
    ctx = get_render_context(sim)
    cam = ctx.cam
    center = np.array(cam.lookat)
    up = np.array([0,0,1])
    eye = center - (azel_to_enu(cam.azimuth, cam.elevation) * cam.distance)
    print(eye)
    ext_mat = lookAt(eye, center, up)
    return ext_mat
    
def get_intrinsic_mat(sim: cymj.MjSim, width: int, height: int):
    fovy = sim.model.vis.global_.fovy
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    int_mat = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
    return int_mat

def world_to_pixel(points, ext_mat, int_mat):
    """
    Output OpenCV keypoint coorindate.
    Lower left corner origin
    X right, Y up
    """
    gl2cv = np.eye(4)
    gl2cv[:3,:3] = Rotation.from_rotvec(np.array([np.pi,0,0])).as_matrix()
    this_ext_mat = gl2cv @ ext_mat
    cam_points = points @ this_ext_mat[:3,:3].T + this_ext_mat[:3,-1]
    pix_points = cam_points @ int_mat.T
    pixs = (pix_points[:,:2].T / pix_points[:,-1]).T
    return pixs
