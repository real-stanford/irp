import numpy as np
from scipy.interpolate import interp1d
from common.trapezoid import get_trapezoid_phase_profile


def get_movej_trajectory(j_start, j_end, acceleration, speed, dt=0.001):
    assert(acceleration > 0)
    assert(speed > 0)

    j_delta = j_end - j_start
    j_delta_abs = np.abs(j_delta)
    j_delta_max = np.max(j_delta_abs)

    if j_delta_max != 0:
        # compute phase parameters
        phase_vel = speed / j_delta_max
        phase_acc = acceleration * (phase_vel)

        phase = get_trapezoid_phase_profile(dt=dt,
            speed=phase_vel, acceleration=phase_acc)
        
        interp = interp1d([0,1], [j_start, j_end], 
            axis=0, fill_value='extrapolate')
        j_traj = interp(phase)
    else:
        j_traj = np.array([j_start, j_end])

    assert(np.allclose(j_traj[0], j_start))
    assert(np.allclose(j_traj[-1], j_end))
    return j_traj
