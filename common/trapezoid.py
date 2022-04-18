import numpy as np


def get_trapezoid_phase_profile(
        dt=0.01,
        start_phase=0.0,
        end_phase=1.0,
        speed=1.0, 
        acceleration=1.0, 
        start_padding=0.0, 
        end_padding=0.0,
        dtype=np.float64):
    
    # calculate duration
    assert(end_phase > start_phase)
    total_travel = end_phase - start_phase

    t_cruise = None
    t_acc = None
    max_speed = None
    tri_max_speed = np.sqrt(acceleration * total_travel)
    if tri_max_speed <= speed:
        # triangle
        t_acc = total_travel / tri_max_speed
        t_cruise = 0
        max_speed = tri_max_speed
    else:
        # trapozoid
        t_acc = speed / acceleration
        tri_travel = t_acc * speed
        t_cruise = (total_travel - tri_travel) / speed
        max_speed = speed

    duration = start_padding + end_padding + t_acc * 2 + t_cruise
    key_point_diff_arr = np.array([
        start_padding, t_acc, t_cruise, t_acc, end_padding], dtype=dtype)
    key_point_time_arr = np.cumsum(key_point_diff_arr)

    all_time_steps = np.linspace(0, duration, int(np.ceil(duration / dt)), dtype=dtype)
    phase_steps = np.zeros_like(all_time_steps)

    # start_padding
    mask_idxs = np.flatnonzero(all_time_steps < key_point_time_arr[0])
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = start_phase
    
    # acceleartion
    mask_idxs = np.flatnonzero(
        (key_point_time_arr[0] <= all_time_steps) 
        & (all_time_steps <= key_point_time_arr[1]))
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = acceleration / 2 * np.square(
            all_time_steps[mask_idxs] - key_point_time_arr[0])
    acc_dist = acceleration / 2 * (t_acc ** 2)
    
    # cruise
    mask_idxs = np.flatnonzero(
        (key_point_time_arr[1] < all_time_steps) 
        & (all_time_steps < key_point_time_arr[2]))
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = max_speed * (
            all_time_steps[mask_idxs] - key_point_time_arr[1]) + acc_dist
    cruise_dist = t_cruise * max_speed

    # deceleration
    mask_idxs = np.flatnonzero(
        (key_point_time_arr[2] <= all_time_steps) 
        & (all_time_steps <= key_point_time_arr[3]))
    if len(mask_idxs) > 0:
        curr_time_steps = all_time_steps[mask_idxs] - key_point_time_arr[2]
        phase_steps[mask_idxs] = max_speed * curr_time_steps \
            - acceleration / 2 * np.square(curr_time_steps) \
            + acc_dist + cruise_dist
    
    # end_padding
    int_end_phase = acc_dist * 2 + cruise_dist
    mask_idxs = np.flatnonzero(key_point_time_arr[3] <= all_time_steps)
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = int_end_phase
        
    return phase_steps