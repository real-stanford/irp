import numpy as np
import zarr
from hydra.utils import to_absolute_path
from common.sample_util import get_nd_index_volume

def select_rope_and_goals(
        zarr_path, n_ropes, n_goals,
        mask_names=('split/is_test',),
        seed=0):
    root = zarr.open(to_absolute_path(zarr_path), 'r')
    assert(len(mask_names) > 0)
    mask = None
    for name in mask_names:
        this_mask = root[name][:]
        if mask is None:
            mask = this_mask
        else:
            mask = mask & this_mask

    rope_idx_volume = get_nd_index_volume(mask.shape)
    test_rope_idxs = rope_idx_volume[mask]

    rs = np.random.RandomState(seed=seed)
    rope_ids = test_rope_idxs[
        rs.choice(len(test_rope_idxs), 
        size=n_ropes, replace=False)]
    # select goals
    max_hitrate_array = root['control/max_hitrate']
    img_shape = max_hitrate_array.shape[-2:]
    goal_coord_img = get_nd_index_volume(img_shape)
    rope_goal_dict = dict()
    for i in range(len(rope_ids)):
        rope_id = rope_ids[i]
        this_hitrate_img = max_hitrate_array[tuple(rope_id)]
        valid_goal_mask = this_hitrate_img > 0.95
        valid_goals = goal_coord_img[valid_goal_mask]
        rope_goal_dict[tuple(rope_id.tolist())] = valid_goals[
            rs.choice(len(valid_goals), n_goals, replace=False)]
    return rope_goal_dict
