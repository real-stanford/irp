import numpy as np
import cv2

def get_traj_occupancy(coords, transformer):
    grid = np.zeros(
        transformer.shape + (3,), 
        dtype=np.uint8)
    traj_pix = transformer.to_grid(coords).astype(np.int32)
    result = cv2.polylines(
        grid, [traj_pix[:,::-1]], False, color=(1,1,1))
    occu = result[:,:,0] > 0
    return occu

def get_dist_function(img):
    """
    Approximation algorithm is used.
    """
    non_zero_coords = np.stack(np.nonzero(img)).T
    dist_img, nn_pix_idx = cv2.distanceTransformWithLabels(
        src=(~img).astype(np.uint8),
        distanceType=cv2.DIST_L2, 
        maskSize=cv2.DIST_MASK_PRECISE, 
        labelType=cv2.DIST_LABEL_PIXEL)
    nn_pix_coord = non_zero_coords[nn_pix_idx-1]
    return dist_img, nn_pix_coord

def get_dist_function_precise(img):
    dist_img = cv2.distanceTransform(
        src=(~img).astype(np.uint8), 
        distanceType=cv2.DIST_L2, 
        maskSize=cv2.DIST_MASK_PRECISE)
    return dist_img
