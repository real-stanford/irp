import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint
import numpy as np
import scipy.ndimage as sn


from common.sample_util import get_nd_index_volume


def build_augmentation_pipeline(cfg: dict, height=None, width=None, apply_prob=0.5):
    sometimes = lambda aug: iaa.Sometimes(apply_prob, aug)
    pipeline = iaa.Sequential(random_order=False)

    if cfg.get("mirror", None):
        opt = cfg.get("mirror", None)  # fliplr
        if type(opt) == int:
            pipeline.add(sometimes(iaa.Fliplr(opt)))
        else:
            pipeline.add(sometimes(iaa.Fliplr(0.5)))

    if cfg.get("rotation", None) > 0:
        pipeline.add(
            iaa.Sometimes(
                cfg.get("rotratio", None),
                iaa.Affine(rotate=(-cfg.get("rotation", None), cfg.get("rotation", None))),
            )
        )

    if cfg.get("motion_blur", None):
        opts = cfg.get("motion_blur_params", None)
        pipeline.add(sometimes(iaa.MotionBlur(**opts)))

    if cfg.get("covering", None):
        pipeline.add(
            sometimes(iaa.CoarseDropout(0.02, size_percent=0.3, per_channel=0.5))
        )

    if cfg.get("elastic_transform", None):
        pipeline.add(sometimes(iaa.ElasticTransformation(sigma=5)))

    if cfg.get("gaussian_noise", False):
        opt = cfg.get("gaussian_noise", False)
        if type(opt) == int or type(opt) == float:
            pipeline.add(
                sometimes(
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, opt), per_channel=0.5
                    )
                )
            )
        else:
            pipeline.add(
                sometimes(
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    )
                )
            )
    if cfg.get("grayscale", False):
        pipeline.add(sometimes(iaa.Grayscale(alpha=(0.5, 1.0))))

    def get_aug_param(cfg_value):
        if isinstance(cfg_value, dict):
            opt = cfg_value
        else:
            opt = {}
        return opt

    cfg_cnt = cfg.get("contrast", {})
    cfg_cnv = cfg.get("convolution", {})

    contrast_aug = ["histeq", "clahe", "gamma", "sigmoid", "log", "linear"]
    for aug in contrast_aug:
        aug_val = cfg_cnt.get(aug, False)
        cfg_cnt[aug] = aug_val
        if aug_val:
            cfg_cnt[aug + "ratio"] = cfg_cnt.get(aug + "ratio", 0.1)

    convolution_aug = ["sharpen", "emboss", "edge"]
    for aug in convolution_aug:
        aug_val = cfg_cnv.get(aug, False)
        cfg_cnv[aug] = aug_val
        if aug_val:
            cfg_cnv[aug + "ratio"] = cfg_cnv.get(aug + "ratio", 0.1)

    if cfg_cnt["histeq"]:
        opt = get_aug_param(cfg_cnt["histeq"])
        pipeline.add(
            iaa.Sometimes(
                cfg_cnt["histeqratio"], iaa.AllChannelsHistogramEqualization(**opt)
            )
        )

    if cfg_cnt["clahe"]:
        opt = get_aug_param(cfg_cnt["clahe"])
        pipeline.add(
            iaa.Sometimes(cfg_cnt["claheratio"], iaa.AllChannelsCLAHE(**opt))
        )

    if cfg_cnt["log"]:
        opt = get_aug_param(cfg_cnt["log"])
        pipeline.add(iaa.Sometimes(cfg_cnt["logratio"], iaa.LogContrast(**opt)))

    if cfg_cnt["linear"]:
        opt = get_aug_param(cfg_cnt["linear"])
        pipeline.add(
            iaa.Sometimes(cfg_cnt["linearratio"], iaa.LinearContrast(**opt))
        )

    if cfg_cnt["sigmoid"]:
        opt = get_aug_param(cfg_cnt["sigmoid"])
        pipeline.add(
            iaa.Sometimes(cfg_cnt["sigmoidratio"], iaa.SigmoidContrast(**opt))
        )

    if cfg_cnt["gamma"]:
        opt = get_aug_param(cfg_cnt["gamma"])
        pipeline.add(iaa.Sometimes(cfg_cnt["gammaratio"], iaa.GammaContrast(**opt)))

    if cfg_cnv["sharpen"]:
        opt = get_aug_param(cfg_cnv["sharpen"])
        pipeline.add(iaa.Sometimes(cfg_cnv["sharpenratio"], iaa.Sharpen(**opt)))

    if cfg_cnv["emboss"]:
        opt = get_aug_param(cfg_cnv["emboss"])
        pipeline.add(iaa.Sometimes(cfg_cnv["embossratio"], iaa.Emboss(**opt)))

    if cfg_cnv["edge"]:
        opt = get_aug_param(cfg_cnv["edge"])
        pipeline.add(iaa.Sometimes(cfg_cnv["edgeratio"], iaa.EdgeDetect(**opt)))

    if height is not None and width is not None:
        if not cfg.get("crop_by", False):
            crop_by = 0.15
        else:
            crop_by = cfg.get("crop_by", False)
        pipeline.add(
            iaa.Sometimes(
                cfg.get("cropratio", 0.4),
                iaa.CropAndPad(percent=(-crop_by, crop_by), keep_size=False),
            )
        )
        pipeline.add(iaa.Resize({"height": height, "width": width}))
    return pipeline


def get_gaussian_scoremap(shape, keypoint: np.ndarray, sigma=5, dtype=np.float32):
    """
    keypoint is float32
    """
    coord_img = get_nd_index_volume(shape).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap


def compute_target_part_scoremap_numpy(
    joint_id, coords, size):
    stride = 8
    half_stride = 4
    locref_scale = 1 / 7.2801
    pos_dist_thresh = 17
    scale = 0.8

    dist_thresh = float(pos_dist_thresh * scale)
    dist_thresh_sq = dist_thresh ** 2
    num_joints = 1

    scmap = np.zeros(np.concatenate([size, np.array([num_joints])]))
    locref_size = np.concatenate([size, np.array([num_joints * 2])])
    locref_mask = np.zeros(locref_size)
    locref_map = np.zeros(locref_size)

    width = size[1]
    height = size[0]
    grid = np.mgrid[:height, :width].transpose((1, 2, 0))

    for person_id in range(len(coords)):
        # for k, j_id in enumerate(joint_id[person_id]):
        k = 0
        j_id = 0

        joint_pt = coords[person_id][k, :]
        j_x = np.asscalar(joint_pt[0])
        j_x_sm = round((j_x - half_stride) / stride)
        j_y = np.asscalar(joint_pt[1])
        j_y_sm = round((j_y - half_stride) / stride)
        min_x = round(max(j_x_sm - dist_thresh - 1, 0))
        max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
        min_y = round(max(j_y_sm - dist_thresh - 1, 0))
        max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))
        x = grid.copy()[:, :, 1]
        y = grid.copy()[:, :, 0]
        dx = j_x - x * stride - half_stride
        dy = j_y - y * stride - half_stride
        dist = dx ** 2 + dy ** 2
        mask1 = dist <= dist_thresh_sq
        mask2 = (x >= min_x) & (x <= max_x)
        mask3 = (y >= min_y) & (y <= max_y)
        mask = mask1 & mask2 & mask3
        scmap[mask, j_id] = 1
        locref_mask[mask, j_id * 2 + 0] = 1
        locref_mask[mask, j_id * 2 + 1] = 1
        locref_map[mask, j_id * 2 + 0] = (dx * locref_scale)[mask]
        locref_map[mask, j_id * 2 + 1] = (dy * locref_scale)[mask]

    weights = np.np.ones(scmap.shape)
    return scmap, weights, locref_map, locref_mask
