"""Per-frame characteristics (spec §7.5): flow contrast inside/outside the object, camera motion."""
import math

import cv2
import numpy as np

MAG_EPS = 0.5


def fused_mag(fw: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Max over available directions of |flow|; 0 where neither exists (as the author code)."""
    mf = np.sqrt(fw[:, 0].astype(np.float32) ** 2 + fw[:, 1].astype(np.float32) ** 2)
    mb = np.sqrt(bw[:, 0].astype(np.float32) ** 2 + bw[:, 1].astype(np.float32) ** 2)
    out = np.fmax(mf, mb)
    return np.nan_to_num(out, nan=0.0)


def frame_vector(fw: np.ndarray, bw: np.ndarray, t: int) -> np.ndarray:
    f = fw[t].astype(np.float32)
    if not np.isnan(f).all():
        return f
    return -bw[t].astype(np.float32)


def region_masks(gt_small: np.ndarray, erode_px: int = 3, dilate_px: int = 10):
    g = gt_small.astype(np.uint8)
    obj = cv2.erode(g, np.ones((2 * erode_px + 1,) * 2, np.uint8)).astype(bool)
    dil = cv2.dilate(g, np.ones((2 * dilate_px + 1,) * 2, np.uint8)).astype(bool)
    return obj, ~dil


def flow_contrast(vec: np.ndarray, mag: np.ndarray, obj: np.ndarray, bg: np.ndarray) -> dict:
    nan = {"mag_obj": math.nan, "mag_bg": math.nan, "mag_contrast": math.nan, "vec_contrast": math.nan}
    if obj.sum() == 0 or bg.sum() == 0 or np.isnan(vec).all():
        return nan
    mag_obj, mag_bg = float(np.median(mag[obj])), float(np.median(mag[bg]))
    v_obj = np.array([np.nanmean(vec[0][obj]), np.nanmean(vec[1][obj])])
    v_bg = np.array([np.nanmean(vec[0][bg]), np.nanmean(vec[1][bg])])
    return {
        "mag_obj": mag_obj, "mag_bg": mag_bg,
        "mag_contrast": mag_obj / (mag_bg + MAG_EPS),
        "vec_contrast": float(np.linalg.norm(v_obj - v_bg)),
    }


def camera_motion(poses: np.ndarray, depth_median: np.ndarray, chunk: np.ndarray):
    """Rotation angle (deg) and depth-normalised translation to the neighbouring frame in the same chunk.

    Rows must be ordered so that neighbours in a chunk are adjacent in temporal order; callers pass
    frames sorted by (chunk, pos_in_chunk).
    """
    n = len(poses)
    rot = np.full(n, math.nan)
    trans = np.full(n, math.nan)
    for i in range(n):
        j = i + 1 if i + 1 < n and chunk[i + 1] == chunk[i] else (i - 1 if i > 0 and chunk[i - 1] == chunk[i] else None)
        if j is None:
            continue
        ri, rj = poses[i][:3, :3], poses[j][:3, :3]
        r_rel = ri.T @ rj
        c = np.clip((np.trace(r_rel) - 1) / 2, -1.0, 1.0)
        rot[i] = math.degrees(math.acos(c))
        t_rel = ri.T @ (poses[j][:3, 3] - poses[i][:3, 3])
        d = depth_median[i]
        trans[i] = float(np.linalg.norm(t_rel) / d) if d and np.isfinite(d) and d > 0 else math.nan
    return rot, trans
