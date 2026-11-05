import math

import numpy as np

from tools.analysis.features import camera_motion, flow_contrast, frame_vector, fused_mag, region_masks


def _scene(obj_uv, bg_uv, size=60, box=(20, 40)):
    vec = np.zeros((2, size, size), np.float32)
    vec[0], vec[1] = bg_uv
    gt = np.zeros((size, size), bool)
    gt[box[0]:box[1], box[0]:box[1]] = True
    vec[0][gt], vec[1][gt] = obj_uv
    mag = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec, mag, gt


def test_moving_object_static_background():
    vec, mag, gt = _scene((3, 0), (0, 0))
    obj, bg = region_masks(gt)
    r = flow_contrast(vec, mag, obj, bg)
    assert math.isclose(r["mag_contrast"], 6.0, rel_tol=1e-4)
    assert math.isclose(r["vec_contrast"], 3.0, rel_tol=1e-4)


def test_direction_only_difference():
    vec, mag, gt = _scene((-3, 0), (3, 0))
    obj, bg = region_masks(gt)
    r = flow_contrast(vec, mag, obj, bg)
    assert math.isclose(r["mag_contrast"], 3 / 3.5, rel_tol=1e-4)
    assert math.isclose(r["vec_contrast"], 6.0, rel_tol=1e-4)


def test_tiny_object_gives_nan():
    vec, mag, gt = _scene((3, 0), (0, 0), box=(30, 32))
    obj, bg = region_masks(gt)
    assert obj.sum() == 0
    r = flow_contrast(vec, mag, obj, bg)
    assert all(math.isnan(v) for v in r.values())


def test_empty_gt_gives_nan():
    vec, mag, _ = _scene((0, 0), (1, 0))
    obj, bg = region_masks(np.zeros((60, 60), bool))
    r = flow_contrast(vec, mag, obj, bg)
    assert math.isnan(r["mag_obj"]) and math.isnan(r["mag_contrast"])


def test_fused_mag_and_frame_vector():
    fw = np.full((2, 2, 4, 4), np.nan, np.float32)
    bw = np.full((2, 2, 4, 4), np.nan, np.float32)
    fw[0] = 3.0                    # frame0 forward (3,3)
    bw[1, 0], bw[1, 1] = 1.0, 2.0  # frame1 backward only
    mag = fused_mag(fw, bw)
    assert np.allclose(mag[0], np.sqrt(18)) and np.allclose(mag[1], np.sqrt(5))
    v = frame_vector(fw, bw, 1)
    assert np.allclose(v[0], -1.0) and np.allclose(v[1], -2.0)


def test_fused_mag_single_frame_zero():
    nan = np.full((1, 2, 4, 4), np.nan, np.float32)
    assert np.allclose(fused_mag(nan, nan), 0.0)
    assert np.isnan(frame_vector(nan, nan, 0)).all()


def _pose(rot_deg, t):
    a = math.radians(rot_deg)
    p = np.eye(4, dtype=np.float32)
    p[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    p[:3, 3] = t
    return p


def test_camera_motion():
    poses = np.stack([_pose(0, (0, 0, 0)), _pose(30, (0.5, 0, 0)), _pose(0, (9, 9, 9))])
    rot, trans = camera_motion(poses, np.array([2.0, 2.0, 2.0]), np.array([0, 0, 1]))
    assert np.allclose(rot[:2], 30, atol=1e-3) and np.allclose(trans[:2], 0.25, atol=1e-5)
    assert math.isnan(rot[2]) and math.isnan(trans[2])
