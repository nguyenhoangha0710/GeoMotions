import numpy as np
from PIL import Image

from tools.analysis.taxonomy import Thresholds, instance_misses, label_frame, mask_stats, read_instance_ids, sam_group

TH = Thresholds()


def _box(y0, y1, x0, x1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def _iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


def _label(pred, gt, j=None):
    return label_frame(mask_stats(pred, gt), _iou(pred, gt) if j is None else j, TH)


def test_empty_ok_and_false_alarm():
    empty = np.zeros((100, 100), bool)
    assert _label(empty, empty) == "empty_ok"
    assert _label(_box(0, 4, 0, 5), empty) == "false_alarm"  # 20 px = 0.2% >= 0.1%


def test_missed_and_wrong_location():
    gt = _box(40, 60, 40, 60)
    assert _label(np.zeros_like(gt), gt) == "missed"
    assert _label(_box(0, 20, 0, 20), gt) == "wrong_location"


def test_good_under_over():
    gt = _box(40, 60, 40, 60)
    assert _label(gt.copy(), gt) == "good"
    assert _label(_box(40, 50, 40, 60), gt) == "under"
    assert _label(_box(30, 70, 30, 70), gt) == "over"


def test_good_at_exact_threshold():
    gt = _box(40, 60, 40, 60)
    assert _label(_box(40, 50, 40, 60), gt, j=0.7) == "good"


def test_instance_misses():
    ids = np.zeros((100, 100), int)
    ids[10:30, 10:30] = 1
    ids[60:80, 60:80] = 2
    assert instance_misses(ids == 1, ids) == (2, 1)


def test_read_instance_ids_palette(tmp_path):
    arr = np.zeros((10, 10), np.uint8)
    arr[1:3, 1:3] = 1
    arr[5:7, 5:7] = 2
    img = Image.fromarray(arr, mode="P")
    img.putpalette([0, 0, 0, 128, 0, 0, 0, 128, 0] + [0] * 759)
    img.save(tmp_path / "a.png")
    ids = read_instance_ids(tmp_path / "a.png")
    assert set(np.unique(ids)) == {0, 1, 2}


def test_read_instance_ids_gray(tmp_path):
    arr = np.zeros((10, 10), np.uint8)
    arr[1:3, 1:3] = 255
    arr[5:7, 5:7] = 128
    Image.fromarray(arr, mode="L").save(tmp_path / "g.png")
    ids = read_instance_ids(tmp_path / "g.png")
    assert len(set(np.unique(ids)) - {0}) == 2 and ids[0, 0] == 0


def test_sam_group():
    assert sam_group(0.5, 0.8) == "sam_rescued"
    assert sam_group(0.8, 0.5) == "sam_broke"
    assert sam_group(0.9, 0.95) == "both_good"
    assert sam_group(0.2, 0.3) == "both_bad"
