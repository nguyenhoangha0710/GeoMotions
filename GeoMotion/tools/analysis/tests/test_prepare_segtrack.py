import numpy as np
import pytest
from PIL import Image

from tools.analysis.prepare_segtrack import prepare_segtrack


def _img(path, value, mode="RGB", size=(8, 6)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3) if mode == "RGB" else (size[1], size[0]), value, np.uint8)
    Image.fromarray(arr, mode=mode).save(path)


@pytest.fixture
def raw(tmp_path):
    src = tmp_path / "SegTrackv2"
    # single object, png
    for i in (1, 2):
        _img(src / "JPEGImages" / "frog" / f"{i:010d}.png", 50 + i)
        _img(src / "Annotations" / "frog" / f"{i:010d}.png", 255 if i == 1 else 7)
    # bmp images + merged GT bmp + per-object subfolders (ignored)
    for s in ("penguin_00000", "penguin_00001"):
        _img(src / "JPEGImages" / "penguin" / f"{s}.bmp", 100)
        _img(src / "Annotations" / "penguin" / f"{s}.bmp", 128)
        _img(src / "Annotations" / "penguin" / "1" / f"{s}.png", 255)
    # extra GT without image (worm 0000000000)
    for i in (0, 1, 2):
        if i:
            _img(src / "JPEGImages" / "worm" / f"{i:010d}.png", 10)
        _img(src / "Annotations" / "worm" / f"{i:010d}.png", 255)
    return src, tmp_path / "SegTrackv2_std"


def test_layout_and_formats(raw):
    src, dst = raw
    report = prepare_segtrack(src, dst)
    imgs = dst / "JPEGImages_jpg_standardized"
    gts = dst / "GroundTruth"
    assert sorted(p.name for p in (imgs / "penguin").iterdir()) == ["penguin_00000.png", "penguin_00001.png"]
    assert sorted(p.name for p in (gts / "penguin").iterdir()) == ["penguin_00000.png", "penguin_00001.png"]
    assert not (gts / "penguin" / "1").exists()
    assert set(report["video"]) == {"frog", "penguin", "worm"}


def test_pixels_are_lossless(raw):
    src, dst = raw
    prepare_segtrack(src, dst)
    a = np.array(Image.open(dst / "JPEGImages_jpg_standardized" / "frog" / "0000000002.png"))
    assert a.shape == (6, 8, 3) and (a == 52).all()
    g = np.array(Image.open(dst / "GroundTruth" / "frog" / "0000000002.png").convert("L"))
    assert (g == 7).all()  # intermediate GT values kept; eval.py applies its own > 0 rule
    gp = np.array(Image.open(dst / "GroundTruth" / "penguin" / "penguin_00001.png").convert("L"))
    assert (gp == 128).all()


def test_extra_gt_dropped_and_reported(raw):
    src, dst = raw
    report = prepare_segtrack(src, dst).set_index("video")
    assert sorted(p.name for p in (dst / "GroundTruth" / "worm").iterdir()) == ["0000000001.png", "0000000002.png"]
    assert report.loc["worm", "dropped_gt"] == "0000000000"
    assert report.loc["worm", "n_img"] == report.loc["worm", "n_gt"] == 2


def test_image_without_gt_raises(raw):
    src, dst = raw
    (src / "Annotations" / "frog" / "0000000002.png").unlink()
    with pytest.raises(ValueError, match="frog"):
        prepare_segtrack(src, dst)
