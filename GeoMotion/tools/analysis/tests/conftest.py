import os
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture
def tmp_dataset(tmp_path):
    """Fake DAVIS-style dataset: 2 videos x 5 frames (64x48), with binary GT."""
    image_root = tmp_path / "JPEGImages" / "480p"
    ann_root = tmp_path / "Annotations" / "480p"
    rng = np.random.default_rng(0)
    for video in ["blackswan", "camel"]:
        (image_root / video).mkdir(parents=True)
        (ann_root / video).mkdir(parents=True)
        for i in range(5):
            img = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
            Image.fromarray(img).save(image_root / video / f"{i:05d}.jpg")
            gt = np.zeros((48, 64), np.uint8)
            gt[10:30, 20:40] = 255
            Image.fromarray(gt).save(ann_root / video / f"{i:05d}.png")
    return str(image_root), str(ann_root)


# core/eval/eval_mask.py imports tqdm at module level; provide a no-op stand-in when absent (test env only).
try:
    import tqdm  # noqa: F401
except ImportError:  # pragma: no cover
    import types
    _stub = types.ModuleType("tqdm")
    _stub.tqdm = lambda it=None, *a, **k: it
    sys.modules["tqdm"] = _stub
