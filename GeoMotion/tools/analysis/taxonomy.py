"""Per-frame error taxonomy (spec §7.3, §7.4). Pure numpy."""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

LABELS = ("empty_ok", "false_alarm", "missed", "wrong_location", "good", "under", "over", "mixed")
SAM_GROUPS = ("sam_rescued", "sam_broke", "both_good", "both_bad")


@dataclass(frozen=True)
class Thresholds:
    tau_area: float = 0.001
    tau_miss: float = 0.1
    tau_good: float = 0.7
    tau_pr: float = 0.8


def mask_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    n = pred.size
    inter = int((pred & gt).sum())
    np_, ng = int(pred.sum()), int(gt.sum())
    return {
        "area_pred": np_ / n, "area_gt": ng / n, "inter": inter,
        "precision": inter / np_ if np_ else math.nan,
        "recall": inter / ng if ng else math.nan,
    }


def label_frame(stats: dict, j: float, th: Thresholds = Thresholds()) -> str:
    if stats["area_gt"] == 0:
        return "empty_ok" if stats["area_pred"] < th.tau_area else "false_alarm"
    if stats["recall"] < th.tau_miss:
        return "missed" if stats["area_pred"] < th.tau_area else "wrong_location"
    if j >= th.tau_good:
        return "good"
    p, r = stats["precision"], stats["recall"]
    if p >= th.tau_pr and r < th.tau_pr:
        return "under"
    if r >= th.tau_pr and p < th.tau_pr:
        return "over"
    return "mixed"


def read_instance_ids(path) -> np.ndarray:
    """Instance id map (0 = background) from palette, grayscale or RGB annotations."""
    img = Image.open(path)
    if img.mode == "P":
        return np.array(img).astype(np.int64)
    if img.mode in ("L", "I", "I;16", "1"):
        arr = np.array(img.convert("L")).astype(np.int64)
        vals = [v for v in np.unique(arr) if v != 0]
        out = np.zeros_like(arr)
        for k, v in enumerate(vals, start=1):
            out[arr == v] = k
        return out
    rgb = np.array(img.convert("RGB")).reshape(-1, 3)
    colors, inv = np.unique(rgb, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    bg = np.where((colors == 0).all(axis=1))[0]
    out = np.zeros(inv.shape, np.int64)
    k = 1
    for c in range(len(colors)):
        if len(bg) and c == bg[0]:
            continue
        out[inv == c] = k
        k += 1
    return out.reshape(img.size[1], img.size[0])


def instance_misses(pred: np.ndarray, ids: np.ndarray, recall_thr: float = 0.5):
    pred = pred.astype(bool)
    inst = [i for i in np.unique(ids) if i != 0]
    missed = sum(1 for i in inst if (pred & (ids == i)).sum() / (ids == i).sum() < recall_thr)
    return len(inst), missed


def sam_group(j_pre: float, j_post: float, tau_good: float = 0.7) -> str:
    pre_ok, post_ok = j_pre >= tau_good, j_post >= tau_good
    if not pre_ok and post_ok:
        return "sam_rescued"
    if pre_ok and not post_ok:
        return "sam_broke"
    return "both_good" if pre_ok else "both_bad"
