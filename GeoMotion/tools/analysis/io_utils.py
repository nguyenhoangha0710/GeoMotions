"""File helpers shared by the runner and the analyzer (no torch dependency)."""
import csv
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

DONE_FLAG = "done.flag"


def save_prob_png(path, prob: np.ndarray) -> None:
    """Store probabilities in [0,1] as uint8 by truncation, like eval._resize_mask."""
    Image.fromarray((np.asarray(prob, np.float32) * 255).astype(np.uint8), mode="L").save(path)


def load_gray_png(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def seq_dir(run_dir, dataset: str, video: str) -> Path:
    return Path(run_dir) / dataset / video


def is_done(path) -> bool:
    return (Path(path) / DONE_FLAG).exists()


def mark_done(path) -> None:
    (Path(path) / DONE_FLAG).write_text("ok\n")


def reset_seq_dir(path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def append_failed(run_dir, dataset: str, video: str, err: BaseException) -> None:
    path = Path(run_dir) / "failed.csv"
    new = not path.exists()
    trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))[-1500:]
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "video", "error", "trace"])
        w.writerow([dataset, video, f"{type(err).__name__}: {err}", trace])
