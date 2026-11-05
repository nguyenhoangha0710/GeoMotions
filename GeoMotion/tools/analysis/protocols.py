"""Frame/sequence selection and chunking, mirroring eval.py exactly.

References in eval.py:
  - sequence lists: _find_davis_sequences_16, _find_davis_2016_sequences, _find_davis_2017_sequences,
    _find_segtrack_sequences, _find_fbms_sequences
  - grouped protocol: _evaluate_single_sequence (stride=4 interleaved groups, chunks of sequence_length)
  - fbms protocol: _evaluate_single_sequence_fbms (rgb_from_gt=True, one group)
"""
import math
import os
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

DAVIS16_VAL = (
    'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
    'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
    'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox',
)
DAVIS17_VAL = (
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow',
    'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish',
    'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox',
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    image_rel: str
    ann_rel: str
    mode: Literal["grouped", "fbms"]
    val_list: Optional[Tuple[str, ...]]


DATASETS = {
    "2016": DatasetSpec("2016", "DAVIS/JPEGImages/480p", "DAVIS/Annotations/480p", "grouped", DAVIS16_VAL),
    "2016-M": DatasetSpec("2016-M", "DAVIS2017-M/DAVIS/JPEGImages/480p",
                          "DAVIS2017-M/DAVIS/Annotations/480p", "grouped", DAVIS16_VAL),
    "2017-M": DatasetSpec("2017-M", "DAVIS2017-M/DAVIS/JPEGImages/480p",
                          "DAVIS2017-M/DAVIS/Annotations/480p", "grouped", DAVIS17_VAL),
    "segtrack": DatasetSpec("segtrack", "SegTrackv2/JPEGImages_jpg_standardized", "SegTrackv2/GroundTruth",
                            "grouped", None),
    "fbms": DatasetSpec("fbms", "FBMS59_clean/JPEGImages", "FBMS59_clean/Annotations", "fbms", None),
}

RGB_EXTS = (".jpg", ".jpeg", ".png")


def natural_key(s: str) -> list:
    """Copy of eval._natural_key."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_sequences(spec: DatasetSpec, image_root: str) -> List[str]:
    seqs = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    if spec.val_list is not None:
        seqs = [d for d in seqs if d in spec.val_list]
    return sorted(seqs)


def select_frames(spec: DatasetSpec, frames_dir: str, gt_dir: str) -> List[str]:
    if spec.mode == "grouped":
        return sorted(f for f in os.listdir(frames_dir) if f.endswith(RGB_EXTS))
    gt_stems = sorted(
        (os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))),
        key=natural_key,
    )
    selected = []
    for stem in gt_stems:
        for ext in RGB_EXTS:
            found = None
            for cand in (stem + ext, stem + ext.upper()):
                if os.path.exists(os.path.join(frames_dir, cand)):
                    found = cand
                    break
            if found:
                selected.append(found)
                break
    return selected


def frame_number(name: str, fallback: int) -> int:
    nums = re.findall(r"\d+", os.path.splitext(name)[0])
    return int(nums[-1]) if nums else fallback


@dataclass
class FrameSlot:
    frame_file: str
    orig_idx: int
    frame_num: int
    group: int
    chunk: int
    pos_in_chunk: int
    chunk_len: int
    is_chunk_start: bool
    is_chunk_end: bool
    gap_prev: float
    gap_next: float


def plan_chunks(frame_files: List[str], mode: str, sequence_length: int, stride: int = 4) -> List[List[FrameSlot]]:
    if mode == "grouped":
        groups = [[(j * stride + g, frame_files[j * stride + g]) for j in range(len(frame_files[g::stride]))]
                  for g in range(stride)]
    elif mode == "fbms":
        groups = [list(enumerate(frame_files))]
    else:
        raise ValueError(f"unknown mode {mode}")
    nums = [frame_number(f, i) for i, f in enumerate(frame_files)]
    chunks, chunk_id = [], 0
    for g, members in enumerate(groups):
        for start in range(0, len(members), sequence_length):
            part = members[start:start + sequence_length]
            n = len(part)
            slots = []
            for pos, (idx, name) in enumerate(part):
                prev = nums[part[pos - 1][0]] if pos > 0 else None
                nxt = nums[part[pos + 1][0]] if pos < n - 1 else None
                slots.append(FrameSlot(
                    frame_file=name, orig_idx=idx, frame_num=nums[idx], group=g, chunk=chunk_id,
                    pos_in_chunk=pos, chunk_len=n, is_chunk_start=pos == 0, is_chunk_end=pos == n - 1,
                    gap_prev=float(nums[idx] - prev) if prev is not None else math.nan,
                    gap_next=float(nxt - nums[idx]) if nxt is not None else math.nan,
                ))
            chunks.append(slots)
            chunk_id += 1
    return chunks


def sam_order(frame_files: List[str], mode: str) -> List[str]:
    """Order in which frames are fed to SAM2 refinement."""
    if mode == "grouped":
        return list(frame_files)
    by_stem = {os.path.splitext(f)[0]: f for f in frame_files}
    return [by_stem[s] for s in sorted(by_stem)]
