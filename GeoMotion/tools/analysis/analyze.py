"""Analyzer: per-frame metrics, error labels, features and summaries from a run_dump output.

Usage (from repo root, CPU only):
  python tools/analysis/analyze.py --run_id 2026-11-05_cvpr_seq32 [--datasets 2016 fbms] [--no_media]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.eval.eval_mask import db_eval_boundary, db_eval_iou  # noqa: E402
from tools.analysis import io_utils  # noqa: E402
from tools.analysis.features import camera_motion, flow_contrast, frame_vector, fused_mag, region_masks  # noqa: E402
from tools.analysis.protocols import DATASETS, natural_key  # noqa: E402
from tools.analysis.taxonomy import (Thresholds, instance_misses, label_frame, mask_stats,  # noqa: E402
                                     read_instance_ids, sam_group)

AUTHOR_THR, STD_THR = 0.1, 0.5
# Table 1 of the paper; column mapping of the HTML table is uncertain -> check against the PDF.
PAPER = {
    "2016": {"J": 84.5, "F": 85.0, "JF": 84.7},
    "2016-M": {"J": 83.5, "F": 84.3, "JF": 83.9},
    "2017-M": {"J": 81.1},
    "fbms": {"J": 72.5, "F": 78.5},
    "segtrack": {"J": 77.3},
}


def to_gt_size(mask, shape_hw) -> np.ndarray:
    """Same as eval._resize_mask: uint8 -> PIL bilinear -> /255."""
    arr = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((shape_hw[1], shape_hw[0]), Image.BILINEAR)
    return np.array(img) / 255.0


def _gt_files(spec, gt_dir):
    files = [f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg"))]
    if spec.mode == "fbms":
        files = [f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        return sorted(files, key=natural_key)
    return sorted(files)


def analyze_sequence(run_dir, spec, data_root, video, th: Thresholds = Thresholds()) -> pd.DataFrame:
    d = io_utils.seq_dir(run_dir, spec.key, video)
    frames = pd.read_csv(d / "frames.csv", dtype={"frame_name": str})
    flow, poses = np.load(d / "flow.npz"), np.load(d / "poses.npz")
    fw, bw = flow["fw"].astype(np.float32), flow["bw"].astype(np.float32)
    mags = fused_mag(fw, bw)
    fh, fwid = fw.shape[-2:]
    gt_dir = Path(data_root) / spec.ann_rel / video
    gt_files = _gt_files(spec, gt_dir)

    # pair frames with GT exactly like eval.py
    if spec.mode == "grouped":
        mismatch = len(gt_files) != len(frames)
        pairs = list(zip(range(len(frames)), gt_files))
    else:
        mismatch = False
        by_stem = {os.path.splitext(g)[0]: g for g in gt_files}
        pairs = [(i, by_stem[n]) for i, n in enumerate(frames["frame_name"]) if n in by_stem]

    # camera motion needs chunk-ordered rows
    order = frames.sort_values(["chunk", "pos_in_chunk"]).index.to_numpy()
    rot_s, tr_s = camera_motion(poses["poses"][order], frames["depth_median"].to_numpy()[order],
                                frames["chunk"].to_numpy()[order])
    rot, tr = np.empty(len(frames)), np.empty(len(frames))
    rot[order], tr[order] = rot_s, tr_s

    # JR as eval._evaluate_chunk: computed over the evaluated frames of the video
    rows = []
    for i, gt_file in pairs:
        fr = frames.iloc[i]
        name = fr["frame_name"]
        gt_raw = np.array(Image.open(gt_dir / gt_file).convert("L"))
        gt = gt_raw > 0
        ids = read_instance_ids(gt_dir / gt_file)
        row = {"dataset": spec.key, "video": video, **fr.to_dict(), "gt_count_mismatch": mismatch,
               "gt_empty": not gt.any(), "area_gt": float(gt.mean()),
               "n_inst": int(len([v for v in np.unique(ids) if v != 0]))}
        for r, sub in (("pre", "prob_pre_sam"), ("post", "mask_post_sam")):
            m = to_gt_size(io_utils.load_gray_png(d / sub / f"{name}.png"), gt.shape)
            for t in (AUTHOR_THR, STD_THR):
                p = m > t
                row[f"J_{r}_{t}"] = float(db_eval_iou(gt, p))
                row[f"F_{r}_{t}"] = float(db_eval_boundary(gt, p, bound_th=0.008))
            p = m > AUTHOR_THR
            st = mask_stats(p, gt)
            row[f"P_{r}"], row[f"R_{r}"], row[f"area_pred_{r}"] = st["precision"], st["recall"], st["area_pred"]
            row[f"label_{r}"] = label_frame(st, row[f"J_{r}_{AUTHOR_THR}"], th)
            row[f"n_inst_missed_{r}"] = instance_misses(p, ids)[1]
        row["dJ"] = row[f"J_post_{AUTHOR_THR}"] - row[f"J_pre_{AUTHOR_THR}"]
        row["sam_group"] = sam_group(row[f"J_pre_{AUTHOR_THR}"], row[f"J_post_{AUTHOR_THR}"], th.tau_good)
        gt_small = np.array(Image.fromarray(gt.astype(np.uint8)).resize((fwid, fh), Image.NEAREST)).astype(bool)
        obj, bg = region_masks(gt_small)
        row.update(flow_contrast(frame_vector(fw, bw, i), mags[i], obj, bg))
        row["cam_rot_deg"], row["cam_trans_rel"] = float(rot[i]), float(tr[i])
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["JR_video_author"] = float((df[f"J_post_{AUTHOR_THR}"] > 0.5).mean())
    return df


def add_dir_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dir_only"] = False
    for ds, g in df.groupby("dataset"):
        mq, vq = g["mag_contrast"].quantile(0.25), g["vec_contrast"].quantile(0.75)
        df.loc[g.index, "dir_only"] = (g["mag_contrast"] <= mq) & (g["vec_contrast"] >= vq)
    return df


def summarize(df: pd.DataFrame, paper: dict = PAPER) -> pd.DataFrame:
    out = []
    for ds, g in df.groupby("dataset"):
        ref = paper.get(ds, {})
        for r in ("pre", "post"):
            ja, fa = g[f"J_{r}_{AUTHOR_THR}"], g[f"F_{r}_{AUTHOR_THR}"]
            jr_video = g.groupby("video")[f"J_{r}_{AUTHOR_THR}"].transform(lambda s: (s > 0.5).mean())
            per_vid = g.groupby("video").agg(J=(f"J_{r}_{STD_THR}", "mean"), F=(f"F_{r}_{STD_THR}", "mean"),
                                             JR=(f"J_{r}_{STD_THR}", lambda s: (s > 0.5).mean()))
            for proto, J, F, JR in (("author", ja.mean(), fa.mean(), jr_video.mean()),
                                    ("davis_std", per_vid["J"].mean(), per_vid["F"].mean(), per_vid["JR"].mean())):
                J, F, JR = 100 * J, 100 * F, 100 * JR
                JF = (J + F) / 2
                row = {"dataset": ds, "refine": r, "protocol": proto, "n_videos": g["video"].nunique(),
                       "n_frames": len(g), "J": J, "F": F, "JF": JF, "JR": JR,
                       "paper_J": ref.get("J", math.nan), "paper_F": ref.get("F", math.nan),
                       "paper_JF": ref.get("JF", math.nan)}
                row["diff_J"] = J - row["paper_J"]
                row["diff_JF"] = JF - row["paper_JF"]
                out.append(row)
    return pd.DataFrame(out)


def sensitivity(df: pd.DataFrame, taus=(0.5, 0.7, 0.8)) -> pd.DataFrame:
    rows = []
    for tau in taus:
        th = Thresholds(tau_good=tau)
        for _, r in df.iterrows():
            st = {"area_pred": r["area_pred_post"], "area_gt": r["area_gt"], "precision": r["P_post"],
                  "recall": r["R_post"]}
            rows.append({"dataset": r["dataset"], "tau_good": tau,
                         "label": label_frame(st, r[f"J_post_{AUTHOR_THR}"], th)})
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    return s.groupby(["dataset", "tau_good", "label"]).size().unstack(fill_value=0).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--no_media", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.results_root) / args.run_id
    datasets = args.datasets or [d for d in DATASETS if (run_dir / d).is_dir()]
    frames = []
    for key in datasets:
        spec = DATASETS[key]
        videos = sorted(v.name for v in (run_dir / key).iterdir() if io_utils.is_done(v))
        print(f"=== {key}: {len(videos)} videos")
        for v in videos:
            frames.append(analyze_sequence(run_dir, spec, args.data_root, v))
    df = add_dir_only(pd.concat(frames, ignore_index=True))
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "frame_table.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    sensitivity(df).to_csv(out / "sensitivity.csv", index=False)
    print(summary[summary["protocol"] == "author"].round(2).to_string(index=False))
    from tools.analysis.report import build_report
    build_report(run_dir, df, summary, datasets, args.data_root, media=not args.no_media)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
