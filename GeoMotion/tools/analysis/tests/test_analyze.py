import json
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from core.eval.eval_mask import db_eval_boundary, db_eval_iou
from tools.analysis import io_utils
from tools.analysis.analyze import analyze_sequence, summarize, to_gt_size
from tools.analysis.protocols import DATASETS
from tools.analysis.taxonomy import Thresholds
from tools.analysis.verify_against_eval import main as verify_main

H, W, S = 48, 64, 32  # GT size, saved prediction size


@pytest.fixture
def fake_run(tmp_path):
    data_root, run_dir = tmp_path / "data", tmp_path / "results" / "r1"
    ann = data_root / "DAVIS" / "Annotations" / "480p" / "blackswan"
    img = data_root / "DAVIS" / "JPEGImages" / "480p" / "blackswan"
    ann.mkdir(parents=True); img.mkdir(parents=True)
    out = io_utils.seq_dir(run_dir, "2016", "blackswan")
    (out / "prob_pre_sam").mkdir(parents=True); (out / "mask_post_sam").mkdir()
    rows, n = [], 6
    for i in range(n):
        gt = np.zeros((H, W), np.uint8); gt[10:30, 20 + i:40 + i] = 255
        Image.fromarray(gt).save(ann / f"{i:05d}.png")
        Image.new("RGB", (W, H)).save(img / f"{i:05d}.jpg")
        prob = np.zeros((S, S), np.float32); prob[7:20, 10:22] = 0.6 + 0.05 * i
        io_utils.save_prob_png(out / "prob_pre_sam" / f"{i:05d}.png", prob)
        post = np.zeros((S, S), np.float32); post[6:20, 9:21] = 1.0
        io_utils.save_prob_png(out / "mask_post_sam" / f"{i:05d}.png", post)
        rows.append({"frame_name": f"{i:05d}", "orig_idx": i, "frame_num": i, "group": i % 4,
                     "chunk": i % 4, "pos_in_chunk": i // 4, "chunk_len": 2 if i % 4 < 2 else 1,
                     "is_chunk_start": i < 4, "is_chunk_end": True, "gap_prev": 4.0, "gap_next": 4.0,
                     "flow_mag_mean": 1.0, "flow_mag_median": 1.0, "flow_mag_p95": 2.0, "has_fw": True,
                     "has_bw": True, "depth_median": 2.0, "sam_no_prompt": False})
    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)
    fw = np.zeros((n, 2, 16, 16), np.float16); bw = np.zeros_like(fw)
    np.savez_compressed(out / "flow.npz", fw=fw, bw=bw, frame_names=np.array([r["frame_name"] for r in rows]))
    np.savez_compressed(out / "poses.npz", poses=np.stack([np.eye(4, dtype=np.float32)] * n),
                        frame_names=np.array([r["frame_name"] for r in rows]))
    io_utils.mark_done(out)
    return run_dir, data_root


def test_author_metric_matches_manual(fake_run):
    run_dir, data_root = fake_run
    df = analyze_sequence(run_dir, DATASETS["2016"], data_root, "blackswan", Thresholds())
    assert len(df) == 6 and not df["gt_count_mismatch"].any()
    gt = np.array(Image.open(data_root / "DAVIS/Annotations/480p/blackswan/00003.png").convert("L")) > 0
    prob = io_utils.load_gray_png(run_dir / "2016/blackswan/prob_pre_sam/00003.png")
    pred = to_gt_size(prob, gt.shape) > 0.1
    row = df.set_index("frame_name").loc["00003"]
    assert math.isclose(row["J_pre_0.1"], db_eval_iou(gt, pred), rel_tol=1e-9)
    assert math.isclose(row["F_pre_0.1"], db_eval_boundary(gt, pred, bound_th=0.008), rel_tol=1e-9)
    assert row["label_post"] in ("good", "under", "over", "mixed")
    assert {"mag_contrast", "vec_contrast", "cam_rot_deg", "sam_group", "dJ"} <= set(df.columns)


def test_gt_count_mismatch_flag(fake_run):
    run_dir, data_root = fake_run
    Image.fromarray(np.zeros((H, W), np.uint8)).save(data_root / "DAVIS/Annotations/480p/blackswan/00006.png")
    df = analyze_sequence(run_dir, DATASETS["2016"], data_root, "blackswan", Thresholds())
    assert df["gt_count_mismatch"].all() and len(df) == 6


def _row(video, j, j5):
    return {"dataset": "2016", "video": video, "J_post_0.1": j, "F_post_0.1": j, "J_post_0.5": j5,
            "F_post_0.5": j5, "J_pre_0.1": j, "F_pre_0.1": j, "J_pre_0.5": j5, "F_pre_0.5": j5}


def test_summary_two_protocols():
    df = pd.DataFrame([_row("a", 1.0, 1.0)] * 2 + [_row("b", 0.5, 0.5)] * 8)
    s = summarize(df).set_index(["refine", "protocol"])
    assert math.isclose(s.loc[("post", "author"), "J"], 60.0)
    assert math.isclose(s.loc[("post", "davis_std"), "J"], 75.0)
    # author JR: frame-weighted mean of per-video JR (a: 1.0, b: 0.0) -> 0.2
    assert math.isclose(s.loc[("post", "author"), "JR"], 20.0)
    assert math.isclose(s.loc[("post", "author"), "paper_JF"], 84.7)


def test_verify_script(tmp_path):
    ft = pd.DataFrame([_row("a", 0.9, 0.9), _row("a", 0.7, 0.7), _row("b", 0.5, 0.5)])
    ft.to_csv(tmp_path / "ft.csv", index=False)
    pd.DataFrame({"Sequence": ["a", "b"], "Mean_IoU": [0.8, 0.5], "Mean_F_Score": [0.8, 0.5]}).to_csv(
        tmp_path / "ok.csv", index=False)
    pd.DataFrame({"Sequence": ["a", "b"], "Mean_IoU": [0.81, 0.5], "Mean_F_Score": [0.8, 0.5]}).to_csv(
        tmp_path / "bad.csv", index=False)
    base = ["--frame_table", str(tmp_path / "ft.csv"), "--dataset", "2016"]
    assert verify_main(["--eval_csv", str(tmp_path / "ok.csv")] + base) == 0
    assert verify_main(["--eval_csv", str(tmp_path / "bad.csv")] + base) == 1


@pytest.fixture
def fake_fbms_run(tmp_path):
    """FBMS-style: sparse GT, stems not zero-padded (natural order != plain order)."""
    data_root, run_dir = tmp_path / "data", tmp_path / "results" / "r2"
    ann = data_root / "FBMS59_clean" / "Annotations" / "cars1"
    img = data_root / "FBMS59_clean" / "JPEGImages" / "cars1"
    ann.mkdir(parents=True); img.mkdir(parents=True)
    out = io_utils.seq_dir(run_dir, "fbms", "cars1")
    (out / "prob_pre_sam").mkdir(parents=True); (out / "mask_post_sam").mkdir()
    stems = ["cars1_1", "cars1_10", "cars1_2"]  # plain-sorted = SAM order written by run_dump
    for k, stem in enumerate(stems):
        gt = np.zeros((H, W), np.uint8); gt[5 + 10 * k:15 + 10 * k, 5:25] = 255  # distinct per frame
        Image.fromarray(gt).save(ann / f"{stem}.png")
        Image.new("RGB", (W, H)).save(img / f"{stem}.jpg")
        m = (np.array(Image.fromarray(gt).resize((S, S), Image.NEAREST)) > 0).astype(np.float32)
        io_utils.save_prob_png(out / "prob_pre_sam" / f"{stem}.png", m)
        io_utils.save_prob_png(out / "mask_post_sam" / f"{stem}.png", m)
    rows = [{"frame_name": s, "orig_idx": i, "frame_num": int(s.split("_")[1]), "group": 0, "chunk": 0,
             "pos_in_chunk": i, "chunk_len": 3, "is_chunk_start": i == 0, "is_chunk_end": i == 2,
             "gap_prev": 1.0, "gap_next": 1.0, "flow_mag_mean": 0.0, "flow_mag_median": 0.0, "flow_mag_p95": 0.0,
             "has_fw": True, "has_bw": True, "depth_median": 2.0, "sam_no_prompt": False}
            for i, s in enumerate(stems)]
    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)
    np.savez_compressed(out / "flow.npz", fw=np.zeros((3, 2, 16, 16), np.float16),
                        bw=np.zeros((3, 2, 16, 16), np.float16), frame_names=np.array(stems))
    np.savez_compressed(out / "poses.npz", poses=np.stack([np.eye(4, dtype=np.float32)] * 3),
                        frame_names=np.array(stems))
    io_utils.mark_done(out)
    return run_dir, data_root


def test_fbms_pairs_gt_by_stem(fake_fbms_run):
    run_dir, data_root = fake_fbms_run
    df = analyze_sequence(run_dir, DATASETS["fbms"], data_root, "cars1", Thresholds())
    assert sorted(df["frame_name"]) == ["cars1_1", "cars1_10", "cars1_2"]
    assert (df["J_post_0.1"] > 0.5).all()  # own GT -> high J; boxes of other frames do not overlap (J=0)
    assert not df["gt_count_mismatch"].any()
