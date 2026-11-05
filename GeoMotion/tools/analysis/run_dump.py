"""Runner: re-run GeoMotion with the exact eval.py protocol and dump raw per-frame data.

Usage (from repo root):
  python tools/analysis/run_dump.py --datasets 2016 --gpu 1 --run_id 2026-11-05_cvpr_seq32
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.analysis import io_utils  # noqa: E402
from tools.analysis.protocols import DATASETS, list_sequences, plan_chunks, sam_order, select_frames  # noqa: E402

SAM_PROMPT_THR = 0.8  # motion_seg_inference.preprocess_mask default
FLOW_HALF = 259


# ---------------------------------------------------------------- torch helpers
def make_flow_encoder_microbatch(inner, k: int):
    """Wrap a per-frame module so it runs on at most k frames at a time (identical math)."""
    import torch

    class FlowEncoderMicrobatch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.k = k

        def forward(self, x):
            if self.k <= 0 or x.shape[0] <= self.k:
                return self.inner(x)
            return torch.cat([self.inner(x[i:i + self.k]) for i in range(0, x.shape[0], self.k)], dim=0)

    return FlowEncoderMicrobatch()


def compute_chunk_flows(video01, raft_fn):
    """Mirror compute_optical_flow_magnitude_batch(bidirectional=True, fusion_method='max').

    video01: [N,3,H,W] in [0,1]. Returns fw, bw ([N,2,H,W], NaN where absent) and fused magnitude [N,H,W].
    """
    import torch
    n, _, h, w = video01.shape
    fw = torch.full((n, 2, h, w), float("nan"), device=video01.device)
    bw = torch.full((n, 2, h, w), float("nan"), device=video01.device)
    mag = torch.zeros((n, h, w), device=video01.device)
    for t in range(n - 1):
        fw[t] = raft_fn(video01[t], video01[t + 1])
        bw[t + 1] = raft_fn(video01[t + 1], video01[t])
    for t in range(n):
        vals = []
        if t > 0:
            vals.append(torch.sqrt(bw[t, 0] ** 2 + bw[t, 1] ** 2))
        if t < n - 1:
            vals.append(torch.sqrt(fw[t, 0] ** 2 + fw[t, 1] ** 2))
        if vals:
            mag[t] = torch.max(torch.stack(vals), dim=0)[0] if len(vals) > 1 else vals[0]
    return fw, bw, mag


def half_res_flow(flow):
    """[N,2,H,W] torch -> float16 numpy [N,2,259,259] (area downsample, displacements scaled by 0.5)."""
    import torch.nn.functional as F
    out = F.interpolate(flow.float(), size=(FLOW_HALF, FLOW_HALF), mode="area") * 0.5
    return out.cpu().numpy().astype(np.float16)


def predict_chunk(inf, frames_pil):
    """Same computation as MotionSegmentationInference.predict_motion_mask, keeping intermediates."""
    import torch
    from motion_seg_inference import compute_optical_flow_raft, denormalize_imagenet
    video_tensor = inf.preprocess_video(frames_pil).to(inf.device)
    with torch.no_grad():
        video01 = denormalize_imagenet(video_tensor)[0]
        fw, bw, mag = compute_chunk_flows(video01, compute_optical_flow_raft)
        pred = inf.model(video_tensor, mag[None])
    lp = pred["local_points"][0][..., 2].flatten(1)
    return {
        "prob": pred["motion_mask"][0].float().cpu().numpy(),
        "poses": pred["camera_poses"][0].float().cpu().numpy(),
        "depth_median": lp.median(dim=1).values.float().cpu().numpy(),
        "fw": half_res_flow(fw), "bw": half_res_flow(bw),
        "mag": mag.cpu().numpy(),
    }


def _has_prompt_both_parities(probs: np.ndarray) -> bool:
    any_prompt = [(probs[i] > SAM_PROMPT_THR).sum() > 0 for i in range(len(probs))]
    return any(any_prompt[1::2]) and any(any_prompt[0::2])


def _is_no_prompt_error(err: BaseException) -> bool:
    """SAM2 raises this when propagate_in_video starts without any prompt."""
    msg = str(err).lower()
    return "no input points or masks" in msg or "no points are provided" in msg


def sam_refine_safe(frames_pil, probs, predictor, refine_fn=None, device="cuda"):
    """Mirror eval._refine_with_sam; return (masks [N,H,W] float32, no_prompt flag).

    refine_sam prompts odd frames then even frames; eval.py crashes when a parity has no prompt.
    Here that case (or any SAM error) yields empty masks and no_prompt=True.
    """
    n, mh, mw = probs.shape
    empty = np.zeros((n, mh, mw), np.float32)
    if not _has_prompt_both_parities(probs):
        return empty, True
    import torch
    if refine_fn is None:
        from motion_seg_inference import refine_sam as refine_fn
    arr = np.stack([np.array(f.resize((mw, mh), Image.BILINEAR)).astype(np.float32) / 255.0 for f in frames_pil])
    frame_tensors = torch.from_numpy(arr).float().permute(0, 3, 1, 2)
    buf = torch.zeros((n, mh, mw), dtype=torch.float32, device=device)
    try:
        refine_fn(frame_tensors=frame_tensors, mask_list=probs, p_masks_sam=buf, offset=0, predictor=predictor)
    except RuntimeError as e:
        if not _is_no_prompt_error(e):
            raise  # e.g. CUDA OOM -> per-video handler writes failed.csv, no done.flag
        print(f"    [warn] SAM2 had no prompt ({e}); using empty masks")
        return empty, True
    return buf.detach().cpu().numpy(), False


# ---------------------------------------------------------------- per sequence
def _gt_size(spec, gt_dir):
    from tools.analysis.protocols import natural_key
    files = [f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = sorted(files, key=natural_key) if spec.mode == "fbms" else sorted(files)
    w, h = Image.open(os.path.join(gt_dir, files[0])).size
    return w, h


def run_sequence(inf, predictor, spec, data_root, video, run_dir, sequence_length, use_sam=True):
    image_root = os.path.join(data_root, spec.image_rel)
    ann_root = os.path.join(data_root, spec.ann_rel)
    frames_dir, gt_dir = os.path.join(image_root, video), os.path.join(ann_root, video)
    out = io_utils.seq_dir(run_dir, spec.key, video)
    io_utils.reset_seq_dir(out)
    (out / "prob_pre_sam").mkdir()
    (out / "mask_post_sam").mkdir()

    files = select_frames(spec, frames_dir, gt_dir)
    gt_w, gt_h = _gt_size(spec, gt_dir)
    frames = {f: Image.open(os.path.join(frames_dir, f)).convert("RGB").resize((gt_w, gt_h), Image.BILINEAR)
              for f in files}

    t0 = time.time()
    per_frame, slots = {}, {}
    for chunk in plan_chunks(files, spec.mode, sequence_length):
        res = predict_chunk(inf, [frames[s.frame_file] for s in chunk])
        for i, s in enumerate(chunk):
            slots[s.frame_file] = s
            per_frame[s.frame_file] = {k: res[k][i] for k in ("prob", "poses", "depth_median", "fw", "bw", "mag")}
    t_model = time.time() - t0

    order = sam_order(files, spec.mode)
    probs = np.stack([per_frame[f]["prob"] for f in order])
    t1 = time.time()
    if use_sam:
        post, no_prompt = sam_refine_safe([frames[f] for f in order], probs, predictor)
    else:
        post, no_prompt = (probs > 0.1).astype(np.float32), False
    t_sam = time.time() - t1

    rows = []
    for i, f in enumerate(order):
        stem = os.path.splitext(f)[0]
        io_utils.save_prob_png(out / "prob_pre_sam" / f"{stem}.png", per_frame[f]["prob"])
        io_utils.save_prob_png(out / "mask_post_sam" / f"{stem}.png", post[i])
        s, m = slots[f], per_frame[f]["mag"]
        rows.append({
            "frame_name": stem, "orig_idx": s.orig_idx, "frame_num": s.frame_num, "group": s.group,
            "chunk": s.chunk, "pos_in_chunk": s.pos_in_chunk, "chunk_len": s.chunk_len,
            "is_chunk_start": s.is_chunk_start, "is_chunk_end": s.is_chunk_end,
            "gap_prev": s.gap_prev, "gap_next": s.gap_next,
            "flow_mag_mean": float(m.mean()), "flow_mag_median": float(np.median(m)),
            "flow_mag_p95": float(np.percentile(m, 95)),
            "has_fw": not np.isnan(per_frame[f]["fw"]).all(), "has_bw": not np.isnan(per_frame[f]["bw"]).all(),
            "depth_median": float(per_frame[f]["depth_median"]), "sam_no_prompt": bool(no_prompt),
        })
    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)
    names = np.array([os.path.splitext(f)[0] for f in order])
    np.savez_compressed(out / "flow.npz", fw=np.stack([per_frame[f]["fw"] for f in order]),
                        bw=np.stack([per_frame[f]["bw"] for f in order]), frame_names=names)
    np.savez_compressed(out / "poses.npz", poses=np.stack([per_frame[f]["poses"] for f in order]),
                        frame_names=names)
    (out / "timing.json").write_text(json.dumps({"model_s": t_model, "sam_s": t_sam, "n_frames": len(order)}))
    io_utils.mark_done(out)
    print(f"  {video}: {len(order)} frames, model {t_model:.0f}s, sam {t_sam:.0f}s, no_prompt={no_prompt}")


# ---------------------------------------------------------------- CLI
def _sha8(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()[:8]


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    except OSError:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True, choices=list(DATASETS))
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--sequence_length", type=int, default=32)
    ap.add_argument("--flow_cnn_microbatch", type=int, default=8)
    ap.add_argument("--no_sam", action="store_true")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--model_path", default="checkpoint/best_model.pth")
    ap.add_argument("--pi3_model_path", default="checkpoint/model.safetensors")
    ap.add_argument("--sam2_config_path", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2_checkpoint_path", default="sam2-main/checkpoints/sam2.1_hiera_large.pt")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torchvision
    from motion_seg_inference import MotionSegmentationInference

    run_dir = Path(args.results_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.update({
        "run_id": args.run_id, "git_head": _git_head(), "sequence_length": args.sequence_length,
        "flow_cnn_microbatch": args.flow_cnn_microbatch, "use_sam": not args.no_sam,
        "model_path": args.model_path, "model_sha8": _sha8(args.model_path),
        "pi3_model_path": args.pi3_model_path, "sam2_checkpoint_path": args.sam2_checkpoint_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "sam2_has_C": importlib.util.find_spec("sam2._C") is not None,
        "datasets": sorted(set(cfg.get("datasets", [])) | set(args.datasets)),
        "started": cfg.get("started", time.strftime("%Y-%m-%d %H:%M:%S")),
    })
    cfg_path.write_text(json.dumps(cfg, indent=2))

    inf = MotionSegmentationInference(args.model_path, pi3_model_path=args.pi3_model_path, device="cuda")
    if args.flow_cnn_microbatch > 0:
        mad = inf.model.motion_aware_decoder
        mad.flow_encoder = make_flow_encoder_microbatch(mad.flow_encoder, args.flow_cnn_microbatch)
    predictor = None
    if not args.no_sam:
        from sam2 import build_sam
        predictor = build_sam.build_sam2_video_predictor(args.sam2_config_path, args.sam2_checkpoint_path,
                                                         device="cuda")

    for key in args.datasets:
        spec = DATASETS[key]
        seqs = list_sequences(spec, os.path.join(args.data_root, spec.image_rel))
        if args.videos:
            seqs = [s for s in seqs if s in args.videos]
        print(f"=== {key}: {len(seqs)} videos")
        for video in seqs:
            out = io_utils.seq_dir(run_dir, key, video)
            if io_utils.is_done(out):
                print(f"  {video}: done, skip")
                continue
            try:
                run_sequence(inf, predictor, spec, args.data_root, video, run_dir, args.sequence_length,
                             use_sam=not args.no_sam)
            except Exception as e:  # keep going on OOM or data errors
                print(f"  {video}: FAILED {type(e).__name__}: {e}")
                io_utils.append_failed(run_dir, key, video, e)
                torch.cuda.empty_cache()

    cfg = json.loads(cfg_path.read_text())
    cfg["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg_path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
