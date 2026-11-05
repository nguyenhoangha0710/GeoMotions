"""Report: video table, plots, galleries, mp4s and report.md (spec §8)."""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from tools.analysis import io_utils  # noqa: E402
from tools.analysis.features import frame_vector, fused_mag  # noqa: E402
from tools.analysis.protocols import DATASETS  # noqa: E402
from tools.analysis.taxonomy import LABELS  # noqa: E402

# dataviz reference palette: fixed categorical order, single-hue sequential, recessive grid
C_PRE, C_POST = "#2a78d6", "#eb6834"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#5f5e58", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
J = "J_{}_0.1"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def longest_bad_run(j, thr: float = 0.5) -> int:
    best = cur = 0
    for v in j:
        cur = cur + 1 if v < thr else 0
        best = max(best, cur)
    return best


def video_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, v), g in df.groupby(["dataset", "video"]):
        g = g.sort_values("orig_idx")
        row = {"dataset": ds, "video": v, "n_frames": len(g)}
        for r in ("pre", "post"):
            row[f"JF_{r}"] = 100 * ((g[J.format(r)] + g[f"F_{r}_0.1"]) / 2).mean()
        row["JR"] = 100 * (g[J.format("post")] > 0.5).mean()
        row["min_J"] = 100 * g[J.format("post")].min()
        row["longest_bad_run"] = longest_bad_run(g[J.format("post")])
        for lab in LABELS:
            row[f"n_{lab}"] = int((g["label_post"] == lab).sum())
        for c in ("area_gt", "mag_contrast", "vec_contrast", "cam_rot_deg", "gap_prev"):
            if c in g:
                row[f"mean_{c}"] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "JF_post"]).reset_index(drop=True)


# ---------------------------------------------------------------- images
def flow_to_color(vec: np.ndarray) -> np.ndarray:
    u, v = vec[0].astype(np.float32), vec[1].astype(np.float32)
    bad = np.isnan(u) | np.isnan(v)
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = (np.degrees(np.arctan2(v, u)) % 360) / 2  # OpenCV hue 0..180
    scale = np.percentile(mag[~bad], 99) if (~bad).any() and mag[~bad].max() > 0 else 1.0
    hsv = np.zeros(u.shape + (3,), np.uint8)
    hsv[..., 0] = ang.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / scale * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[bad] = 0
    return rgb


def _gray3(x01):
    return np.repeat((np.clip(x01, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)


def render_panel(rgb, gt, prob_pre, mask_post, mag, vec) -> np.ndarray:
    """RGB | GT | prob pre | mask post | error overlay | |flow| | flow direction."""
    heat = (plt.get_cmap("magma")(np.clip(prob_pre, 0, 1))[..., :3] * 255).astype(np.uint8)
    pred = mask_post > 0.1
    over = (rgb * 0.45).astype(np.uint8)
    over[pred & gt] = (27, 175, 122)     # TP
    over[pred & ~gt] = (235, 104, 52)    # FP
    over[~pred & gt] = (42, 120, 214)    # FN
    m = mag / (np.percentile(mag, 99) + 1e-6) if mag.max() > 0 else mag
    tiles = [rgb, _gray3(gt.astype(np.float32)), heat, _gray3(pred.astype(np.float32)), over, _gray3(m),
             flow_to_color(vec)]
    return np.concatenate(tiles, axis=1)


def select_gallery(df, k_label=20, k_worst_videos=10, k_per_video=3, k_sam_broke=20) -> pd.DataFrame:
    picks = []
    for (ds, lab), g in df.groupby(["dataset", "label_post"]):
        if lab in ("good", "empty_ok"):
            continue
        picks.append(g.nsmallest(k_label, J.format("post")).assign(reason=f"label:{lab}"))
    vids = df.groupby(["dataset", "video"])[J.format("post")].mean().reset_index()
    for ds, g in vids.groupby("dataset"):
        for v in g.nsmallest(k_worst_videos, J.format("post"))["video"]:
            sub = df[(df["dataset"] == ds) & (df["video"] == v)]
            picks.append(sub.nsmallest(k_per_video, J.format("post")).assign(reason="worst_video"))
    for ds, g in df[df["sam_group"] == "sam_broke"].groupby("dataset"):
        picks.append(g.nsmallest(k_sam_broke, J.format("post")).assign(reason="sam_broke"))
    if not picks:
        return df.iloc[0:0].assign(reason=[])
    return pd.concat(picks).drop_duplicates(["dataset", "video", "frame_name"]).reset_index(drop=True)


def write_mp4(frames, path, fps: int = 6) -> bool:
    if shutil.which("ffmpeg") is None or not frames:
        print("  [warn] ffmpeg missing, skip", path)
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            Image.fromarray(f[: h - h % 2, : w - w % 2]).save(os.path.join(tmp, f"{i:05d}.png"))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
               "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
        return subprocess.run(cmd).returncode == 0


def _panel_for(run_dir, data_root, row):
    spec = DATASETS[row["dataset"]]
    d = io_utils.seq_dir(run_dir, spec.key, row["video"])
    name = row["frame_name"]
    img_dir = Path(data_root) / spec.image_rel / row["video"]
    img_path = next(p for p in img_dir.iterdir() if p.stem == name)
    ann_dir = Path(data_root) / spec.ann_rel / row["video"]
    gt_path = next(p for p in ann_dir.iterdir() if p.stem == name)
    gt = np.array(Image.open(gt_path).convert("L")) > 0
    h, w = gt.shape
    rgb = np.array(Image.open(img_path).convert("RGB").resize((w, h), Image.BILINEAR))
    rs = lambda a: np.array(Image.fromarray(a).resize((w, h), Image.BILINEAR)) / 255.0  # noqa: E731
    pre = rs(io_utils.load_gray_png(d / "prob_pre_sam" / f"{name}.png"))
    post = rs(io_utils.load_gray_png(d / "mask_post_sam" / f"{name}.png"))
    flow = np.load(d / "flow.npz")
    idx = list(flow["frame_names"]).index(name)
    fw, bw = flow["fw"].astype(np.float32), flow["bw"].astype(np.float32)
    mag = cv2.resize(fused_mag(fw[idx:idx + 1], bw[idx:idx + 1])[0], (w, h))
    vec = frame_vector(fw, bw, idx)
    vec = np.stack([cv2.resize(vec[0], (w, h)), cv2.resize(vec[1], (w, h))])
    return render_panel(rgb, gt, pre, post, mag, vec)


# ---------------------------------------------------------------- plots
def plot_j_over_time(df, out_dir, k=10):
    worst = df.groupby("video")[J.format("post")].mean().nsmallest(k).index
    n = len(worst)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.8 * n), squeeze=False)
    for ax, v in zip(axes[:, 0], worst):
        g = df[df["video"] == v].sort_values("orig_idx")
        ax.plot(g["orig_idx"], g[J.format("pre")], color=C_PRE, lw=2, label="truoc SAM2")
        ax.plot(g["orig_idx"], g[J.format("post")], color=C_POST, lw=2, label="sau SAM2")
        e = g[g["gt_empty"]]
        ax.scatter(e["orig_idx"], np.zeros(len(e)), color=INK2, s=10, marker="|", label="GT rong")
        b = g[g["is_chunk_start"] | g["is_chunk_end"]]
        ax.scatter(b["orig_idx"], b[J.format("post")], facecolors="none", edgecolors=INK2, s=12,
                   label="dau/cuoi doan")
        ax.set_ylim(-0.05, 1.05)
        _style(ax, title=v, ylabel="J")
    axes[0, 0].legend(fontsize=7, ncol=4, frameon=False, loc="lower left")
    axes[-1, 0].set_xlabel("chi so frame", color=INK2, fontsize=9)
    _save(fig, out_dir / "1_J_theo_thoi_gian_top_video_te.png")


def plot_label_distribution(df, out_dir):
    pre = df["label_pre"].value_counts().reindex(LABELS, fill_value=0)
    post = df["label_post"].value_counts().reindex(LABELS, fill_value=0)
    y = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(y - 0.2, pre.values, height=0.38, color=C_PRE, label="truoc SAM2")
    ax.barh(y + 0.2, post.values, height=0.38, color=C_POST, label="sau SAM2")
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False)
    _style(ax, title="Phan bo nhan loi", xlabel="so frame")
    _save(fig, out_dir / "2_phan_bo_nhan_loi.png")


def plot_j_by_feature(df, out_dir):
    feats = [c for c in ("area_gt", "mag_contrast", "vec_contrast", "cam_rot_deg", "gap_prev") if c in df]
    fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats), 3), squeeze=False)
    for ax, c in zip(axes[0], feats):
        s = df[[c, J.format("post")]].dropna()
        if s[c].nunique() < 2:
            _style(ax, title=c)
            continue
        bins = pd.qcut(s[c], q=min(5, s[c].nunique()), duplicates="drop")
        agg = s.groupby(bins, observed=True)[J.format("post")].agg(["mean", "size"])
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], color=C_POST, width=0.7)
        for xi, (m, n) in enumerate(zip(agg["mean"], agg["size"])):
            ax.text(xi, m + 0.02, f"n={n}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"{iv.left:.2g}-{iv.right:.2g}" for iv in agg.index], rotation=35, fontsize=7)
        ax.set_ylim(0, 1.1)
        _style(ax, title=c, ylabel="J sau SAM2 (TB)")
    _save(fig, out_dir / "3_J_theo_dac_tinh.png")


def plot_contrast_scatter(df, out_dir):
    s = df[["mag_contrast", "vec_contrast", J.format("post")]].dropna()
    s = s[s["mag_contrast"] > 0]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(s["mag_contrast"], s["vec_contrast"], c=s[J.format("post")], cmap=SEQ, vmin=0, vmax=1,
                    s=10, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("J sau SAM2", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    if len(s):
        ax.axvline(s["mag_contrast"].quantile(0.25), color=AXIS, lw=1, ls="--")
        ax.axhline(s["vec_contrast"].quantile(0.75), color=AXIS, lw=1, ls="--")
        ax.set_xscale("log")
    _style(ax, title="Tuong phan do lon vs tuong phan vector (goc tren-trai = dir_only)",
           xlabel="mag_contrast", ylabel="vec_contrast (px)")
    _save(fig, out_dir / "4_mag_vs_vec_contrast.png")


def plot_j_by_position(df, out_dir):
    pos = np.where(df["is_chunk_start"] & df["is_chunk_end"], "doan 1 frame",
                   np.where(df["is_chunk_start"], "dau doan", np.where(df["is_chunk_end"], "cuoi doan", "giua doan")))
    cats = ["dau doan", "giua doan", "cuoi doan", "doan 1 frame"]
    agg = df.assign(pos=pos).groupby("pos")[J.format("post")].agg(["mean", "size"]).reindex(cats)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.arange(len(cats))
    ax.bar(x, agg["mean"].fillna(0), color=C_POST, width=0.6)
    for xi, n in enumerate(agg["size"].fillna(0).astype(int)):
        ax.text(xi, 0.02, f"n={n}", ha="center", fontsize=7, color="#ffffff" if n else INK2)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.05)
    _style(ax, title="J theo vi tri trong doan", ylabel="J sau SAM2 (TB)")
    _save(fig, out_dir / "5_J_theo_vi_tri_doan.png")


def plot_sam_transition(df, out_dir):
    m = pd.crosstab(df["label_pre"], df["label_post"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(m.values, cmap=SEQ)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, fontsize=8)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
    vmax = m.values.max() or 1
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if m.values[i, j]:
                ax.text(j, i, m.values[i, j], ha="center", va="center", fontsize=7,
                        color="#ffffff" if m.values[i, j] > vmax / 2 else INK)
    ax.set_title("Chuyen nhan: truoc SAM2 (hang) -> sau SAM2 (cot)", color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK2)
    _save(fig, out_dir / "6_chuyen_nhan_SAM2.png")


# ---------------------------------------------------------------- report
def _md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def build_report(run_dir, frame_df, summary_df, datasets, data_root, media: bool = True):
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    vt = video_table(frame_df)
    vt.to_csv(out / "video_table.csv", index=False)
    for ds in datasets:
        g = frame_df[frame_df["dataset"] == ds]
        if g.empty:
            continue
        pdir = out / "plots" / ds
        plot_j_over_time(g, pdir)
        plot_label_distribution(g, pdir)
        plot_j_by_feature(g, pdir)
        plot_contrast_scatter(g, pdir)
        plot_j_by_position(g, pdir)
        plot_sam_transition(g, pdir)
    if media:
        gal = select_gallery(frame_df)
        gal.to_csv(out / "gallery_index.csv", index=False)
        for _, r in gal.iterrows():
            panel = _panel_for(run_dir, data_root, r)
            p = out / "gallery" / r["dataset"] / r["reason"].replace(":", "_") / f"{r['video']}_{r['frame_name']}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel).save(p, quality=88)
        for ds in datasets:
            for v in vt[vt["dataset"] == ds].head(5)["video"]:
                rows = frame_df[(frame_df["dataset"] == ds) & (frame_df["video"] == v)].sort_values("orig_idx")
                write_mp4([_panel_for(run_dir, data_root, r) for _, r in rows.iterrows()],
                          out / "videos" / ds / f"{v}.mp4")
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    author = summary_df[summary_df["protocol"] == "author"]
    std = summary_df[summary_df["protocol"] == "davis_std"]
    cols = ["dataset", "refine", "n_videos", "n_frames", "J", "F", "JF", "JR", "paper_J", "paper_JF", "diff_J",
            "diff_JF"]
    lines = [
        f"# Bao cao danh gia GeoMotion — run `{cfg.get('run_id', run_dir.name)}`", "",
        "## 1. So voi paper (cach tinh cua tac gia: TB moi frame, nguong 0.1)", "", _md_table(author[cols]), "",
        "Luu y: anh xa cot Bang 1 ban HTML chua chac chan (2017-M, SegTrackV2 co the la J) — doi chieu PDF.", "",
        "## 2. Cach tinh chuan DAVIS (TB theo video, nguong 0.5)", "", _md_table(std[cols]), "",
        "## 3. Khac biet cau hinh so voi tac gia", "",
        f"- sequence_length: {cfg.get('sequence_length')}, flow_cnn_microbatch: {cfg.get('flow_cnn_microbatch')}",
        f"- SAM2 co extension _C: {cfg.get('sam2_has_C')} (tac gia: co)",
        f"- GPU: {cfg.get('gpu')}, python {cfg.get('python')}, torch {cfg.get('torch')}",
        f"- Video SAM2 khong co prompt (mask rong, eval.py goc se crash): "
        f"{int(frame_df.groupby(['dataset', 'video'])['sam_no_prompt'].first().sum())}", "",
        "## 4. Video te nhat (JF sau SAM2)", "",
        _md_table(vt.groupby("dataset").head(10)[["dataset", "video", "n_frames", "JF_pre", "JF_post", "JR",
                                                  "longest_bad_run"]]), "",
        "## 5. Ty le nhan loi (sau SAM2)", "",
        _md_table(frame_df.groupby("dataset")["label_post"].value_counts(normalize=True).mul(100).round(1)
                  .unstack(fill_value=0.0).reset_index()), "",
        "## 6. Ung vien 'mat huong flow' (dir_only)", "",
        _md_table(frame_df.groupby("dataset").apply(
            lambda g: pd.Series({"n_dir_only": int(g["dir_only"].sum()),
                                 "J_dir_only": 100 * g.loc[g["dir_only"], J.format("post")].mean(),
                                 "J_con_lai": 100 * g.loc[~g["dir_only"], J.format("post")].mean()}),
            include_groups=False).reset_index()), "",
        "## 7. Bieu do", "",
    ]
    for ds in datasets:
        for p in sorted((out / "plots" / ds).glob("*.png")):
            lines.append(f"![{ds} {p.stem}](plots/{ds}/{p.name})")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
