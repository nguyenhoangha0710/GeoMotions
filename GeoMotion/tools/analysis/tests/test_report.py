import shutil

import numpy as np
import pandas as pd
import pytest

from tools.analysis.analyze import add_dir_only, analyze_sequence, summarize
from tools.analysis.protocols import DATASETS
from tools.analysis.report import (build_report, flow_to_color, longest_bad_run, render_panel, select_gallery,
                                   video_table)
from tools.analysis.tests.test_analyze import fake_run  # noqa: F401  (fixture)


def test_longest_bad_run():
    assert longest_bad_run([0.9, 0.4, 0.3, 0.8, 0.2]) == 2
    assert longest_bad_run([]) == 0


def test_flow_to_color():
    vec = np.zeros((2, 1, 3), np.float32)
    vec[0, 0, 0], vec[0, 0, 1] = 1.0, -1.0
    vec[:, 0, 2] = np.nan
    img = flow_to_color(vec)
    assert img.shape == (1, 3, 3) and img.dtype == np.uint8
    assert not np.array_equal(img[0, 0], img[0, 1])
    assert img[0, 0].max() == img[0, 1].max()
    assert img[0, 2].tolist() == [0, 0, 0]


def test_render_panel_shape():
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), np.uint8)
    m = np.zeros((h, w), np.float32)
    panel = render_panel(rgb, m > 0, m, m, m, np.zeros((2, h, w), np.float32))
    assert panel.shape == (48, 448, 3)


def test_select_gallery_caps():
    rng = np.random.default_rng(0)
    labels = ["good", "under", "over", "missed", "mixed"]
    df = pd.DataFrame({
        "dataset": "2016", "video": [f"v{i % 15}" for i in range(300)],
        "frame_name": [f"{i:05d}" for i in range(300)], "J_post_0.1": rng.random(300),
        "label_post": [labels[i % 5] for i in range(300)],
        "sam_group": ["sam_broke" if i % 7 == 0 else "both_good" for i in range(300)],
    })
    g = select_gallery(df)
    assert not g.duplicated(["video", "frame_name"]).any()
    by_label = g[g["reason"].str.startswith("label:")].groupby("reason").size()
    assert (by_label <= 20).all()
    assert (g["reason"] == "sam_broke").sum() <= 20


def test_video_table_sorted(fake_run):  # noqa: F811
    run_dir, data_root = fake_run
    df = analyze_sequence(run_dir, DATASETS["2016"], data_root, "blackswan")
    vt = video_table(df)
    assert list(vt["video"]) == ["blackswan"]
    assert {"JF_pre", "JF_post", "JR", "min_J", "longest_bad_run"} <= set(vt.columns)


@pytest.mark.parametrize("media", [False, True])
def test_build_report_smoke(fake_run, media):  # noqa: F811
    if media and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    run_dir, data_root = fake_run
    (run_dir / "config.json").write_text('{"sam2_has_C": false, "sequence_length": 32}')
    df = add_dir_only(analyze_sequence(run_dir, DATASETS["2016"], data_root, "blackswan"))
    build_report(run_dir, df, summarize(df), ["2016"], data_root, media=media)
    out = run_dir / "analysis"
    assert (out / "report.md").exists() and (out / "video_table.csv").exists()
    assert any((out / "plots").rglob("*.png"))
    if media:
        assert any((out / "gallery").rglob("*.jpg"))
        assert any((out / "videos").rglob("*.mp4"))
