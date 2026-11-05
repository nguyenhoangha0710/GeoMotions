import math

from PIL import Image

from tools.analysis.protocols import DATASETS, list_sequences, plan_chunks, sam_order, select_frames


def _names(n):
    return [f"{i:05d}.jpg" for i in range(n)]


def _flat(chunks):
    return [s for c in chunks for s in c]


def test_grouped_50_frames():
    chunks = plan_chunks(_names(50), "grouped", 32)
    assert [c[0].chunk_len for c in chunks] == [13, 13, 12, 12]
    assert [s.orig_idx for s in chunks[0]] == list(range(0, 50, 4))
    assert [s.group for s in chunks[1]] == [1] * 13
    first, second = chunks[0][0], chunks[0][1]
    assert first.is_chunk_start and math.isnan(first.gap_prev)
    assert second.gap_prev == 4 and second.pos_in_chunk == 1
    assert chunks[0][-1].is_chunk_end and math.isnan(chunks[0][-1].gap_next)


def test_grouped_long_splits_chunks():
    chunks = plan_chunks(_names(200), "grouped", 32)
    group0 = [c for c in chunks if c[0].group == 0]
    assert [len(c) for c in group0] == [32, 18]
    idx = sorted(s.orig_idx for s in _flat(chunks))
    assert idx == list(range(200))
    assert [c[0].chunk for c in chunks] == list(range(len(chunks)))


def test_single_frame_chunk():
    chunks = plan_chunks(_names(129), "grouped", 32)
    singles = [c for c in chunks if c[0].chunk_len == 1]
    assert singles
    s = singles[0][0]
    assert s.is_chunk_start and s.is_chunk_end
    assert math.isnan(s.gap_prev) and math.isnan(s.gap_next)


def test_fbms_select_and_order(tmp_path):
    frames, gts = tmp_path / "img", tmp_path / "gt"
    frames.mkdir(); gts.mkdir()
    for stem in ["cars_1", "cars_10", "cars_2"]:
        Image.new("L", (4, 4)).save(gts / f"{stem}.png")
    (frames / "cars_1.jpg").write_bytes(b"x")
    (frames / "cars_2.JPG").write_bytes(b"x")
    (frames / "cars_10.jpg").write_bytes(b"x")
    files = select_frames(DATASETS["fbms"], str(frames), str(gts))
    assert files == ["cars_1.jpg", "cars_2.JPG", "cars_10.jpg"]
    chunks = plan_chunks(files, "fbms", 32)
    assert len(chunks) == 1 and chunks[0][2].gap_prev == 8
    assert [s.orig_idx for s in chunks[0]] == [0, 1, 2]
    assert sam_order(files, "fbms") == ["cars_1.jpg", "cars_10.jpg", "cars_2.JPG"]


def test_list_sequences_filters_val(tmp_path):
    for v in ["blackswan", "bear"]:
        (tmp_path / v).mkdir()
    (tmp_path / "notes.txt").write_text("x")
    assert list_sequences(DATASETS["2016"], str(tmp_path)) == ["blackswan"]
    assert list_sequences(DATASETS["segtrack"], str(tmp_path)) == ["bear", "blackswan"]


def test_dataset_paths_match_eval_sh():
    assert DATASETS["2016"].image_rel == "DAVIS/JPEGImages/480p"
    assert DATASETS["2017-M"].ann_rel == "DAVIS2017-M/DAVIS/Annotations/480p"
    assert len(DATASETS["2016-M"].val_list) == 20 and len(DATASETS["2017-M"].val_list) == 30
    assert DATASETS["fbms"].mode == "fbms" and DATASETS["segtrack"].mode == "grouped"
