import os

from tools.analysis.check_data import check_layout


def test_check_layout_ok(tmp_dataset):
    image_root, ann_root = tmp_dataset
    df = check_layout(image_root, ann_root, None)
    assert list(df["video"]) == ["blackswan", "camel"]
    assert df["ok"].all()
    assert list(df["n_img"]) == [5, 5]


def test_check_layout_missing_ann(tmp_dataset):
    image_root, ann_root = tmp_dataset
    os.remove(os.path.join(ann_root, "camel", "00004.png"))
    df = check_layout(image_root, ann_root, None).set_index("video")
    assert not df.loc["camel", "ok"]
    assert df.loc["camel", "n_ann"] == 4
    assert df.loc["blackswan", "ok"]
