"""Check that evaluation datasets follow the layout expected by eval.py."""
import argparse
import os

import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _count(path: str, exts) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.lower().endswith(exts))


def check_layout(image_root: str, ann_root: str, sequences=None, require_equal: bool = True) -> pd.DataFrame:
    """One row per video: number of images, number of annotations, ok flag.

    require_equal=False is used for FBMS, whose GT is sparse.
    """
    if sequences is None:
        sequences = sorted(
            d for d in (os.listdir(image_root) if os.path.isdir(image_root) else [])
            if os.path.isdir(os.path.join(image_root, d))
        )
    rows = []
    for video in sequences:
        n_img = _count(os.path.join(image_root, video), IMG_EXTS)
        n_ann = _count(os.path.join(ann_root, video), (".png", ".jpg", ".jpeg"))
        ok = n_img > 0 and n_ann > 0 and (n_img == n_ann if require_equal else True)
        rows.append({"video": video, "n_img": n_img, "n_ann": n_ann, "ok": ok})
    return pd.DataFrame(rows, columns=["video", "n_img", "n_ann", "ok"])


def main(argv=None):
    from tools.analysis.protocols import DATASETS, list_sequences

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--data_root", default="data")
    args = ap.parse_args(argv)
    all_ok = True
    for key in args.datasets:
        spec = DATASETS[key]
        image_root = os.path.join(args.data_root, spec.image_rel)
        ann_root = os.path.join(args.data_root, spec.ann_rel)
        print(f"\n=== {key}: {image_root}")
        if not os.path.isdir(image_root) or not os.path.isdir(ann_root):
            print(f"[FAIL] thieu thu muc: {image_root} hoac {ann_root}")
            all_ok = False
            continue
        seqs = list_sequences(spec, image_root)
        df = check_layout(image_root, ann_root, seqs, require_equal=(spec.mode == "grouped"))
        bad = df[~df["ok"]]
        print(f"{'[OK]  ' if bad.empty else '[FAIL]'} {len(df)} video, {int(df['n_img'].sum())} anh, "
              f"{int(df['n_ann'].sum())} nhan")
        if spec.val_list is not None:
            missing = sorted(set(spec.val_list) - set(seqs))
            if missing:
                print("  thieu video:", " ".join(missing))
                all_ok = False
        if not bad.empty:
            print(bad.to_string(index=False))
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    raise SystemExit(main())
