"""Build the SegTrackv2 layout expected by eval.py from the RCF archive (SegTrackv2_all.tgz).

Input  (RCF):  <src>/JPEGImages/<seq>/*.{png,bmp}   <src>/Annotations/<seq>/*.{png,bmp} (+ per-object subdirs)
Output:        <dst>/JPEGImages_jpg_standardized/<seq>/<stem>.png   <dst>/GroundTruth/<seq>/<stem>.png

- Images and GT are re-encoded losslessly as PNG (bmp is not read by eval.py; pixel values unchanged).
- The merged GT file at the sequence level is used; per-object subfolders are ignored
  (checked on the server: merged == union of objects up to tens of boundary pixels).
- GT values are kept as-is (anti-aliased edges included); eval.py binarises with > 0.
- Images and GT are paired by file stem; GT without an image is dropped (e.g. worm/0000000000),
  because eval.py pairs by sorted order and one extra file would shift every frame.

Usage: python tools/analysis/prepare_segtrack.py --src ~/GeoMotion/data/SegTrackv2 --dst ~/GeoMotion/data/SegTrackv2_std
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

EXTS = (".png", ".bmp", ".jpg", ".jpeg")


def _files(d: Path) -> dict:
    return {p.stem: p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in EXTS}


def _to_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
    else:
        Image.open(src).save(dst)  # lossless re-encode, mode and values preserved


def prepare_segtrack(src, dst) -> pd.DataFrame:
    src, dst = Path(src), Path(dst)
    img_out, gt_out = dst / "JPEGImages_jpg_standardized", dst / "GroundTruth"
    rows = []
    for seq in sorted(p.name for p in (src / "JPEGImages").iterdir() if p.is_dir()):
        imgs = _files(src / "JPEGImages" / seq)
        gts = _files(src / "Annotations" / seq)
        missing = sorted(set(imgs) - set(gts))
        if missing:
            raise ValueError(f"{seq}: {len(missing)} image(s) without GT, e.g. {missing[:3]}")
        dropped = sorted(set(gts) - set(imgs))
        for d in (img_out / seq, gt_out / seq):
            if d.exists():
                shutil.rmtree(d)
        for stem, p in imgs.items():
            _to_png(p, img_out / seq / f"{stem}.png")
            _to_png(gts[stem], gt_out / seq / f"{stem}.png")
        rows.append({"video": seq, "n_img": len(imgs), "n_gt": len(imgs),
                     "img_ext": sorted({p.suffix.lower() for p in imgs.values()})[0],
                     "dropped_gt": ",".join(dropped)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args(argv)
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("--dst must differ from --src")
    report = prepare_segtrack(args.src, args.dst)
    print(report.to_string(index=False))
    print(f"\n{len(report)} videos, {int(report['n_img'].sum())} frames -> {args.dst}")


if __name__ == "__main__":
    main()
