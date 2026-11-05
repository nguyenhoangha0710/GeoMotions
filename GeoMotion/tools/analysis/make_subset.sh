#!/usr/bin/env bash
# Tao tap con (symlink) giu nguyen cau truc thu muc cua eval.sh.
# Dung: bash tools/analysis/make_subset.sh <dataset> <video1> [video2 ...]   (DATA_ROOT=data mac dinh)
# Ket qua: data/subset_<dataset>/<duong dan nhu eval.sh>/<video>
set -euo pipefail
ds="$1"; shift
DATA_ROOT="${DATA_ROOT:-data}"
case "$ds" in
  2016)          img="DAVIS/JPEGImages/480p";              ann="DAVIS/Annotations/480p" ;;
  2016-M|2017-M) img="DAVIS2017-M/DAVIS/JPEGImages/480p";  ann="DAVIS2017-M/DAVIS/Annotations/480p" ;;
  segtrack)      img="SegTrackv2/JPEGImages_jpg_standardized"; ann="SegTrackv2/GroundTruth" ;;
  fbms)          img="FBMS59_clean/JPEGImages";             ann="FBMS59_clean/Annotations" ;;
  *) echo "dataset khong hop le: $ds" >&2; exit 1 ;;
esac
out="$DATA_ROOT/subset_${ds}"
mkdir -p "$out/$img" "$out/$ann"
for v in "$@"; do
  [ -d "$DATA_ROOT/$img/$v" ] || { echo "khong thay $DATA_ROOT/$img/$v" >&2; exit 1; }
  ln -sfn "$(cd "$DATA_ROOT/$img/$v" && pwd)" "$out/$img/$v"
  ln -sfn "$(cd "$DATA_ROOT/$ann/$v" && pwd)" "$out/$ann/$v"
done
echo "image_root=$out/$img"
echo "annotation_root=$out/$ann"
echo "data_root (cho run_dump/analyze)=$out"
