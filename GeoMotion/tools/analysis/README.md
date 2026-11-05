# tools/analysis — chạy lại GeoMotion + lưu dữ liệu phân tích lỗi

Thiết kế: `docs/superpowers/specs/2026-11-05-eval-error-analysis-design.md`
Kế hoạch: `docs/superpowers/plans/2026-11-05-eval-error-analysis.md`

Mọi lệnh chạy **từ thư mục gốc repo** (`~/GeoMotion`) trên server. GPU0 đang bị người khác dùng → mặc định `--gpu 1`.
tmux trên server lỗi → các lệnh dài dùng `nohup ... &` rồi `tail -f <log>`.

## 0. Một lần: kiểm tra code và dữ liệu

```bash
pip install pytest                      # nếu chưa có
python -m pytest tools/analysis/tests -q
```
Kỳ vọng: **mọi test PASS, không có dòng `s` (skip)**. Trên server có torch + `motion_seg_inference`, nên 6 test
cần torch (microbatch, flow, SAM wrapper) phải chạy thật. Nếu thấy `skipped`, chạy `python -m pytest tools/analysis/tests -q -rs`
và gửi lý do skip.

```bash
python tools/analysis/check_data.py --datasets 2016 2016-M 2017-M segtrack fbms
```
Dataset nào `[FAIL]` thì chưa tải / sai cấu trúc; có thể chạy trước các dataset `[OK]`.

### (Tuỳ chọn) Build extension `_C` của SAM2 để khớp tác giả hoàn toàn

```bash
which nvcc && (cd ~/sam2 && SAM2_BUILD_CUDA=1 pip install -e . --no-build-isolation)
python -c "import sam2._C; print('sam2 _C OK')"
```
Không có `nvcc` hoặc build lỗi → bỏ qua; `config.json` sẽ ghi `sam2_has_C: false` và báo cáo nêu khác biệt này.
Kết quả thử: _(ghi vào đây sau khi chạy)_

## 1. Kiểm chứng: Runner phải khớp `eval.py` gốc

Chọn video ngắn (≤ ~64 frame → nhóm ≤ 16 frame): `eval.py` gốc **không có** cách giảm bộ nhớ nên hết bộ nhớ T4
với video dài (ví dụ `parkour` 100 frame → nhóm 25 frame). Runner dùng `install_chunked_flow_forward` (CNN flow +
nội suy theo lô, toán học tương đương — có test `test_chunked_flow_forward_matches_author`) nên chạy được video dài.

```bash
bash tools/analysis/make_subset.sh 2016 blackswan car-shadow

# eval.py gốc, 2 lần (đo độ dao động của SAM2)
for k in a b; do
  CUDA_VISIBLE_DEVICES=1 python eval.py --model_path checkpoint/best_model.pth \
    --pi3_model_path checkpoint/model.safetensors \
    --sam2_checkpoint_path sam2-main/checkpoints/sam2.1_hiera_large.pt \
    --output_dir eval_verify/$k \
    --image_root data/subset_2016/DAVIS/JPEGImages/480p \
    --annotation_root data/subset_2016/DAVIS/Annotations/480p \
    --sequence_length 32 --use_sam_refine True --davis 2016 > eval_verify_$k.log 2>&1
done

# Runner + Analyzer trên cùng tập
python tools/analysis/run_dump.py --datasets 2016 --gpu 1 --run_id verify_2016 --data_root data/subset_2016 \
  > verify_dump.log 2>&1
python tools/analysis/analyze.py --run_id verify_2016 --data_root data/subset_2016 --no_media

# So sánh
python tools/analysis/verify_against_eval.py --eval_csv eval_verify/a/sequence_metrics.csv \
  --frame_table results/verify_2016/analysis/frame_table.csv --dataset 2016
python tools/analysis/verify_against_eval.py --eval_csv eval_verify/a/sequence_metrics.csv \
  --frame_table results/verify_2016/analysis/frame_table.csv --dataset 2016 --tol 1   # chỉ để xem mức lệch
```
- `PASS` → làm tiếp bước 2.
- `FAIL`: so `eval_verify/a` với `eval_verify/b` (dao động của chính `eval.py`). Nếu lệch của Runner ≤ dao động a↔b
  thì chấp nhận, ghi mức đó vào đây. Nếu lớn hơn → **dừng**, gửi output cho Claude.

### 1b. Kiểm chứng các dataset còn lại (khi đã tải)

FBMS dùng logic khác hẳn DAVIS (chỉ frame có GT, 1 nhóm, ghép GT theo tên), nên phải kiểm chứng riêng.
Chọn 2 video mỗi dataset (1 ngắn, 1 dài), ví dụ:

```bash
verify () {  # verify <dataset> <video1> <video2>
  ds=$1; shift
  bash tools/analysis/make_subset.sh $ds "$@" > subset_$ds.txt
  IMG=$(grep image_root subset_$ds.txt | cut -d= -f2); ANN=$(grep annotation_root subset_$ds.txt | cut -d= -f2)
  ROOT=$(grep "data_root" subset_$ds.txt | cut -d= -f2)
  CUDA_VISIBLE_DEVICES=1 python eval.py --model_path checkpoint/best_model.pth \
    --pi3_model_path checkpoint/model.safetensors \
    --sam2_checkpoint_path sam2-main/checkpoints/sam2.1_hiera_large.pt \
    --output_dir eval_verify/$ds --image_root $IMG --annotation_root $ANN \
    --sequence_length 32 --use_sam_refine True --davis $ds > eval_verify_$ds.log 2>&1
  python tools/analysis/run_dump.py --datasets $ds --gpu 1 --run_id verify_$ds --data_root $ROOT > verify_dump_$ds.log 2>&1
  python tools/analysis/analyze.py --run_id verify_$ds --data_root $ROOT --no_media
  python tools/analysis/verify_against_eval.py --eval_csv eval_verify/$ds/sequence_metrics.csv \
    --frame_table results/verify_$ds/analysis/frame_table.csv --dataset $ds
}
verify fbms <video_ngan> <video_dai>        # xem tên video: ls data/FBMS59_clean/JPEGImages
verify segtrack <video_ngan> <video_dai>
verify 2017-M dogs-jump india
```

Nếu FBMS lệch: kiểm tra tên frame có đệm số 0 không (`ls data/FBMS59_clean/JPEGImages/<video> | head`).
Tác giả sắp thứ tự frame cho SAM2 bằng sort chuỗi thường; tên kiểu `cars1_1, cars1_10, cars1_2` sẽ làm SAM2
lan truyền theo thứ tự lộn xộn — Runner tái hiện y hệt điều này (để khớp số), nhưng đây là điểm đáng ghi nhận.

Kết quả kiểm chứng: _(ghi vào đây)_

## 2. Chạy toàn bộ

```bash
RUN=2026-11-05_cvpr_seq32
nohup python tools/analysis/run_dump.py --datasets 2016 2016-M 2017-M segtrack fbms --gpu 1 --run_id $RUN \
  > dump_$RUN.log 2>&1 &
tail -f dump_$RUN.log
```
- Chạy lại cùng lệnh sẽ **bỏ qua video đã xong** (`done.flag`) → dùng được khi bị ngắt.
- Video lỗi (ví dụ hết bộ nhớ) ghi ở `results/$RUN/failed.csv`; các video khác vẫn chạy tiếp.
- GPU0 rảnh thì chia dataset: một lệnh `--gpu 1 --datasets 2016 2016-M 2017-M`, một lệnh `--gpu 0 --datasets segtrack fbms`
  (cùng `--run_id`).
- Ước tính ~5–6 giây/frame trên T4.

## 3. Phân tích (CPU, chạy lại bao nhiêu lần cũng được)

```bash
python tools/analysis/analyze.py --run_id $RUN
```
Kết quả trong `results/$RUN/analysis/`:

| File | Nội dung |
|---|---|
| `metrics_summary.csv` | J/F/JF/JR theo dataset × (trước/sau SAM2) × (cách tác giả / chuẩn DAVIS), so với paper |
| `frame_table.csv` | 1 dòng / frame: metric, nhãn lỗi, đặc tính flow/camera, vị trí trong đoạn |
| `video_table.csv` | 1 dòng / video, xếp từ tệ nhất |
| `sensitivity.csv` | phân bố nhãn khi đổi ngưỡng `tau_good` |
| `plots/<dataset>/` | 6 biểu đồ |
| `gallery/<dataset>/<lý do>/` | ảnh ghép 7 ô: RGB, GT, xác suất trước SAM2, mask sau SAM2, overlay lỗi (xanh lá TP, cam FP, xanh dương FN), độ lớn flow, hướng flow |
| `videos/<dataset>/` | mp4 cho 5 video tệ nhất |
| `report.md` | báo cáo tổng |

## 4. Đưa kết quả về máy

Chỉ cần thư mục `results/$RUN/analysis/` (vài trăm MB). Trong VS Code: chuột phải thư mục → **Download...** → lưu vào
`D:\NguyenHoangHa_nam4\NCKH\Motion Segmentation Research\results\<RUN>\analysis`.
Dữ liệu thô (`results/$RUN/<dataset>/...`) để lại trên server.
