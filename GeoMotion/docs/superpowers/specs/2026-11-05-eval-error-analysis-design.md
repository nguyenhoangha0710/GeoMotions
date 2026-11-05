# Thiết kế: Đánh giá lại GeoMotion và lưu dữ liệu phân tích lỗi

- Ngày: 2026-11-05
- Người yêu cầu: Nguyễn Hoàng Hà
- Trạng thái: chờ duyệt spec
- Phạm vi code: thư mục mới `tools/analysis/`, **không sửa file nào của tác giả** (ngoại lệ duy nhất: xoá dòng import thừa `split_components` trong `eval.py`, xem §9)

## 1. Mục tiêu

1. **Tái tạo kết quả**: chạy checkpoint CVPR (`checkpoint/best_model.pth`) trên 5 benchmark của paper — DAVIS16, DAVIS16-M, DAVIS17-M, SegTrackV2, FBMS-59 — theo đúng giao thức của `eval.py`/`eval.sh`, rồi so với Bảng 1 của paper.
2. **Lưu dữ liệu phân tích lỗi** đủ để trả lời, cho từng frame: sai ở đâu, sai kiểu gì, và có thể vì sao. Kết quả phân tích làm bằng chứng cho research gap (đặc biệt gap "chuyển động yếu" và "mất hướng flow").

### Không thuộc phạm vi
- Train lại, sửa kiến trúc, bản Full GeoMotion.
- Tối ưu tốc độ (ví dụ biên dịch RoPE2D CUDA) — chỉ ghi nhận.
- Chạy baseline khác (SegAnyMotion, ABR...).

### Tiêu chí thành công
- Có `metrics_summary.csv` so sánh với Bảng 1 cho cả 5 dataset.
- Bước kiểm chứng (§6) đạt: J và F của từng video lệch < 0.001 so với `eval.py` gốc (hoặc trong mức dao động đo được của chính `eval.py` nếu SAM2 không tất định).
- Có `frame_table.csv` chứa đầy đủ metric + nhãn lỗi + đặc tính + vị trí cho mọi frame được đánh giá, và báo cáo `report.md` kèm biểu đồ, ảnh lỗi.

## 2. Ràng buộc môi trường

- Server khoa: container Docker, user `hoangha`, repo `~/GeoMotion`, ổ trống 6.7 TB, có Internet.
- 2 × Tesla T4 14.6 GB (sm_75). **GPU0 đang bị người khác dùng** (~3 GB, 99%) → mặc định chạy GPU1; GPU0 chỉ dùng khi rảnh.
- Python 3.14 (conda base), torch 2.9.0+cu128, torchvision 0.24.0. bf16 autocast chạy được.
- SAM2 cài từ GitHub, **không có extension `_C`** → SAM2 bỏ bước lấp lỗ < ~8 px. Tác giả có `_C.so`. Ghi nhận là khác biệt; xử lý ở §9.
- Model chạy fp32 (không autocast). Đỉnh bộ nhớ đo được: N=8 → 5.8 GB, N=16 → 9.8 GB, N=32 → OOM (tensor output flow CNN `[32,128,518,518]` fp32 = 4.1 GiB).
- Tốc độ đo trên `blackswan` (GPU1): ~5.4 s/frame gồm model + SAM2.
- Claude không SSH được vào server. Quy trình: code viết trong repo trên máy Hà → push lên fork `nguyenhoangha0710/GeoMotion` → pull trên server → chạy → copy thư mục `analysis/` về máy.
- tmux trên server lỗi → chạy dài dùng `nohup`.

## 3. Kiến trúc

Phương án "chạy một lần, phân tích nhiều lần":

```
tools/analysis/
├── protocols.py     # chọn video, chọn frame, chia nhóm/chunk — chép đúng logic eval.py
├── run_dump.py      # Runner (GPU): chạy model + SAM2, lưu dữ liệu thô. Không tính metric.
├── analyze.py       # Analyzer (CPU): metric, nhãn lỗi, đặc tính, báo cáo
├── taxonomy.py      # hàm gán nhãn lỗi (thuần numpy, có unit test)
├── features.py      # tính đặc tính frame (diện tích, tương phản flow, camera)
├── report.py        # bảng video, biểu đồ, ảnh ghép, mp4, report.md
└── tests/           # unit test cho protocols, taxonomy, features
```

Ranh giới: Runner chỉ phụ thuộc code tác giả + `protocols.py`; Analyzer chỉ đọc file do Runner ghi + GT + `core/eval/eval_mask.py` của tác giả. Hai phần giao tiếp hoàn toàn qua thư mục `results/`.

## 4. Giao thức (phải giống hệt `eval.py`)

| Dataset key | Ảnh / nhãn (theo eval.sh) | Chọn video | Chọn frame & chia nhóm |
|---|---|---|---|
| `2016` | `data/DAVIS/.../480p` | 20 video val DAVIS16 | mọi frame; **4 nhóm xen kẽ** (stride 4); chunk ≤ `sequence_length` |
| `2016-M` | `data/DAVIS2017-M/DAVIS/.../480p` | 20 video (danh sách 2016) | như trên |
| `2017-M` | như trên | 30 video val DAVIS17 | như trên |
| `segtrack` | `data/SegTrackv2/JPEGImages_jpg_standardized`, `GroundTruth` | mọi thư mục | như trên |
| `fbms` | `data/FBMS59_clean/JPEGImages`, `Annotations` | mọi thư mục | **chỉ frame có GT** (`rgb_from_gt=True`); **1 nhóm duy nhất**; chunk ≤ `sequence_length` |

Chi tiết chung:
- Ảnh được resize về kích thước GT (bilinear) rồi `preprocess_video` resize thẳng về 518×518.
- Model: `MotionSegmentationInference.predict_motion_mask` → xác suất `[N,518,518]`.
- SAM2 (`refine_sam`, SAM2.1-L video predictor): chạy trên **toàn bộ frame của video** sau khi ghép các nhóm. Cơ chế: frame lẻ (ngưỡng 0.8) làm prompt → lấy kết quả cho frame chẵn; rồi đảo lại. ⇒ mask sau SAM2 của frame t **không dùng trực tiếp** dự đoán của frame t.
- Metric: `db_eval_iou`, `db_eval_boundary(bound_th=0.008)`; mask dự đoán được resize về kích thước GT qua uint8 + PIL bilinear (`_resize_mask`); ngưỡng `> 0.1`; GT `> 0`.
- `sequence_length` mặc định 32, ghi vào `config.json`.

`protocols.py` chép lại logic trên thành hàm thuần (đầu vào: danh sách tên file; đầu ra: danh sách nhóm/chunk) để unit test được.

## 5. Runner (`run_dump.py`)

### 5.1 Luồng xử lý một video
1. Gọi `protocols.py` lấy danh sách frame, nhóm, chunk.
2. Với mỗi chunk:
   a. Tiền xử lý ảnh y như `predict_motion_mask`.
   b. **Tự tính flow** bằng `compute_optical_flow_raft` của tác giả cho mọi cặp kề (forward + backward), giữ vector `(u,v)`; tính độ lớn và gộp `max` y như `compute_optical_flow_magnitude_batch` (không gọi lại RAFT lần hai).
   c. Gọi `model(video_tensor, flow_magnitudes)` như `predict_motion_mask`. Output dict của model có sẵn `motion_mask`, `camera_poses` `[B,N,4,4]` và `local_points` `[B,N,H,W,3]`; lấy `motion_mask`, `camera_poses`, và độ sâu trung vị mỗi frame = median của `local_points[...,2]` (chỉ lưu con số, không lưu point map). Pose chỉ có nghĩa **trong cùng một chunk** (π³ không có hệ toạ độ chung giữa các chunk).
   d. **Giảm bộ nhớ**: chạy `motion_aware_decoder.flow_encoder` theo lô con (mặc định 8 frame) bằng wrapper bao quanh module, ghép kết quả theo chiều batch. Toán học tương đương (CNN xử lý từng frame độc lập). Bật bằng `--flow_cnn_microbatch` (mặc định 8; 0 = tắt).
3. Ghép xác suất các nhóm theo chỉ số gốc → gọi `refine_sam` y như `_refine_with_sam` → mask sau SAM2.
4. Ghi file (§5.2). Ghi `done.flag` cuối cùng.

### 5.2 Định dạng output
```
results/<run_id>/
├── config.json                 # checkpoint (đường dẫn + sha256 8 ký tự), git commit, sequence_length,
│                               # flow_cnn_microbatch, GPU, phiên bản python/torch/torchvision/sam2,
│                               # sam2 có _C hay không, thời gian bắt đầu/kết thúc
├── failed.csv                  # dataset, video, lỗi, traceback rút gọn
└── <dataset>/<video>/
    ├── prob_pre_sam/<frame>.png    # xác suất trước SAM2, uint8 = (p*255).astype(uint8) (cắt cụt, giống _resize_mask), 518×518
    ├── mask_post_sam/<frame>.png   # mask sau SAM2, 0/255, kích thước 518×518
    ├── flow.npz                    # fw, bw: float16 [N,2,259,259] (nửa res, đã nhân hệ số 0.5); NaN nếu không có
    ├── poses.npz                   # camera_poses float32 [N,4,4] (hệ toạ độ riêng từng chunk)
    ├── frames.csv                  # mỗi dòng 1 frame (cột ở dưới)
    ├── timing.json                 # thời gian model, SAM2
    └── done.flag
```
Cột `frames.csv`: `frame_name, orig_idx, group, chunk, pos_in_chunk, chunk_len, is_chunk_start, is_chunk_end, gap_prev, gap_next` (khoảng cách theo số frame gốc), `flow_mag_mean, flow_mag_median, flow_mag_p95` (độ lớn đã gộp max, toàn ảnh), `has_fw, has_bw`, `depth_median` (π³, đơn vị tương đối), `sam_no_prompt`.

Ghi chú: lưu uint8 không làm sai lệch metric vì `_resize_mask` của tác giả cũng chuyển uint8.

### 5.3 CLI
```
python tools/analysis/run_dump.py --datasets 2016 2016-M --gpu 1 --run_id 2026-11-05_cvpr_seq32 \
    [--videos blackswan camel] [--sequence_length 32] [--flow_cnn_microbatch 8] [--no_sam]
```
Đường dẫn dataset mặc định theo `eval.sh`, ghi đè được bằng `--data_root`.

### 5.4 Xử lý sự cố
- **Resume**: bỏ qua video đã có `done.flag`.
- **OOM / lỗi khác**: ghi `failed.csv`, giải phóng bộ nhớ, chạy tiếp video sau. Không tự động giảm `sequence_length` (sẽ làm sai giao thức) — chỉ báo.
- **SAM2 không có prompt nào** (mọi frame < 0.8): `propagate_in_video` có thể lỗi → bắt lỗi, ghi mask sau SAM2 = rỗng, đánh dấu `sam_no_prompt=1` trong `frames.csv`, ghi cảnh báo. Hành vi này **khác** `eval.py` (vốn sẽ crash), cần ghi rõ trong báo cáo.

## 6. Kiểm chứng tái tạo

1. Tập nhỏ: 2 video/dataset (DAVIS16: `blackswan`, `parkour`; các dataset khác chọn 1 ngắn + 1 dài khi có dữ liệu), tạo bằng symlink.
2. Chạy `eval.py` gốc trên tập nhỏ → `sequence_metrics.csv`.
3. Chạy Runner + Analyzer (giao thức "tác giả") trên cùng tập.
4. **Đạt** nếu |ΔJ|, |ΔF| < 0.001 cho mọi video.
5. Đo độ tất định: chạy `eval.py` gốc 2 lần trên `blackswan`. Nếu chính nó dao động > 0.001, ngưỡng đạt = mức dao động đó, ghi rõ trong báo cáo.
6. Kiểm tra riêng: flow magnitude do Runner tính khớp `compute_optical_flow_magnitude_batch` (max |Δ| < 1e-4 px) trên 1 chunk; output model có/không microbatch khớp (max |Δp| < 1e-5).

Không đạt → dừng, tìm nguyên nhân trước khi chạy toàn bộ.

## 7. Analyzer (`analyze.py`)

### 7.1 Metric từng frame (`frame_metrics` → nằm trong `frame_table.csv`)
Cho mỗi frame có GT, cho cả `pre` (xác suất trước SAM2) và `post` (mask sau SAM2), ở ngưỡng 0.1 và 0.5: J, F, precision, recall, diện tích dự đoán.

### 7.2 Tổng hợp (`metrics_summary.csv`)
| Cách | Gộp | Ngưỡng |
|---|---|---|
| `author` | trung bình mọi frame của mọi video | 0.1 |
| `davis_std` | trung bình trong video → trung bình giữa video | 0.5 |

Cột: `dataset, refine(pre/post), protocol, J, F, JF, JR, paper_JF, diff`. Giá trị paper lấy từ bảng hằng trong code (DAVIS16-M 83.9; DAVIS17-M 81.1 (J); FBMS J 72.5 / F 78.5; DAVIS16 84.7; SegTrackV2 77.3) — ghi chú rằng ánh xạ cột Bảng 1 cần đối chiếu PDF.

### 7.3 Nhãn lỗi (B) — `taxonomy.py`
P = dự đoán nhị phân, G = GT. Xét theo thứ tự, gán nhãn đầu tiên khớp. Các ngưỡng là tham số (mặc định trong ngoặc):

| Nhãn | Điều kiện |
|---|---|
| `empty_ok` | \|G\|=0 và \|P\| < `tau_area` (0.1% ảnh) |
| `false_alarm` | \|G\|=0 và \|P\| ≥ `tau_area` |
| `missed` | \|G\|>0, recall < `tau_miss` (0.1), \|P\| < `tau_area` |
| `wrong_location` | \|G\|>0, recall < `tau_miss`, \|P\| ≥ `tau_area` |
| `good` | J ≥ `tau_good` (0.7) |
| `under` | J < 0.7, precision ≥ `tau_pr` (0.8), recall < 0.8 |
| `over` | J < 0.7, recall ≥ 0.8, precision < 0.8 |
| `mixed` | còn lại |

Mức instance: đọc GT gốc (palette/grayscale) trước khi nhị phân hoá; mỗi ID ≠ 0 là một vật. Cột `n_inst`, `n_inst_missed` (recall vật < 0.5). DAVIS16 chỉ có 1 vật.

Độ nhạy: báo cáo kèm bảng phân bố nhãn khi `tau_good` ∈ {0.5, 0.7, 0.8}.

### 7.4 SAM2 (A)
`dJ = J_post − J_pre` (ngưỡng 0.1). Nhóm: `sam_rescued` (pre < 0.7 ≤ post), `sam_broke` (pre ≥ 0.7 > post), `both_good`, `both_bad`. Ma trận chuyển nhãn pre → post. Báo cáo nhắc rằng mask post của frame t được lan truyền từ prompt của các frame khác (§4).

### 7.5 Đặc tính (C) — `features.py`
Dùng GT resize về 259×259 (nearest) cho khớp flow nửa res.
- `obj_area` (tỉ lệ), `n_inst`.
- Vùng vật = G co 3 px; vùng nền = phần ngoài G nới 10 px (bỏ qua nếu vùng rỗng).
- `mag_obj`, `mag_bg`: trung vị độ lớn flow đã gộp max (tái tạo từ fw/bw).
- `mag_contrast = mag_obj / (mag_bg + 0.5)`.
- `vec_contrast = ‖mean(u,v)_obj − mean(u,v)_bg‖` (dùng fw; frame cuối dùng −bw).
- `dir_only` = 1 nếu `mag_contrast` thuộc tứ phân vị thấp nhất **và** `vec_contrast` thuộc tứ phân vị cao nhất của dataset (ứng viên "magnitude làm mất thông tin").
- Camera: `cam_rot_deg` (góc quay tương đối với frame kề **trong cùng chunk**), `cam_trans_rel` = ‖Δt‖ / `depth_median` (bất biến tỉ lệ). Frame đầu chunk dùng frame kề sau. Chunk 1 frame → NaN.

### 7.6 Vị trí (D)
Lấy thẳng từ `frames.csv`: `group, chunk, pos_in_chunk, chunk_len, is_chunk_start, is_chunk_end, gap_prev, gap_next`.

### 7.7 Output Analyzer
```
results/<run_id>/analysis/
├── metrics_summary.csv
├── frame_table.csv          # 1 dòng / frame: định danh + metric + nhãn + đặc tính + vị trí
├── video_table.csv          # 1 dòng / video, sắp xếp từ tệ nhất (§8)
├── sensitivity.csv
├── plots/<dataset>/*.png
├── gallery/<dataset>/<label>/<video>_<frame>.jpg
├── videos/<dataset>/<video>.mp4
└── report.md
```
Toàn bộ thư mục `analysis/` là thứ duy nhất cần copy về máy (ước tính vài trăm MB).

CLI: `python tools/analysis/analyze.py --run_id <id> [--datasets ...] [--no_media]`. Chạy lại được bất cứ lúc nào, không cần GPU.

## 8. Báo cáo (E) — `report.py`

- `video_table.csv`: JF pre/post, JR, số frame mỗi nhãn, J nhỏ nhất, **chuỗi frame tệ liên tiếp dài nhất** (J_post < 0.5), trung bình đặc tính.
- Biểu đồ (mỗi dataset):
  1. J theo thời gian (pre & post) cho 10 video tệ nhất, vạch ranh giới chunk, đánh dấu frame GT rỗng.
  2. Phân bố nhãn lỗi pre vs post (cột chồng).
  3. J theo nhóm đặc tính (diện tích, `mag_contrast`, `vec_contrast`, camera, `gap`) kèm số frame/nhóm.
  4. Scatter `mag_contrast` × `vec_contrast`, màu theo J_post.
  5. J theo `pos_in_chunk` / `is_chunk_start|end` / `chunk_len`.
  6. Heatmap ma trận chuyển nhãn qua SAM2.
- Ảnh ghép 7 ô: RGB | GT | prob pre (colormap) | mask post | overlay lỗi (TP xanh lá, FP đỏ, FN xanh dương) | độ lớn flow | hướng flow (bánh xe màu HSV). Chọn: mỗi nhãn ≤ 20 frame J thấp nhất; 3 frame tệ nhất của mỗi video trong top 10; ≤ 20 frame `sam_broke`. Giới hạn chỉnh bằng tham số.
- mp4 (ffmpeg, H.264 `-preset veryfast`) cho 5 video tệ nhất mỗi dataset.
- `report.md`: bảng so sánh paper, khác biệt cấu hình (§9), phát hiện chính có số liệu, link biểu đồ, danh sách video tệ nhất.

## 9. Khác biệt đã biết so với tác giả (ghi vào config.json và report.md)

| Mục | Tác giả | Ở đây | Xử lý |
|---|---|---|---|
| GPU | RTX 5090 | T4 | Chỉ ảnh hưởng tốc độ |
| SAM2 `_C` | có | không | Thử build extension (`pip install -e .` trong `~/sam2` với `SAM2_BUILD_CUDA=1`, cần nvcc). Nếu không build được: ghi nhận; ước lượng ảnh hưởng bằng cách so post-SAM trên tập nhỏ nếu sau này build được |
| Flow CNN | 1 lô | lô con 8 frame | Tương đương toán học, kiểm chứng §6 |
| `eval.py` | import `split_components` (không tồn tại) | xoá dòng import | Không đổi hành vi |
| Frame không có prompt SAM2 | crash | mask rỗng + cờ | Ghi rõ |
| Python / thư viện | 3.12, torch 2.9 cu128 | 3.14, torch 2.9 cu128 | Ghi phiên bản |

## 10. Kiểm thử

- `tests/test_protocols.py`: với danh sách tên frame giả (50, 84, 7 frame; FBMS kiểu GT thưa), nhóm/chunk/vị trí khớp logic `eval.py` (so với hàm tham chiếu chép từ `eval.py`).
- `tests/test_taxonomy.py`: mask nhỏ tự tạo cho từng nhãn, kể cả trường hợp biên (G rỗng, P rỗng, J đúng 0.7).
- `tests/test_features.py`: flow tổng hợp (vật dịch phải, nền đứng yên; vật cùng độ lớn nhưng ngược hướng với nền) → `mag_contrast`, `vec_contrast` đúng kỳ vọng.
- Kiểm chứng tích hợp: §6.
- Chạy được bằng `python -m pytest tools/analysis/tests` trên CPU.

## 11. Quy trình chạy

1. (Song song, không phụ thuộc code mới) Chạy `eval.py` gốc toàn bộ DAVIS16 trên GPU1 bằng `nohup` → số liệu tham chiếu.
2. Tải các dataset còn lại vào `data/` theo cấu trúc README.
3. Pull code mới; `pytest`.
4. Kiểm chứng §6.
5. Chạy Runner toàn bộ (`nohup`, GPU1; thêm GPU0 nếu rảnh, chia dataset).
6. Chạy Analyzer → copy `results/<run_id>/analysis/` về `Motion Segmentation Research/` trên máy.

Ước tính: dữ liệu thô 3–5 GB; thời gian Runner ~5–6 s/frame → DAVIS16 ~2 giờ, cả 5 dataset ~8–10 giờ trên 1 GPU.

## 12. Rủi ro & câu hỏi mở

- Dataset DAVIS2017-M, SegTrackV2, FBMS59 chưa tải; nguồn: nhãn motion DAVIS2017 từ OCLR, SegTrackV2/FBMS theo tiền xử lý motiongrouping (có thể có bản sẵn trên HF `longlian/RCF-UnsupVideoSeg-Datasets` — cần kiểm tra cấu trúc khớp README).
- Số frame thực tế của FBMS/SegTrackV2 có thể làm thời gian chạy khác ước tính.
- Pose π³ có thể kém tin cậy khi vật chuyển động chiếm phần lớn ảnh → đặc tính camera chỉ dùng để phân nhóm, không coi là ground truth.
- SAM2 có thể không tất định trên GPU → xử lý ở §6 bước 5.
