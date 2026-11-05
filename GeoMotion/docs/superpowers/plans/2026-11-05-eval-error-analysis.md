# Đánh giá lại GeoMotion + dữ liệu phân tích lỗi — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm `tools/analysis/` để chạy lại GeoMotion đúng giao thức `eval.py`, lưu dữ liệu thô từng frame, rồi tính metric, gán nhãn lỗi, đặc tính và báo cáo.

**Architecture:** Runner (GPU, `run_dump.py`) gọi lại code tác giả theo giao thức tách ra trong `protocols.py`, ghi `results/<run_id>/<dataset>/<video>/`. Analyzer (CPU, `analyze.py` + `taxonomy.py` + `features.py` + `report.py`) chỉ đọc thư mục đó và GT. Hai phần giao tiếp hoàn toàn qua file.

**Tech Stack:** Python 3.14, numpy, pandas, Pillow, opencv, scipy, matplotlib, torch 2.9 / torchvision 0.24 (chỉ Runner), SAM2, pytest, ffmpeg.

**Spec:** `docs/superpowers/specs/2026-11-05-eval-error-analysis-design.md`

## Global Constraints

- Không sửa file của tác giả, trừ: xoá dòng `    split_components,` trong `eval.py` (Task 1).
- Mọi code mới nằm trong `tools/analysis/`; test trong `tools/analysis/tests/`; chạy test bằng `python -m pytest tools/analysis/tests -q` từ gốc repo; test không cần GPU (test cần torch dùng `pytest.importorskip("torch")`).
- Giao thức phải giống `eval.py`: stride nhóm = 4 (DAVIS/SegTrack), FBMS 1 nhóm + `rgb_from_gt=True`; `sequence_length` mặc định 32; ảnh resize về kích thước GT (PIL BILINEAR) rồi 518×518 (LANCZOS, trong `preprocess_video`).
- Metric: `db_eval_iou`, `db_eval_boundary(bound_th=0.008)` từ `core/eval/eval_mask.py`; dự đoán → GT size qua `(m*255).astype(np.uint8)` + PIL BILINEAR + `/255`; ngưỡng tác giả `> 0.1`; GT `> 0`.
- Mọi phân tích A/B (nhãn lỗi, nhóm SAM2) dùng ngưỡng 0.1 (giao thức tác giả); metric báo thêm ở 0.5.
- Ngưỡng nhãn mặc định: `tau_area=0.001`, `tau_miss=0.1`, `tau_good=0.7`, `tau_pr=0.8`; instance missed khi recall < 0.5.
- Đặc tính: GT resize 259×259 NEAREST; vùng vật = G co 3 px; nền = ngoài G nới 10 px; `mag_contrast = mag_obj/(mag_bg+0.5)`.
- Flow lưu float16 `[N,2,259,259]`, downsample `F.interpolate(mode="area")` rồi ×0.5; frame không có chiều đó = NaN.
- `flow_cnn_microbatch` mặc định 8 (0 = tắt).
- Đường dẫn dataset mặc định theo `eval.sh` (tương đối với `--data_root`, mặc định `data`).
- Không tự giảm `sequence_length` khi OOM; ghi `failed.csv`.

## Review Focus

1. Số file GT ≠ số frame ảnh (DAVIS ghép theo thứ tự như `eval.py`) → giữ hành vi tác giả nhưng ghi cột `gt_count_mismatch` và cảnh báo. Test ở Task 5.
2. Chunk dài 1 frame (đoạn cuối nhóm) → không có flow: độ lớn = 0 (như tác giả), vector NaN, đặc tính camera NaN, không crash. Test ở Task 2 và Task 4.
3. SAM2 không có prompt nào (mọi xác suất ≤ 0.8) → mask sau SAM2 rỗng, `sam_no_prompt=1`, video vẫn `done`. Test ở Task 3.
4. Video bị ngắt giữa chừng (có thư mục, không có `done.flag`) → chạy lại ghi đè sạch; video có `done.flag` → bỏ qua. Test ở Task 3.
5. GT rỗng hoặc vùng vật rỗng sau khi co (vật rất nhỏ) → đặc tính flow NaN, nhãn `empty_ok`/`false_alarm`, không chia cho 0. Test ở Task 4.

---

### Task 1: Khung thư mục, sửa import `eval.py`, kiểm tra dữ liệu

**Files:**
- Modify: `eval.py:5` (xoá dòng `    split_components,`)
- Create: `tools/analysis/__init__.py`, `tools/analysis/tests/__init__.py`, `tools/analysis/tests/conftest.py`
- Create: `tools/analysis/check_data.py`
- Test: `tools/analysis/tests/test_check_data.py`

**Interfaces:**
- Produces: `conftest.py` thêm gốc repo vào `sys.path`; fixture `tmp_dataset(tmp_path)` tạo DAVIS giả (2 video × 5 jpg 64×48 + 5 png nhãn) và trả về `(image_root, ann_root)`.
- Produces: `check_layout(image_root: str, ann_root: str, sequences: list[str] | None) -> pandas.DataFrame` cột `video, n_img, n_ann, ok`; CLI `python tools/analysis/check_data.py --datasets 2016 2016-M 2017-M segtrack fbms [--data_root data]` in bảng + tổng.

- [ ] **Step 1: Test** `test_check_layout_ok` (2 video đủ → mọi `ok` True), `test_check_layout_missing_ann` (xoá 1 png → `ok` False, `n_ann=4`).
- [ ] **Step 2:** Run `python -m pytest tools/analysis/tests/test_check_data.py -q` → FAIL (module chưa có).
- [ ] **Step 3:** Viết `check_layout` + CLI (dùng `protocols.DATASETS` khi Task 2 xong; ở task này CLI tạm khai báo đường dẫn theo bảng §4 spec, Task 2 thay bằng import). Xoá dòng import trong `eval.py`.
- [ ] **Step 4:** Run lại test → PASS. Run `python -c "import ast;ast.parse(open('eval.py').read())"` → không lỗi; `grep -c split_components eval.py` → `0`.
- [ ] **Step 5: Commit** `git add eval.py tools/analysis && git commit -m "chore(analysis): scaffold, data check, drop unused import in eval.py"`

### Task 2: `protocols.py` — chọn video, chọn frame, chia nhóm/chunk

**Files:**
- Create: `tools/analysis/protocols.py`
- Test: `tools/analysis/tests/test_protocols.py`

**Interfaces:**
- Produces:
  - `DATASETS: dict[str, DatasetSpec]`; `@dataclass DatasetSpec(key: str, image_rel: str, ann_rel: str, mode: Literal["grouped","fbms"], val_list: tuple[str,...] | None)`. Đường dẫn: `2016` → `DAVIS/JPEGImages/480p`, `DAVIS/Annotations/480p`; `2016-M`, `2017-M` → `DAVIS2017-M/DAVIS/JPEGImages/480p`, `.../Annotations/480p`; `segtrack` → `SegTrackv2/JPEGImages_jpg_standardized`, `SegTrackv2/GroundTruth`; `fbms` → `FBMS59_clean/JPEGImages`, `FBMS59_clean/Annotations`. `val_list`: 20 video DAVIS16 cho `2016` và `2016-M`, 30 video cho `2017-M` (chép nguyên văn từ `eval.py` `_find_davis_sequences_16` / `_find_davis_2017_sequences`), `None` cho segtrack/fbms.
  - `list_sequences(spec: DatasetSpec, image_root: str) -> list[str]` (sorted, chỉ thư mục, lọc theo `val_list`).
  - `natural_key(s: str) -> list` (chép `_natural_key`).
  - `select_frames(spec, frames_dir: str, gt_dir: str) -> list[str]`: `grouped` → `sorted` file đuôi `.jpg/.jpeg/.png`; `fbms` → stem GT sắp `natural_key`, tìm RGB theo thứ tự đuôi `.jpg,.jpeg,.png` rồi bản viết hoa, bỏ stem không có RGB.
  - `frame_number(name: str, fallback: int) -> int`: số nguyên cuối cùng trong stem, không có thì `fallback`.
  - `@dataclass FrameSlot(frame_file: str, orig_idx: int, frame_num: int, group: int, chunk: int, pos_in_chunk: int, chunk_len: int, is_chunk_start: bool, is_chunk_end: bool, gap_prev: float, gap_next: float)` — `gap_*` = chênh `frame_num` với frame kề trong cùng chunk, NaN nếu không có.
  - `plan_chunks(frame_files: list[str], mode: str, sequence_length: int, stride: int = 4) -> list[list[FrameSlot]]` theo thứ tự xử lý của `eval.py` (nhóm 0..stride-1, trong nhóm chunk tăng dần). `grouped`: nhóm `g` = `frame_files[g::stride]`, `orig_idx = j*stride + g`. `fbms`: 1 nhóm, `orig_idx = j`. `chunk` đánh số liên tục trong toàn video.
  - `sam_order(frame_files: list[str], mode: str) -> list[str]`: `grouped` → theo `orig_idx` (= thứ tự `frame_files`); `fbms` → `sorted(stem)` (plain sort, như `sorted(all_motion_masks_dict.keys())`).

- [ ] **Step 1: Test**
  - `test_grouped_50_frames`: 50 file `00000.jpg..00049.jpg`, L=32 → 4 chunk, `chunk_len` `[13,13,12,12]`; chunk 0 `orig_idx` = `[0,4,...,48]`; `gap_prev` của slot thứ 2 = 4; slot đầu `is_chunk_start` và `gap_prev` NaN.
  - `test_grouped_long_splits_chunks`: 200 file, L=32 → nhóm 0 có 50 frame → chunk dài `[32,18]`; tổng slot = 200, mọi `orig_idx` xuất hiện đúng 1 lần.
  - `test_single_frame_chunk`: 129 file, L=32 → tồn tại chunk `chunk_len == 1` với `is_chunk_start and is_chunk_end`, `gap_prev`, `gap_next` NaN.
  - `test_fbms_select_and_order` (tmp dir): GT `cars_1.png, cars_10.png, cars_2.png`; RGB có `cars_1.jpg, cars_2.JPG, cars_10.jpg` → `select_frames` = `["cars_1.jpg","cars_2.JPG","cars_10.jpg"]`; `plan_chunks(...,"fbms",32)` 1 chunk, `gap_prev` của `cars_10` = 8; `sam_order` = `["cars_1.jpg","cars_10.jpg","cars_2.JPG"]`.
  - `test_list_sequences_filters_val`: thư mục có `blackswan`, `bear` → `2016` trả `["blackswan"]`; `segtrack` trả cả hai.
- [ ] **Step 2:** Run `python -m pytest tools/analysis/tests/test_protocols.py -q` → FAIL.
- [ ] **Step 3:** Viết `protocols.py`; cập nhật `check_data.py` dùng `DATASETS`/`list_sequences`.
- [ ] **Step 4:** Run → PASS (cả test Task 1).
- [ ] **Step 5: Commit** `feat(analysis): protocols mirroring eval.py grouping`

### Task 3: `run_dump.py` — Runner

**Files:**
- Create: `tools/analysis/run_dump.py`, `tools/analysis/io_utils.py`
- Test: `tools/analysis/tests/test_run_dump.py`

**Interfaces:**
- Consumes: `protocols.*` (Task 2); từ tác giả: `motion_seg_inference.MotionSegmentationInference`, `compute_optical_flow_raft`, `denormalize_imagenet`, `refine_sam`, `DEFAULT_SAM2_*`; `sam2.build_sam.build_sam2_video_predictor`.
- Produces (`io_utils.py`, không phụ thuộc torch):
  - `save_prob_png(path, prob: np.ndarray[float]) ` ghi `(prob*255).astype(np.uint8)`; `load_gray_png(path) -> np.ndarray[uint8]`.
  - `seq_dir(run_dir, dataset, video) -> Path`; `is_done(path) -> bool`; `mark_done(path)`; `reset_seq_dir(path)` (xoá thư mục cũ nếu chưa done rồi tạo lại).
  - `append_failed(run_dir, dataset, video, err: BaseException)` → `failed.csv` cột `dataset,video,error,trace`.
  - Định dạng file theo spec §5.2; `frames.csv` cột: `frame_name, orig_idx, frame_num, group, chunk, pos_in_chunk, chunk_len, is_chunk_start, is_chunk_end, gap_prev, gap_next, flow_mag_mean, flow_mag_median, flow_mag_p95, has_fw, has_bw, depth_median, sam_no_prompt`.
- Produces (`run_dump.py`, cần torch):
  - `class FlowEncoderMicrobatch(torch.nn.Module)(inner: nn.Module, k: int)` — forward tách dim 0 theo lô `k`, `torch.cat`.
  - `compute_chunk_flows(video01: Tensor[N,3,H,W], raft_fn) -> tuple[fw, bw, mag]`: `fw[t]` = `raft_fn(f_t, f_{t+1})` (t < N-1, còn lại NaN), `bw[t]` = `raft_fn(f_t, f_{t-1})` (t > 0); `mag[t]` = max theo pixel của độ lớn các chiều có mặt (N=1 → zeros), y hệt `compute_optical_flow_magnitude_batch(bidirectional=True, fusion_method='max')`.
  - `predict_chunk(inf, frames_pil: list[Image], microbatch: int) -> dict(prob np.float32[N,518,518], poses np.float32[N,4,4], depth_median np.float32[N], fw, bw, mag)` — `inf.preprocess_video` → `denormalize_imagenet` → `compute_chunk_flows(..., compute_optical_flow_raft)` → `inf.model(video_tensor, mag[None])`; `depth_median` = median `local_points[...,2]`.
  - `sam_refine_safe(frames_pil, probs: np.ndarray[N,518,518], predictor, refine_fn=refine_sam) -> tuple[np.ndarray[N,518,518] float32, bool no_prompt]` — tái tạo `_refine_with_sam` (frame resize về 518 BILINEAR, `/255`, tensor `[N,3,H,W]`, buffer CUDA); nếu `(probs > 0.8).sum() == 0` hoặc `refine_fn` ném lỗi khi không có prompt → trả zeros, `True`.
  - `run_sequence(inf, predictor, spec, data_root, video, run_dir, sequence_length, microbatch, use_sam) -> None` (ghi file, `timing.json`, `done.flag`).
  - CLI theo spec §5.3: `--datasets`, `--gpu`, `--run_id`, `--videos`, `--sequence_length 32`, `--flow_cnn_microbatch 8`, `--no_sam`, `--data_root data`, `--model_path checkpoint/best_model.pth`, `--pi3_model_path checkpoint/model.safetensors`, `--sam2_checkpoint_path sam2-main/checkpoints/sam2.1_hiera_large.pt`. Ghi `config.json` (theo spec §5.2, gồm `sam2_has_C` = `importlib.util.find_spec("sam2._C") is not None`, sha256 8 ký tự đầu của checkpoint, `git rev-parse HEAD`). Bọc `inf.model.motion_aware_decoder.flow_encoder` bằng `FlowEncoderMicrobatch` khi `microbatch > 0`. Mỗi video trong `try/except` → `append_failed`, `torch.cuda.empty_cache()`, tiếp tục.

- [ ] **Step 1: Test (không cần GPU)**
  - `test_save_prob_truncates`: prob `0.999` → pixel `254`; `0.1` → `25`.
  - `test_resume_skips_done` / `test_reset_partial`: thư mục có `done.flag` → `is_done` True; thư mục có file lẻ không flag → `reset_seq_dir` xoá file cũ.
  - `test_microbatch_equivalence` (importorskip torch): `Conv2d` ngẫu nhiên seed 0, input `[10,1,32,32]` → output bọc `k=3` khớp `allclose(atol=1e-6)`.
  - `test_chunk_flows_matches_author` (importorskip torch, importorskip `motion_seg_inference`): `raft_fn` giả trả flow hằng `(t+1, -(t+1))` phụ thuộc thứ tự cặp; so `mag` với `compute_optical_flow_magnitude_batch` sau khi monkeypatch `motion_seg_inference.compute_optical_flow_raft` và `RAFT_MODEL` khác None → `allclose(atol=1e-5)`; với N=1 → `mag` toàn 0, `fw`, `bw` NaN.
  - `test_sam_no_prompt`: `probs` toàn 0.5, `refine_fn` giả ném `RuntimeError` → trả zeros và `True`; `refine_fn` giả ghi 1 vào buffer khi có prompt → trả mask đó và `False`.
- [ ] **Step 2:** Run `python -m pytest tools/analysis/tests/test_run_dump.py -q` → FAIL.
- [ ] **Step 3:** Viết `io_utils.py`, `run_dump.py`.
- [ ] **Step 4:** Run → PASS (các test cần `motion_seg_inference` sẽ skip trên máy không có torch/sam2; phải PASS trên server).
- [ ] **Step 5: Commit** `feat(analysis): runner dumping pre/post-SAM masks, flow, poses`

### Task 4: `taxonomy.py` + `features.py`

**Files:**
- Create: `tools/analysis/taxonomy.py`, `tools/analysis/features.py`
- Test: `tools/analysis/tests/test_taxonomy.py`, `tools/analysis/tests/test_features.py`

**Interfaces:**
- Produces (`taxonomy.py`):
  - `@dataclass Thresholds(tau_area=0.001, tau_miss=0.1, tau_good=0.7, tau_pr=0.8)`.
  - `mask_stats(pred: bool[H,W], gt: bool[H,W]) -> dict(area_pred, area_gt, inter, precision, recall)` (tỉ lệ theo diện tích ảnh; precision NaN nếu pred rỗng, recall NaN nếu gt rỗng).
  - `label_frame(stats: dict, j: float, th: Thresholds) -> str` theo bảng spec §7.3 (thứ tự `empty_ok, false_alarm, missed, wrong_location, good, under, over, mixed`).
  - `read_instance_ids(path) -> np.ndarray[int]` (mode `P` → chỉ số palette; `L` → giá trị xám; `RGB` → id theo màu duy nhất; 0 = nền).
  - `instance_misses(pred: bool[H,W], ids: int[H,W], recall_thr=0.5) -> tuple[int, int]` (`n_inst`, `n_inst_missed`).
  - `sam_group(j_pre, j_post, tau_good=0.7) -> str` ∈ `sam_rescued, sam_broke, both_good, both_bad`.
- Produces (`features.py`):
  - `fused_mag(fw: float[N,2,h,w], bw) -> float[N,h,w]` (max các chiều có, NaN-safe; không chiều nào → 0).
  - `frame_vector(fw, bw, t) -> float[2,h,w]` (`fw[t]` nếu không NaN, ngược lại `-bw[t]`, cả hai NaN → NaN).
  - `region_masks(gt_small: bool[h,w], erode_px=3, dilate_px=10) -> tuple[bool[h,w], bool[h,w]]` (cv2, kernel vuông `2r+1`).
  - `flow_contrast(vec: float[2,h,w], mag: float[h,w], obj, bg) -> dict(mag_obj, mag_bg, mag_contrast, vec_contrast)`; vùng rỗng → NaN.
  - `camera_motion(poses: float[N,4,4], depth_median: float[N], chunk: int[N]) -> tuple[float[N], float[N]]` (`cam_rot_deg`, `cam_trans_rel`; kề sau trong cùng chunk, frame cuối dùng kề trước; chunk 1 frame → NaN).

- [ ] **Step 1: Test**
  - Taxonomy, ảnh 100×100: gt rỗng & pred rỗng → `empty_ok`; gt rỗng & pred 20 px → `false_alarm`; gt ô 20×20, pred rỗng → `missed`; pred ô 20×20 ở góc khác → `wrong_location`; pred = gt → `good`; pred = nửa trên gt (J 0.5) → `under`; pred = gt nới 10 px → `over`; `j=0.7` đúng biên → `good`.
  - `test_instance_misses`: 2 vật id 1, 2; pred chỉ phủ vật 1 → `(2, 1)`.
  - `test_read_instance_ids_palette`: PNG mode `P` giá trị 0/1/2 → ids `{1,2}`.
  - `test_sam_group`: `(0.5,0.8)→sam_rescued`, `(0.8,0.5)→sam_broke`.
  - Features: nền flow `(0,0)`, vật `(+3,0)` → `mag_contrast ≈ 3/0.5 = 6`, `vec_contrast ≈ 3`; nền `(+3,0)`, vật `(-3,0)` → `mag_contrast ≈ 3/3.5 ≈ 0.857`, `vec_contrast ≈ 6` (trường hợp "chỉ khác hướng"); vật 2×2 px → vùng co rỗng → mọi giá trị NaN.
  - `test_frame_vector_last_frame`: `fw[t]` NaN, `bw[t]=(1,2)` → trả `(-1,-2)`.
  - `test_camera_motion`: 2 pose, quay 30° quanh y, dịch `(0.5,0,0)`, depth 2.0 → `rot≈30`, `trans_rel≈0.25` cho cả 2 frame; chunk 1 frame → NaN.
- [ ] **Step 2:** Run `python -m pytest tools/analysis/tests/test_taxonomy.py tools/analysis/tests/test_features.py -q` → FAIL.
- [ ] **Step 3:** Viết hai module.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat(analysis): error taxonomy and frame features`

### Task 5: `analyze.py` — metric, bảng frame, tổng hợp

**Files:**
- Create: `tools/analysis/analyze.py`
- Test: `tools/analysis/tests/test_analyze.py`

**Interfaces:**
- Consumes: `protocols`, `io_utils`, `taxonomy`, `features`; `core.eval.eval_mask.db_eval_iou`, `db_eval_boundary`.
- Produces:
  - `to_gt_size(mask01_or_uint8, shape_hw) -> float[H,W]` (đúng `_resize_mask`).
  - `analyze_sequence(run_dir, spec, data_root, video, th: Thresholds) -> pandas.DataFrame` — một dòng/frame có GT: định danh `dataset, video, frame_name` + cột `frames.csv` + cho `r ∈ {pre, post}`, `t ∈ {0.1, 0.5}`: `J_{r}_{t}, F_{r}_{t}`; ở 0.1: `P_{r}, R_{r}, area_pred_{r}, label_{r}, n_inst_missed_{r}`; chung: `area_gt, n_inst, sam_group, dJ, mag_obj, mag_bg, mag_contrast, vec_contrast, cam_rot_deg, cam_trans_rel, gt_empty, gt_count_mismatch`. Ghép GT: `grouped` theo thứ tự (`zip` như `eval.py`, `gt_count_mismatch = n_gt != n_frames`); `fbms` theo stem.
  - `add_dir_only(frame_df) -> frame_df` (cột `dir_only` theo tứ phân vị trong từng dataset).
  - `summarize(frame_df, paper: dict) -> pandas.DataFrame` cột `dataset, refine, protocol, J, F, JF, JR, paper_JF, diff`; `author` = trung bình mọi frame ở 0.1, JR = trung bình theo frame của JR từng video (như `_compute_dataset_metrics`); `davis_std` = trung bình theo video ở 0.5 rồi trung bình giữa video; giá trị ×100.
  - `PAPER = {"2016-M": 83.9, "2017-M": 81.1, "fbms": (72.5, 78.5), "2016": 84.7, "segtrack": 77.3}` + ghi chú "J cho 2017-M; J/F cho fbms; cần đối chiếu PDF".
  - `sensitivity(frame_df) -> DataFrame` (phân bố `label_post` khi `tau_good ∈ {0.5,0.7,0.8}`).
  - CLI `python tools/analysis/analyze.py --run_id ID [--datasets ...] [--no_media]` → ghi `analysis/frame_table.csv`, `metrics_summary.csv`, `sensitivity.csv`, gọi `report.build_report` trừ khi `--no_media`.
- Produces script `tools/analysis/verify_against_eval.py --eval_csv eval_orig/.../sequence_metrics.csv --frame_table results/ID/analysis/frame_table.csv --dataset 2016 [--tol 0.001]` → in bảng `video, J_eval, J_ours, dJ, F_eval, F_ours, dF, ok`, exit code 1 nếu có dòng không đạt (so `Mean_IoU`/`Mean_F_Score` với trung bình `J_post_0.1`/`F_post_0.1` từng video).

- [ ] **Step 1: Test** (fixture: run dir giả 1 video DAVIS 6 frame 64×48, prob/mask/flow/poses/frames.csv tự tạo; GT tương ứng)
  - `test_author_metric_matches_manual`: J/F từng frame bằng gọi trực tiếp `db_eval_iou`/`db_eval_boundary` trên mask qua `_resize_mask`-tương-đương.
  - `test_gt_count_mismatch_flag`: thêm 1 png GT thừa → cột `gt_count_mismatch` True, không crash.
  - `test_summary_two_protocols`: 2 video độ dài 2 và 8 frame, J khác nhau → `author` ≠ `davis_std`, tính tay khớp.
  - `test_verify_script`: CSV eval giả khớp → exit 0; lệch 0.01 → exit 1.
- [ ] **Step 2:** Run `python -m pytest tools/analysis/tests/test_analyze.py -q` → FAIL.
- [ ] **Step 3:** Viết `analyze.py`, `verify_against_eval.py`.
- [ ] **Step 4:** Run toàn bộ test → PASS.
- [ ] **Step 5: Commit** `feat(analysis): analyzer metrics, frame table, eval verification`

### Task 6: `report.py` — bảng video, biểu đồ, ảnh ghép, mp4, report.md

**Files:**
- Create: `tools/analysis/report.py`
- Test: `tools/analysis/tests/test_report.py`

**Interfaces:**
- Consumes: `frame_df`, `summary_df` (Task 5), `io_utils`.
- Produces:
  - `longest_bad_run(j: Sequence[float], thr=0.5) -> int`.
  - `video_table(frame_df) -> DataFrame` (spec §8, sắp `JF_post` tăng dần).
  - `flow_to_color(vec: float[2,h,w]) -> uint8[h,w,3]` (HSV: hue = hướng, value = độ lớn chuẩn hoá p99; NaN → đen).
  - `render_panel(rgb, gt, prob_pre, mask_post, mag, vec) -> uint8[H, 7W, 3]` (thứ tự ô theo spec §8; overlay TP xanh lá, FP đỏ, FN xanh dương).
  - `select_gallery(frame_df, k_label=20, k_worst_videos=10, k_per_video=3, k_sam_broke=20) -> DataFrame` (khử trùng lặp, cột `reason`).
  - `write_mp4(frames: list[np.ndarray], path, fps=6)` (ffmpeg H.264 `-preset veryfast -pix_fmt yuv420p`; không có ffmpeg → cảnh báo, bỏ qua).
  - 6 hàm vẽ `plot_*(frame_df, out_dir)` theo spec §8 (matplotlib, Agg).
  - `build_report(run_dir, frame_df, summary_df, datasets, data_root, media=True)` → `video_table.csv`, `plots/`, `gallery/`, `videos/`, `report.md` (gồm bảng paper, bảng khác biệt §9 spec lấy từ `config.json`, top video tệ, link biểu đồ).

- [ ] **Step 1: Test**
  - `test_longest_bad_run`: `[0.9,0.4,0.3,0.8,0.2]` → `2`.
  - `test_flow_to_color`: vector `(1,0)` và `(-1,0)` cùng độ lớn → khác hue, cùng value; NaN → `(0,0,0)`.
  - `test_render_panel_shape`: ảnh 48×64 → `(48, 448, 3)`.
  - `test_select_gallery_caps`: frame_df giả 300 dòng → mỗi nhãn ≤ 20, không trùng `(video, frame_name)`.
  - `test_build_report_smoke` (dùng fixture Task 5, `media=False` và `media=True` khi có ffmpeg): tạo `report.md`, `video_table.csv`, ≥ 1 file trong `plots/`.
- [ ] **Step 2:** Run `python -m pytest tools/analysis/tests/test_report.py -q` → FAIL.
- [ ] **Step 3:** Viết `report.py`.
- [ ] **Step 4:** Run toàn bộ test → PASS.
- [ ] **Step 5: Commit** `feat(analysis): reports, plots, galleries`

### Task 7: Hướng dẫn chạy + kiểm chứng trên server

**Files:**
- Create: `tools/analysis/README.md` (lệnh `nohup` cho từng bước §11 spec, cách copy `analysis/` về máy, cách đọc `report.md`)
- Create: `tools/analysis/make_subset.sh` (`make_subset.sh <dataset> <video1> <video2> [data_root]` tạo symlink vào `data/subset_<dataset>/` đúng cấu trúc)

- [ ] **Step 1:** Trên server: `python -m pytest tools/analysis/tests -q` → mọi test PASS, không skip test cần torch/`motion_seg_inference`.
- [ ] **Step 1b:** Thử build SAM2 `_C` (spec §9): `which nvcc`; nếu có → `cd ~/sam2 && SAM2_BUILD_CUDA=1 pip install -e . --no-build-isolation`; kiểm tra `python -c "import sam2._C"`. Ghi kết quả (được / không, lý do) vào `tools/analysis/README.md`. Không build được thì tiếp tục, `config.json` tự ghi `sam2_has_C=false`.
- [ ] **Step 2:** `bash tools/analysis/make_subset.sh 2016 blackswan parkour`; chạy `eval.py` gốc trên subset 2 lần (đo dao động) → `eval_verify/a`, `eval_verify/b`.
- [ ] **Step 3:** `python tools/analysis/run_dump.py --datasets 2016 --videos blackswan parkour --gpu 1 --run_id verify_2016` rồi `python tools/analysis/analyze.py --run_id verify_2016 --no_media`.
- [ ] **Step 4:** `python tools/analysis/verify_against_eval.py --eval_csv eval_verify/a/sequence_metrics.csv --frame_table results/verify_2016/analysis/frame_table.csv --dataset 2016` → exit 0 (hoặc lệch ≤ mức dao động a↔b; ghi vào README).
- [ ] **Step 5: Commit** `docs(analysis): run guide and verification results`

Sau Task 7: chạy toàn bộ (ngoài kế hoạch code) theo `tools/analysis/README.md`.
