# Chuẩn bị benchmark GeoMotion

Ngày kiểm tra: 05/11/2026. Ưu tiên: GeoMotion gốc trước, Full GeoMotion sau.
Chưa tải dataset/weights, chưa chạy inference. Metadata checkpoint được đọc từ API Hugging Face; RAFT/SAM2 và DAVIS được kiểm tra bằng HTTP HEAD.

## Dataset cần chuẩn bị

Paper gốc đánh giá trên DAVIS2016, DAVIS2016-Moving, DAVIS2017, SegTrack-v2 và FBMS-59. Dataset huấn luyện không cần tải để chạy evaluation bằng checkpoint có sẵn.

| Dataset | Nguồn và chuẩn bị | Điểm cần xác minh |
|---|---|---|
| DAVIS2016 | [Trang chính thức](https://davischallenge.org/davis2016/code.html); lấy RGB và GT ở 480p | `eval.py` chọn 20 video validation. Phải ghi lại đúng danh sách video và nguồn mask. |
| DAVIS2016-Moving | Dùng RGB của DAVIS; cần chốt danh sách video và GT theo protocol | `eval.py` chọn 20 video; `eval_adapter_raft.py` chọn 16 video. Chưa nên coi hai cấu hình là cùng benchmark. |
| DAVIS2017 | [TrainVal 480p](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip), 832,766,765 bytes | `eval.py` chọn 30 video validation. Evaluator hiện tính mask foreground nhị phân, không phải chấm instance VOS chính thức. |
| DAVIS2017-motion | RGB DAVIS2017; [GT được OCLR liên kết](https://drive.google.com/drive/folders/16zqUO1WuAp8Am4Giitc9hn2rMao6x_MA?usp=sharing) | Không đồng nhất với GT DAVIS2017 gốc. Link Drive tồn tại, chưa kiểm tra file bên trong. Bổ sung sau khi chốt benchmark gốc. |
| FBMS-59 | [Trang Freiburg](https://lmbweb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html); chuyển sang JPEGImages/Annotations theo OCLR/motiongrouping | GT thưa. Script GeoMotion lấy mọi thư mục trong image_root; cần giới hạn rõ test split. Helper `FBMS_clean_2.py` liệt kê 30 video test. |
| SegTrack-v2 | [Trang tác giả](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html), 14 video; [ZIP](https://web.engr.oregonstate.edu/~lif/SegTrack2/SegTrackv2.zip) | Chuẩn hóa ảnh/tên frame và hợp mask nhiều object thành foreground theo protocol. Python trên máy local báo lỗi chứng chỉ khi HEAD link ZIP; chưa xác minh tải thành công. |

OCLR hướng dẫn dùng DAVIS2016 trực tiếp, DAVIS2017-motion có annotation riêng; FBMS/SegTrack cần preprocessing theo motiongrouping. Không thay GT giữa các protocol mà vẫn giữ cùng nhãn kết quả.

Nguồn chuẩn bị:
- [OCLR README](https://github.com/Jyxarthur/OCLR_model#dataset-preparation)
- [FBMS_clean.py](https://github.com/charigyang/motiongrouping/blob/main/tools/FBMS_clean.py)
- [FBMS_clean_2.py](https://github.com/charigyang/motiongrouping/blob/main/tools/FBMS_clean_2.py)
- [SegTrack_clean.py](https://github.com/charigyang/motiongrouping/blob/main/tools/SegTrack_clean.py)
- [Easi3R DAVIS preparation](https://github.com/Inception3D/Easi3R/blob/main/data/download_prepare_davis.py)

Các helper là mã tham khảo có đường dẫn hard-code, không chạy nguyên trạng. Cần đối chiếu từng basename RGB–GT, không chỉ zip hai danh sách đã sort.

## Checkpoint

GB trong bảng là đơn vị thập phân. Dung lượng file không phải yêu cầu VRAM.

| Thành phần | File | Dung lượng | Nguồn |
|---|---|---:|---|
| GeoMotion gốc | `best_model.pth` | 4,533,837,989 bytes | [Hugging Face](https://huggingface.co/xingyang1/GeoMotion/blob/main/best_model.pth) |
| π³ backbone | `model.safetensors` | 3,834,909,248 bytes | [Hugging Face](https://huggingface.co/yyfz233/Pi3/blob/main/model.safetensors) |
| RAFT Large | `raft_large_C_T_SKHT_V2-ff5fadd5.pth` | 21,106,607 bytes | [PyTorch](https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth) |
| SAM2.1 Large | `sam2.1_hiera_large.pt` | 898,083,611 bytes | [Meta](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |
| Full GeoMotion, giai đoạn sau | `Full_GeoMotion.pth` | 4,534,491,765 bytes | [Hugging Face](https://huggingface.co/xingyang1/GeoMotion/blob/main/Full_GeoMotion.pth) |

Bốn file cho cấu hình gốc cộng khoảng 9.29 GB. README hướng dẫn chuẩn bị π³; mã inference khởi tạo π³ nếu có path rồi load `checkpoint['model_state_dict']` bằng strict mặc định. Chưa tải checkpoint để kiểm tra trực tiếp nội dung, vì vậy chưa kết luận bỏ được file π³. RAFT có thể tự tải weights nếu thiếu file local; cần pin đúng weights khi tái lập. SAM2 là bước refinement; phải ghi rõ bật/tắt trong run manifest. Code hiện import SAM2 và tạo predictor ngay cả ở một số đường chạy tắt refinement.

Metadata để pin phiên bản:

- GeoMotion Hugging Face revision: `a94586c9d6dbb06d8b1746b6efb353073f44659a`
- `best_model.pth` LFS SHA-256: `f7c6f33542f3366833b1737380daad08c7e0388786d238cc88db87d6f2f04a42`
- `Full_GeoMotion.pth` LFS SHA-256: `c20c263046bd60f12bc63475306ec4d732ca6dd43a6de138a937d4204bd46ad5`
- Pi3 revision: `ae722e7039287d0c8fde9f11f197f804f44b510c`
- `model.safetensors` LFS SHA-256: `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`

## Protocol và mốc đối chiếu

Paper Table 1, đơn vị %:

| Dataset | J | F | J&F |
|---|---:|---:|---:|
| DAVIS2016-M | 83.5 | 84.3 | 83.9 |
| DAVIS2017 | 81.1 | — | — |
| FBMS-59 | 72.5 | 78.5 | — |
| DAVIS2016 | 84.5 | 85.0 | 84.7 |
| SegTrackV2 | 77.3 | — | — |

[Paper gốc](https://arxiv.org/html/2602.21810v1). Đây là mốc công bố, chưa phải kết quả đã tái lập.

Các vấn đề rút ra từ code local commit `306aad2`:

1. `eval.py` và `eval_adapter_raft.py` có danh sách moving subset khác nhau. [Issue #5](https://github.com/zjutcvg/GeoMotion/issues/5) hỏi về protocol DAVIS; API trả về chưa có comment tại thời điểm kiểm tra.
2. Với DAVIS/SegTrack, mặc định chia frame theo stride 4, rồi chia chunk theo `sequence_length` (CLI mặc định 32). Đổi chunk để giảm VRAM sẽ đổi ngữ cảnh thời gian, cần coi là cấu hình thí nghiệm riêng.
3. FBMS chỉ đưa RGB khớp GT vào inference trong đường evaluation hiện tại. Cần giữ lại lựa chọn này khi so với repo và ghi rõ nó khác inference trên toàn video.
4. Đánh giá nhị phân dùng `pred > 0.1`. Overlay dùng ngưỡng 127/255; vì vậy ảnh overlay hiện có có thể không thể hiện chính xác mask đang được chấm điểm.
5. Điểm dataset trong code gốc gộp tất cả frame, nên video dài có trọng số lớn hơn. Nên lưu thêm trung bình theo video và chỉ rõ cách tổng hợp khi đối chiếu.
6. GT không đọc được có thể bị bỏ qua; Full evaluator cắt ngắn khi số prediction/GT lệch. Cần phát hiện trước và ghi lỗi, tránh âm thầm thay tập được chấm.

## Dữ liệu cần lưu khi benchmark

- Run manifest: commit, checksum weights, environment/GPU, dataset/GT source, exact sequence list, frame list, chunk length/stride, threshold, SAM2 config.
- Mỗi frame: basename RGB/GT/prediction; J, F, J&F; TP/FP/FN; trạng thái inference và đánh giá.
- Probability mask thô trước SAM2; mask sau SAM2; binary mask đúng ngưỡng chấm; ảnh RGB/GT/prediction/error map.
- JSONL lỗi kèm traceback, video, chunk và các frame liên quan; tách frame lỗi chạy với frame dự đoán hợp lệ nhưng điểm thấp.
- Coverage: số video/frame dự kiến, chạy thành công, thất bại, thiếu GT. Không gán frame lỗi điểm 0 rồi gọi đó là prediction, cũng không bỏ lỗi khỏi báo cáo coverage.
- Tổng hợp theo frame và theo video; danh sách frame/video kém nhất. Giữ artifacts của video chạy dở.

Thứ tự chuẩn bị đề xuất: kiểm tra server → DAVIS2016 + checkpoint gốc → thử một video đầy đủ với cấu hình benchmark → toàn bộ DAVIS2016 → moving subset đã chốt → DAVIS2017 → SegTrack/FBMS → Full GeoMotion. Hai GPU 16 GB không tự cộng thành một GPU 32 GB; kiểm tra peak VRAM thực tế trước khi chọn lịch chạy.
