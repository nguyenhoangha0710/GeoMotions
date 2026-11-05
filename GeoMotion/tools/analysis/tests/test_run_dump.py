import numpy as np
import pytest
from PIL import Image

from tools.analysis import io_utils
from tools.analysis.run_dump import sam_refine_safe


def test_save_prob_truncates(tmp_path):
    p = tmp_path / "a.png"
    io_utils.save_prob_png(p, np.array([[0.999, 0.1], [0.0, 1.0]], np.float32))
    assert io_utils.load_gray_png(p).tolist() == [[254, 25], [0, 255]]


def test_resume_skips_done(tmp_path):
    d = io_utils.seq_dir(tmp_path, "2016", "blackswan")
    d.mkdir(parents=True)
    assert not io_utils.is_done(d)
    io_utils.mark_done(d)
    assert io_utils.is_done(d)


def test_reset_partial(tmp_path):
    d = io_utils.seq_dir(tmp_path, "2016", "camel")
    (d / "prob_pre_sam").mkdir(parents=True)
    (d / "prob_pre_sam" / "00000.png").write_bytes(b"old")
    io_utils.reset_seq_dir(d)
    assert d.is_dir() and not any(d.iterdir())


def test_append_failed(tmp_path):
    try:
        raise RuntimeError("CUDA out of memory")
    except RuntimeError as e:
        io_utils.append_failed(tmp_path, "2016", "dog", e)
    io_utils.append_failed(tmp_path, "2016", "cows", ValueError("x"))
    import pandas as pd
    df = pd.read_csv(tmp_path / "failed.csv")
    assert list(df["video"]) == ["dog", "cows"]
    assert "out of memory" in df["error"][0]


def _frames(n):
    return [Image.new("RGB", (16, 12)) for _ in range(n)]


def test_sam_no_prompt_skips_refine():
    called = []
    probs = np.full((4, 8, 8), 0.5, np.float32)
    out, no_prompt = sam_refine_safe(_frames(4), probs, predictor=None,
                                     refine_fn=lambda **kw: called.append(1))
    assert no_prompt and not called
    assert out.shape == (4, 8, 8) and out.max() == 0


def test_sam_one_parity_without_prompt():
    probs = np.zeros((4, 8, 8), np.float32)
    probs[1, 2:4, 2:4] = 0.9  # only an odd frame has a prompt -> even-prompt pass would crash in eval.py
    out, no_prompt = sam_refine_safe(_frames(4), probs, predictor=None, refine_fn=lambda **kw: None)
    assert no_prompt and out.max() == 0


def test_sam_refine_error_is_caught():
    torch = pytest.importorskip("torch")
    probs = np.full((2, 8, 8), 0.9, np.float32)

    def boom(**kw):
        raise RuntimeError("No input points or masks are provided")
    out, no_prompt = sam_refine_safe(_frames(2), probs, predictor=None, refine_fn=boom, device="cpu")
    assert no_prompt and out.max() == 0


def test_sam_refine_passes_through():
    torch = pytest.importorskip("torch")
    probs = np.full((2, 8, 8), 0.9, np.float32)
    seen = {}

    def fake(frame_tensors, mask_list, p_masks_sam, offset, predictor):
        seen["shape"] = tuple(frame_tensors.shape)
        seen["max"] = float(frame_tensors.max())
        p_masks_sam[:] = 1.0
    out, no_prompt = sam_refine_safe(_frames(2), probs, predictor=None, refine_fn=fake, device="cpu")
    assert not no_prompt and out.min() == 1.0
    assert seen["shape"] == (2, 3, 8, 8) and seen["max"] <= 1.0


def test_microbatch_equivalence():
    torch = pytest.importorskip("torch")
    from tools.analysis.run_dump import make_flow_encoder_microbatch
    torch.manual_seed(0)
    inner = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(inplace=True))
    x = torch.randn(10, 1, 32, 32)
    wrapped = make_flow_encoder_microbatch(inner, 3)
    assert torch.allclose(inner(x), wrapped(x), atol=1e-6)


def _const_raft(torch):
    def raft(f1, f2):
        # flow depends on the (ordered) pair through the frames' mean value
        a, b = float(f1.mean()), float(f2.mean())
        out = torch.zeros(2, f1.shape[1], f1.shape[2])
        out[0] = (b - a) * 10
        out[1] = 1.0
        return out
    return raft


def test_chunk_flows_single_frame():
    torch = pytest.importorskip("torch")
    from tools.analysis.run_dump import compute_chunk_flows
    fw, bw, mag = compute_chunk_flows(torch.rand(1, 3, 8, 8), _const_raft(torch))
    assert torch.isnan(fw).all() and torch.isnan(bw).all()
    assert mag.shape == (1, 8, 8) and float(mag.abs().max()) == 0.0


def test_chunk_flows_fused_max():
    torch = pytest.importorskip("torch")
    from tools.analysis.run_dump import compute_chunk_flows
    video = torch.stack([torch.full((3, 4, 4), v) for v in (0.0, 0.1, 0.5)])
    fw, bw, mag = compute_chunk_flows(video, _const_raft(torch))
    assert torch.isnan(fw[2]).all() and torch.isnan(bw[0]).all()
    # frame 1: backward to frame0 -> u=-1, forward to frame2 -> u=4 ; magnitudes sqrt(2), sqrt(17)
    assert torch.allclose(mag[1], torch.full((4, 4), 17 ** 0.5), atol=1e-5)
    assert torch.allclose(mag[0], torch.full((4, 4), 2 ** 0.5), atol=1e-5)
    assert torch.allclose(bw[1][0], torch.full((4, 4), -1.0), atol=1e-5)


def test_chunk_flows_matches_author(monkeypatch):
    torch = pytest.importorskip("torch")
    msi = pytest.importorskip("motion_seg_inference")
    from tools.analysis.run_dump import compute_chunk_flows
    raft = _const_raft(torch)
    monkeypatch.setattr(msi, "compute_optical_flow_raft", raft)
    monkeypatch.setattr(msi, "RAFT_MODEL", object(), raising=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frames01 = torch.rand(5, 3, 16, 16)
    video_norm = ((frames01 - mean) / std)[None]
    ref = msi.compute_optical_flow_magnitude_batch(video_norm, "cpu", bidirectional=True, fusion_method="max")
    _, _, mag = compute_chunk_flows(msi.denormalize_imagenet(video_norm)[0], raft)
    assert torch.allclose(ref[0], mag, atol=1e-5)


def test_sam_other_errors_propagate():
    """A CUDA OOM (or any non-'no prompt' error) must not be turned into an empty, finished video."""
    torch = pytest.importorskip("torch")
    probs = np.full((2, 8, 8), 0.9, np.float32)

    def oom(**kw):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.10 GiB")
    with pytest.raises(RuntimeError, match="out of memory"):
        sam_refine_safe(_frames(2), probs, predictor=None, refine_fn=oom, device="cpu")


def test_sam_no_prompt_message_matcher():
    from tools.analysis.run_dump import _is_no_prompt_error
    assert _is_no_prompt_error(RuntimeError("No input points or masks are provided for any object"))
    assert not _is_no_prompt_error(RuntimeError("CUDA out of memory. Tried to allocate 4.10 GiB"))


def test_chunked_flow_forward_matches_author():
    """Patched MotionAwareDecoder.forward (encode + downsample per micro-batch) == author's forward."""
    torch = pytest.importorskip("torch")
    mod = pytest.importorskip("pi3.models.pi3_conf_flow_cam_lowfeature_wo_mean")
    from tools.analysis.run_dump import install_chunked_flow_forward
    torch.manual_seed(0)
    C, B, N, H, W, start = 8, 1, 7, 28, 28, 1
    hw = start + (H // 14) * (W // 14)
    ref = mod.MotionAwareDecoder(hidden_dim=2 * C, patch_size=14).eval()
    args = dict(low_hidden=torch.randn(B * N, hw, 2 * C), hidden=torch.randn(B * N, hw, 2 * C),
                optical_flow=torch.rand(B, N, H, W) * 5, camera_hidden=torch.randn(B * N, hw, 512),
                patch_start_idx=start, H=H, W=W, B=B, N=N)
    with torch.no_grad():
        expected = ref(**args)
        install_chunked_flow_forward(ref, 3)
        got = ref(**args)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


class _FakePredictor:
    def __init__(self):
        self.calls = []

    def init_state(self, video_path, offload_video_to_cpu=False, offload_state_to_cpu=False,
                   async_loading_frames=False):
        self.calls.append((video_path, offload_video_to_cpu, offload_state_to_cpu))
        return {"video_path": video_path}


def test_sam_low_memory_predictor_offloads():
    from tools.analysis.run_dump import make_sam_low_memory
    p = make_sam_low_memory(_FakePredictor())
    p.init_state("/tmp/frames")
    assert p.calls == [("/tmp/frames", True, True)]


class _FakeModel:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def test_model_offloaded_moves_and_restores():
    from tools.analysis.run_dump import model_offloaded
    m = _FakeModel()
    with model_offloaded(m, "cuda"):
        assert m.moves == ["cpu"]
    assert m.moves == ["cpu", "cuda"]


def test_model_offloaded_restores_on_error():
    from tools.analysis.run_dump import model_offloaded
    m = _FakeModel()
    with pytest.raises(ValueError):
        with model_offloaded(m, "cuda"):
            raise ValueError("sam failed")
    assert m.moves == ["cpu", "cuda"]
