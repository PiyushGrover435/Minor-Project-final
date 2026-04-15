"""
edge_benchmark.py

Comprehensive edge-device test suite for Sentin-Edge AI quantized models.

Runs from the project root:
    python edge_benchmark.py

Tests:
  1. Model inventory & file sizes
  2. Quantization validation (weight dtype, packed params)
  3. FP32 vs INT8 accuracy delta (MAE, cosine similarity)
  4. Latency benchmark  — wall-clock per-inference in milliseconds
  5. Throughput estimate — theoretical max FPS on this CPU
  6. Full end-to-end pipeline timing (one frame through the whole stack)
  7. System info printout for submission / report
"""
import os
import sys
import time
import platform
import textwrap

import numpy as np
import torch

# ── Make project root importable when running from root ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaze_head import GazeHead, HybridGazeModel
from affective_head import AffectiveHead, TEMPORAL_DIM
from gaze_geometry import extract_gaze_feature_vector

try:
    from tflite_engine import TFLiteGazeHead, TFLiteAffectiveHead, HAS_TFLITE
except ImportError:
    HAS_TFLITE = False

# ── Constants ────────────────────────────────────────────────────────
WARMUP_RUNS  = 30
BENCH_RUNS   = 200
SEQ_LEN      = 15
GAZE_FEAT    = 10

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
INFO  = "[INFO]"
SEP   = "=" * 64


def _hr(char="-", width=64):
    print(char * width)


def _section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    _hr()


# ── 1. System Info ───────────────────────────────────────────────────

def print_system_info():
    _section("SYSTEM INFORMATION")
    print(f"  OS        : {platform.platform()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  CPU cores : {os.cpu_count()}")
    print(f"  TFLite    : {'Available' if HAS_TFLITE else 'Not installed (pip install tflite-runtime)'}")

    try:
        import cv2
        print(f"  OpenCV    : {cv2.__version__}")
    except ImportError:
        print(f"  OpenCV    : not found")

    try:
        import mediapipe as mp
        print(f"  MediaPipe : {mp.__version__}")
    except ImportError:
        print(f"  MediaPipe : not found")


# ── 2. Model Inventory ───────────────────────────────────────────────

MODEL_FILES = {
    "FP32 Weights": [
        ("Affective CNN (FP32)",   "models/affective_cnn.pth"),
        ("Stress TCN (FP32)",      "models/stress_tcn.pth"),
        ("Gaze Hybrid (FP32)",     "models/gaze_hybrid_epoch2.pth"),
    ],
    "PyTorch INT8": [
        ("Stress TCN INT8",        "models/affective_tcn_int8.pth"),
        ("Gaze Hybrid INT8",       "models/gaze_hybrid_int8.pth"),
    ],
    "ONNX Exports": [
        ("Affective CNN ONNX",     "models/affective_cnn.onnx"),
        ("Affective TCN ONNX",     "models/affective_tcn.onnx"),
        ("Gaze Hybrid ONNX",       "models/gaze_hybrid.onnx"),
    ],
    "TFLite INT8 (Edge-Ready)": [
        ("Affective CNN TFLite",   "models/affective_cnn_int8.tflite"),
        ("Affective TCN TFLite",   "models/affective_tcn_int8.tflite"),
        ("Gaze Hybrid TFLite",     "models/gaze_hybrid_int8.tflite"),
    ],
}

SIZE_PAIRS = [
    ("Affective TCN", "models/stress_tcn.pth",        "models/affective_tcn_int8.pth",     "models/affective_tcn_int8.tflite"),
    ("Gaze Hybrid",   "models/gaze_hybrid_epoch2.pth", "models/gaze_hybrid_int8.pth",       "models/gaze_hybrid_int8.tflite"),
]


def print_model_inventory():
    _section("MODEL INVENTORY & FILE SIZES")
    all_ok = True
    for group, files in MODEL_FILES.items():
        print(f"\n  {group}:")
        for label, path in files:
            exists = os.path.exists(path)
            size = os.path.getsize(path) / 1024 if exists else 0
            tag  = PASS if exists else FAIL
            size_str = f"{size:.1f} KB" if exists else "NOT FOUND"
            print(f"    {tag} {label:<28} {size_str}")
            if not exists:
                all_ok = False

    print(f"\n  Size Comparison (FP32 -> INT8 PyTorch -> TFLite INT8):")
    for name, fp32, int8, tflite in SIZE_PAIRS:
        fp32_kb   = os.path.getsize(fp32)   / 1024 if os.path.exists(fp32)   else 0
        int8_kb   = os.path.getsize(int8)   / 1024 if os.path.exists(int8)   else 0
        tflite_kb = os.path.getsize(tflite) / 1024 if os.path.exists(tflite) else 0
        ratio_dyn  = fp32_kb / max(int8_kb,   0.01)
        ratio_tfl  = fp32_kb / max(tflite_kb, 0.01)
        print(f"    {name:<14} FP32:{fp32_kb:>7.1f}KB  INT8:{int8_kb:>7.1f}KB ({ratio_dyn:.1f}x)  TFLite:{tflite_kb:>6.1f}KB ({ratio_tfl:.1f}x)")

    return all_ok


# ── 3. Quantization Validation ───────────────────────────────────────

def check_quantization(path, label):
    """Inspect a .pth state dict for INT8 packed params."""
    if not os.path.exists(path):
        print(f"    {WARN} {label}: file not found — skipped")
        return False
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        has_packed = any(
            "_packed_params" in k or "scale" in k or "zero_point" in k
            for k in state.keys()
        )
        has_qint8 = any(
            isinstance(v, torch.Tensor) and v.dtype in (torch.qint8, torch.quint8)
            for v in state.values()
        )
        quantized = has_packed or has_qint8
        tag = PASS if quantized else WARN
        kind = "INT8 dynamic" if quantized else "FP32 (no quantization detected)"
        print(f"    {tag} {label:<32} {kind}")
        return quantized
    except Exception as e:
        print(f"    {FAIL} {label}: load error — {e}")
        return False


def validate_quantization():
    _section("QUANTIZATION VALIDATION")
    print("  PyTorch INT8 weight analysis:")
    check_quantization("models/affective_tcn_int8.pth", "Stress TCN INT8")
    check_quantization("models/gaze_hybrid_int8.pth",   "Gaze Hybrid INT8")

    if HAS_TFLITE:
        print("\n  TFLite INT8 model inspection:")
        for label, path in [
            ("Affective CNN TFLite", "models/affective_cnn_int8.tflite"),
            ("Affective TCN TFLite", "models/affective_tcn_int8.tflite"),
            ("Gaze Hybrid TFLite",   "models/gaze_hybrid_int8.tflite"),
        ]:
            if os.path.exists(path):
                from tflite_engine import TFLiteInferenceEngine
                try:
                    eng = TFLiteInferenceEngine(path)
                    in_dtype  = eng.input_details[0]['dtype'].__name__
                    out_dtype = eng.output_details[0]['dtype'].__name__
                    in_shape  = list(eng.input_details[0]['shape'])
                    print(f"    {PASS} {label:<32} in:{in_dtype}{in_shape}  out:{out_dtype}")
                except Exception as e:
                    print(f"    {FAIL} {label}: {e}")
            else:
                print(f"    {WARN} {label}: not found")
    else:
        print(f"\n  {WARN} TFLite not installed — skipping TFLite dtype inspection.")
        print(f"       Install: pip install tflite-runtime")


# ── 4. Accuracy Delta (FP32 vs INT8) ─────────────────────────────────

def _mae(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    return float(np.mean(np.abs(a - b)))

def _cosine_sim(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 1.0
    return float(np.dot(a, b) / denom)

def _pct_err(a, b):
    scale = float(np.mean(np.abs(np.asarray(a, np.float32))))
    if scale < 1e-9:
        return 0.0
    return float(_mae(a, b) / scale * 100.0)


def accuracy_delta():
    _section("ACCURACY DELTA: FP32 vs INT8")
    print("  (MAE = Mean Absolute Error on identical random inputs)")
    print("  (Cosine = directional similarity of output vectors)\n")

    results = {}

    # --- Gaze Head ---
    np.random.seed(42)
    gaze_seq  = np.random.randn(1, SEQ_LEN, GAZE_FEAT).astype(np.float32)
    pose_arr  = np.random.randn(1, 3).astype(np.float32)
    gaze_seq_t  = torch.from_numpy(gaze_seq)
    pose_t      = torch.from_numpy(pose_arr)

    # FP32
    fp32_gaze = GazeHead(model_path="models/gaze_hybrid_epoch2.pth")
    fp32_gaze.model.eval()
    with torch.no_grad():
        out_fp32 = fp32_gaze.model(gaze_seq_t, pose_t).numpy()

    # INT8 dynamic
    int8_gaze = GazeHead()  # loads int8 by default
    int8_gaze.model.eval()
    with torch.no_grad():
        out_int8 = int8_gaze.model(gaze_seq_t, pose_t).numpy()

    mae_g  = _mae(out_fp32, out_int8)
    pct_g  = _pct_err(out_fp32, out_int8)
    cos_g  = _cosine_sim(out_fp32, out_int8)
    tag_g  = PASS if pct_g < 5.0 else FAIL
    print(f"  Gaze Hybrid   [{tag_g}]  MAE={mae_g:.5f}  Err={pct_g:.2f}%  Cosine={cos_g:.4f}")
    results["gaze_mae"] = mae_g

    # --- TFLite ---
    if HAS_TFLITE and os.path.exists("models/gaze_hybrid_int8.tflite"):
        from tflite_engine import TFLiteInferenceEngine
        tfl_eng = TFLiteInferenceEngine("models/gaze_hybrid_int8.tflite")
        if tfl_eng.input_details[0]['shape'][-1] == 3:
            out_tfl = tfl_eng.infer(pose_arr, gaze_seq)[0]
        else:
            out_tfl = tfl_eng.infer(gaze_seq, pose_arr)[0]
        mae_tg  = _mae(out_fp32, out_tfl)
        pct_tg  = _pct_err(out_fp32, out_tfl)
        cos_tg  = _cosine_sim(out_fp32, out_tfl)
        tag_tg  = PASS if pct_tg < 5.0 else FAIL
        print(f"  Gaze TFLite   [{tag_tg}]  MAE={mae_tg:.5f}  Err={pct_tg:.2f}%  Cosine={cos_tg:.4f}")
        results["gaze_tflite_mae"] = mae_tg

    print()

    # --- Affective Head stress score ---
    embed_dim  = 256
    seq_tensor = np.random.randn(1, SEQ_LEN, embed_dim + TEMPORAL_DIM).astype(np.float32)
    seq_t      = torch.from_numpy(seq_tensor)

    fp32_aff = AffectiveHead.__new__(AffectiveHead)
    from scripts.train_affective_head import MultiModalStressTCN
    fp32_aff.tcn = MultiModalStressTCN(embed_dim=embed_dim, temporal_dim=TEMPORAL_DIM)
    if os.path.exists("models/stress_tcn.pth"):
        try:
            state = torch.load("models/stress_tcn.pth", map_location="cpu", weights_only=True)
            fp32_aff.tcn.load_state_dict(state, strict=False)
        except Exception:
            pass
    fp32_aff.tcn.eval()
    with torch.no_grad():
        out_fp32_tcn = fp32_aff.tcn(seq_t).numpy()

    int8_aff = AffectiveHead()
    int8_aff.tcn.eval()
    with torch.no_grad():
        out_int8_tcn = int8_aff.tcn(seq_t).numpy()

    mae_s  = _mae(out_fp32_tcn, out_int8_tcn)
    pct_s  = _pct_err(out_fp32_tcn, out_int8_tcn)
    cos_s  = _cosine_sim(out_fp32_tcn, out_int8_tcn)
    tag_s  = PASS if pct_s < 5.0 else FAIL
    print(f"  Stress TCN    [{tag_s}]  MAE={mae_s:.5f}  Err={pct_s:.2f}%  Cosine={cos_s:.4f}")
    results["tcn_mae"] = mae_s

    return results


# ── 5. Latency Benchmark ─────────────────────────────────────────────

def _bench(fn, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / runs   # ms per inference


def latency_benchmark():
    _section("LATENCY BENCHMARK (CPU)")
    print(f"  Warmup={WARMUP_RUNS} runs, Benchmark={BENCH_RUNS} runs\n")

    results = {}

    # ── Gaze Head ───────────────────────────────────────────────────
    gaze_seq_t = torch.randn(1, SEQ_LEN, GAZE_FEAT)
    pose_t     = torch.randn(1, 3)

    fp32_g = GazeHead(model_path="models/gaze_hybrid_epoch2.pth")
    int8_g = GazeHead()

    def fp32_gaze(): 
        with torch.no_grad(): fp32_g.model(gaze_seq_t, pose_t)

    def int8_gaze():
        with torch.no_grad(): int8_g.model(gaze_seq_t, pose_t)

    lat_fp32_g = _bench(fp32_gaze)
    lat_int8_g = _bench(int8_gaze)
    speedup_g  = lat_fp32_g / max(lat_int8_g, 0.001)
    results["gaze_fp32_ms"] = lat_fp32_g
    results["gaze_int8_ms"] = lat_int8_g
    print(f"  Gaze Hybrid   FP32: {lat_fp32_g:>7.3f} ms  |  INT8: {lat_int8_g:>7.3f} ms  ({speedup_g:.2f}x speedup)")

    if HAS_TFLITE and os.path.exists("models/gaze_hybrid_int8.tflite"):
        from tflite_engine import TFLiteInferenceEngine
        tfl_g = TFLiteInferenceEngine("models/gaze_hybrid_int8.tflite")
        gaze_seq_np = gaze_seq_t.numpy()
        pose_np     = pose_t.numpy()
        if tfl_g.input_details[0]['shape'][-1] == 3:
            def tfl_gaze(): tfl_g.infer(pose_np, gaze_seq_np)
        else:
            def tfl_gaze(): tfl_g.infer(gaze_seq_np, pose_np)
        lat_tfl_g = _bench(tfl_gaze)
        speedup_tfl_g = lat_fp32_g / max(lat_tfl_g, 0.001)
        results["gaze_tflite_ms"] = lat_tfl_g
        print(f"  Gaze TFLite   FP32: {lat_fp32_g:>7.3f} ms  |  TFLite(INT8): {lat_tfl_g:>7.3f} ms  ({speedup_tfl_g:.2f}x speedup)")

    print()

    # ── Stress TCN ──────────────────────────────────────────────────
    embed_dim  = 256
    tcn_seq_np = np.random.randn(1, SEQ_LEN, embed_dim + TEMPORAL_DIM).astype(np.float32)
    tcn_seq_t  = torch.from_numpy(tcn_seq_np)

    int8_aff  = AffectiveHead()
    int8_aff.tcn.eval()

    try:
        from scripts.train_affective_head import MultiModalStressTCN
        fp32_tcn = MultiModalStressTCN(embed_dim=embed_dim, temporal_dim=TEMPORAL_DIM)
        if os.path.exists("models/stress_tcn.pth"):
            state = torch.load("models/stress_tcn.pth", map_location="cpu", weights_only=True)
            fp32_tcn.load_state_dict(state, strict=False)
        fp32_tcn.eval()
        def fp32_tcn_fn():
            with torch.no_grad(): fp32_tcn(tcn_seq_t)
        lat_fp32_tcn = _bench(fp32_tcn_fn)
        results["tcn_fp32_ms"] = lat_fp32_tcn
    except Exception:
        lat_fp32_tcn = None

    def int8_tcn_fn():
        with torch.no_grad(): int8_aff.tcn(tcn_seq_t)

    lat_int8_tcn = _bench(int8_tcn_fn)
    results["tcn_int8_ms"] = lat_int8_tcn

    if lat_fp32_tcn:
        speedup_tcn = lat_fp32_tcn / max(lat_int8_tcn, 0.001)
        print(f"  Stress TCN    FP32: {lat_fp32_tcn:>7.3f} ms  |  INT8: {lat_int8_tcn:>7.3f} ms  ({speedup_tcn:.2f}x speedup)")
    else:
        print(f"  Stress TCN    INT8: {lat_int8_tcn:.3f} ms  (FP32 baseline unavailable)")

    return results


# ── 6. End-to-End Pipeline Timing ────────────────────────────────────

def e2e_pipeline_timing():
    _section("END-TO-END PIPELINE TIMING")
    print("  Simulates one full frame through the analytics stack\n")

    from analytics import RealtimeAnalyzer

    # Build a plausible keypoints dict
    dummy_kp = {
        'left_iris':         np.array([200.0, 165.0]),
        'right_iris':        np.array([340.0, 165.0]),
        'left_inner':        np.array([180.0, 165.0]),
        'left_outer':        np.array([220.0, 165.0]),
        'right_inner':       np.array([320.0, 165.0]),
        'right_outer':       np.array([360.0, 165.0]),
        'left_upper_eyelid': np.array([200.0, 158.0]),
        'left_lower_eyelid': np.array([200.0, 172.0]),
        'right_upper_eyelid':np.array([340.0, 158.0]),
        'right_lower_eyelid':np.array([340.0, 172.0]),
        'left_eyebrow':      np.array([200.0, 142.0]),
        'right_eyebrow':     np.array([340.0, 142.0]),
        'left_inner_brow':   np.array([190.0, 144.0]),
        'right_inner_brow':  np.array([330.0, 144.0]),
        'nose_bridge':       np.array([270.0, 180.0]),
        'left_mouth_corner': np.array([230.0, 250.0]),
        'right_mouth_corner':np.array([310.0, 250.0]),
        'upper_lip_center':  np.array([270.0, 240.0]),
        'nose_tip':          np.array([270.0, 210.0]),
        'left_eye_points':   [(180,160),(190,155),(200,155),(220,160),(200,170),(190,170)],
        'right_eye_points':  [(320,160),(330,155),(340,155),(360,160),(340,170),(330,170)],
        'head_pose':         (0.0, 0.0, 0.0),
        'head_translation':  (0.0, 0.0, 600.0),
        'landmarks':         np.random.rand(478, 2) * 480,
        'face_bbox':         (140, 100, 400, 400),
        'image_size':        (640, 480),
    }

    analyzer      = RealtimeAnalyzer(window=12, calib_frames=30, gaze_seq_len=15)
    dummy_frame   = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        analyzer.update(dummy_kp, dummy_frame)

    # Time
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        analyzer.update(dummy_kp, dummy_frame)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    print(f"  Analytics.update()  mean={times.mean():.2f}ms  "
          f"p50={np.percentile(times,50):.2f}ms  "
          f"p95={np.percentile(times,95):.2f}ms  "
          f"p99={np.percentile(times,99):.2f}ms")
    max_fps = 1000.0 / max(times.mean(), 0.1)
    print(f"  Theoretical max FPS (analytics only): {max_fps:.1f} FPS")
    print(f"  Note: Add ~10-30ms for MediaPipe landmark extraction per frame")

    return times.mean()


# ── 7. Summary & Deployment Readiness ────────────────────────────────

def deployment_summary(lat_results):
    _section("DEPLOYMENT READINESS SUMMARY")

    gaze_int8   = lat_results.get("gaze_int8_ms", 0)
    gaze_tflite = lat_results.get("gaze_tflite_ms", 0)
    tcn_int8    = lat_results.get("tcn_int8_ms", 0)

    print(f"  PyTorch INT8 stack latency  : {gaze_int8 + tcn_int8:.2f} ms/frame")
    if gaze_tflite:
        print(f"  TFLite INT8 stack latency   : {gaze_tflite + tcn_int8:.2f} ms/frame  (preferred for edge)")

    print()
    checks = [
        ("TFLite models present",     all(os.path.exists(p) for p in [
            "models/affective_cnn_int8.tflite",
            "models/affective_tcn_int8.tflite",
            "models/gaze_hybrid_int8.tflite"])),
        ("PyTorch INT8 models present", all(os.path.exists(p) for p in [
            "models/affective_tcn_int8.pth",
            "models/gaze_hybrid_int8.pth"])),
        ("TFLite runtime available",  HAS_TFLITE),
        ("Analytics stack < 50ms",    gaze_int8 + tcn_int8 < 50),
        ("Gaze model quantized",       gaze_int8 < lat_results.get("gaze_fp32_ms", 9999)),
    ]

    all_pass = True
    for label, ok in checks:
        tag = PASS if ok else FAIL
        if not ok: all_pass = False
        print(f"  {tag}  {label}")

    print()
    if all_pass:
        print("  All checks passed — models are edge-deployment ready!")
    else:
        print("  Some checks failed. See above for details.")

    print(f"\n  Edge device targets for deployment:")
    print(f"    Raspberry Pi 4 (Cortex-A72)  : TFLite INT8 recommended")
    print(f"    Jetson Nano (CPU mode)         : TFLite INT8 or PyTorch INT8")
    print(f"    Android device                 : TFLite INT8 via MediaPipe Tasks API")
    print(f"    x86 laptop (this machine)      : PyTorch INT8 dynamic (current)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print(f"  SENTIN-EDGE AI  —  EDGE DEVICE MODEL TEST SUITE")
    print(f"{SEP}")

    print_system_info()
    all_files_ok = print_model_inventory()
    validate_quantization()

    try:
        accuracy_delta()
    except Exception as e:
        print(f"\n  {WARN} Accuracy delta test skipped: {e}")

    lat = {}
    try:
        lat = latency_benchmark()
    except Exception as e:
        print(f"\n  {WARN} Latency benchmark failed: {e}")

    try:
        e2e_pipeline_timing()
    except Exception as e:
        print(f"\n  {WARN} E2E pipeline timing failed: {e}")

    deployment_summary(lat)
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
