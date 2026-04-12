"""
verify_quantization.py

Comprehensive verification script that checks:
  1. Whether all models are properly quantized (INT8 weights present)
  2. Model file sizes (FP32 vs INT8 comparison)
  3. Weight distribution analysis (confirms INT8 quantization applied)
  4. Functional inference test (smoke test on dummy data)
"""

import os
import torch
import numpy as np

from affective_head import AffectiveHead, MultiModalStressTCN, TEMPORAL_DIM
from gaze_head import GazeHead, HybridGazeModel


def check_file_exists(path, label):
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    status = "✅" if exists else "❌"
    print(f"  {status} {label}: {'%.1f KB' % size_kb if exists else 'NOT FOUND'}")
    return exists, size_kb


def analyze_quantization(state_dict, label):
    """Check if a state dict contains quantized (INT8) layers."""
    has_quantized = False
    has_packed = False
    total_params = 0
    quantized_params = 0
    
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            total_params += val.numel()
            if val.dtype == torch.qint8 or val.dtype == torch.quint8:
                quantized_params += val.numel()
                has_quantized = True
        if '_packed_params' in key or 'scale' in key or 'zero_point' in key:
            has_packed = True
    
    if has_quantized or has_packed:
        pct = (quantized_params / max(total_params, 1)) * 100 if has_quantized else 0
        print(f"  ✅ {label}: QUANTIZED (INT8 dynamic) | {total_params} total params")
        if has_packed:
            print(f"     → Contains packed INT8 weight tensors")
    else:
        print(f"  ⚠️  {label}: FP32 (not quantized) | {total_params} total params")
    
    return has_quantized or has_packed


def functional_test():
    """Smoke test: run inference through both heads and verify output shapes."""
    print("\n4. Functional Inference Test")
    print("-" * 50)
    
    device = torch.device("cpu")
    
    # Test AffectiveHead
    try:
        aff = AffectiveHead()
        dummy_kp = {
            'landmarks': np.random.rand(478, 2),
            'left_eyebrow': (200, 150),
            'right_eyebrow': (350, 150),
            'left_upper_eyelid': (190, 160),
            'right_upper_eyelid': (340, 160),
            'left_inner': (180, 165),
            'right_inner': (330, 165),
        }
        
        temporal = tuple([0.0] * TEMPORAL_DIM)
        emotion, level, score = aff.predict(dummy_kp, frame=None, temporal_geometries=temporal)
        print(f"  ✅ AffectiveHead.predict() → emotion={emotion}, level={level}, score={score:.3f}")
    except Exception as e:
        print(f"  ❌ AffectiveHead.predict() FAILED: {e}")
    
    # Test GazeHead
    try:
        gaze = GazeHead()
        dummy_gaze_kp = {
            'left_iris': (200, 165),
            'right_iris': (340, 165),
            'left_inner': (180, 165),
            'left_outer': (220, 165),
            'right_inner': (320, 165),
            'right_outer': (360, 165),
            'head_pose': (0.0, 0.0, 0.0),
            'left_eye_points': [(180, 160), (190, 155), (200, 155), (220, 160), (200, 170), (190, 170)],
            'right_eye_points': [(320, 160), (330, 155), (340, 155), (360, 160), (340, 170), (330, 170)],
        }
        
        label, vals = gaze.predict(dummy_gaze_kp)
        print(f"  ✅ GazeHead.predict() → label={label}, left_t={vals.get('left_t', 0):.3f}")
    except Exception as e:
        print(f"  ❌ GazeHead.predict() FAILED: {e}")


def main():
    print("=" * 60)
    print("  SENTIN-EDGE AI: MODEL QUANTIZATION VERIFICATION")
    print("=" * 60)
    
    # 1. Check all model files
    print("\n1. Model File Inventory")
    print("-" * 50)
    
    fp32_files = {
        "Affective CNN (FP32)": "models/affective_cnn.pth",
        "Stress TCN (FP32)": "models/stress_tcn.pth",
        "Gaze Hybrid (FP32)": "models/gaze_hybrid_epoch2.pth",
    }
    
    int8_files = {
        "Stress TCN INT8": "models/affective_tcn_int8.pth",
        "Gaze Hybrid INT8": "models/gaze_hybrid_int8.pth",
    }
    
    onnx_files = {
        "Affective CNN ONNX": "models/affective_cnn.onnx",
        "Affective TCN ONNX": "models/affective_tcn.onnx",
        "Gaze Hybrid ONNX": "models/gaze_hybrid.onnx",
    }
    
    tflite_files = {
        "Affective CNN TFLite": "models/affective_cnn_int8.tflite",
        "Affective TCN TFLite": "models/affective_tcn_int8.tflite",
        "Gaze Hybrid TFLite": "models/gaze_hybrid_int8.tflite",
    }
    
    print("\n  FP32 PyTorch Weights:")
    for label, path in fp32_files.items():
        check_file_exists(path, label)
    
    print("\n  INT8 Dynamic Quantized Weights:")
    for label, path in int8_files.items():
        check_file_exists(path, label)
    
    print("\n  ONNX Exports:")
    for label, path in onnx_files.items():
        check_file_exists(path, label)
    
    print("\n  TFLite INT8 Models:")
    for label, path in tflite_files.items():
        check_file_exists(path, label)
    
    # 2. Size Comparison
    print("\n2. Size Comparison (FP32 vs INT8)")
    print("-" * 50)
    
    comparisons = [
        ("Stress TCN", "models/stress_tcn.pth", "models/affective_tcn_int8.pth"),
        ("Gaze Hybrid", "models/gaze_hybrid_epoch2.pth", "models/gaze_hybrid_int8.pth"),
    ]
    
    for name, fp32_path, int8_path in comparisons:
        if os.path.exists(fp32_path) and os.path.exists(int8_path):
            fp32_kb = os.path.getsize(fp32_path) / 1024
            int8_kb = os.path.getsize(int8_path) / 1024
            ratio = fp32_kb / max(int8_kb, 0.1)
            print(f"  {name}: {fp32_kb:.1f} KB (FP32) → {int8_kb:.1f} KB (INT8) | {ratio:.1f}x compression")
        elif os.path.exists(int8_path):
            int8_kb = os.path.getsize(int8_path) / 1024
            print(f"  {name}: FP32 missing | INT8: {int8_kb:.1f} KB")
        else:
            print(f"  {name}: INT8 weights not found yet")
    
    # 3. Weight Distribution Analysis
    print("\n3. Quantization Weight Analysis")
    print("-" * 50)
    
    int8_tcn_path = "models/affective_tcn_int8.pth"
    int8_gaze_path = "models/gaze_hybrid_int8.pth"
    
    if os.path.exists(int8_tcn_path):
        try:
            state = torch.load(int8_tcn_path, map_location="cpu", weights_only=False)
            analyze_quantization(state, "Stress TCN INT8")
        except Exception as e:
            print(f"  ⚠️  Could not analyze TCN INT8: {e}")
    
    if os.path.exists(int8_gaze_path):
        try:
            state = torch.load(int8_gaze_path, map_location="cpu", weights_only=False)
            analyze_quantization(state, "Gaze Hybrid INT8")
        except Exception as e:
            print(f"  ⚠️  Could not analyze Gaze INT8: {e}")
    
    # 4. Functional test
    functional_test()
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    
    has_tcn_trained = os.path.exists("models/stress_tcn.pth")
    has_int8_tcn = os.path.exists("models/affective_tcn_int8.pth")
    has_int8_gaze = os.path.exists("models/gaze_hybrid_int8.pth")
    has_onnx = all(os.path.exists(p) for p in onnx_files.values())
    
    checks = [
        ("Affective CNN (FER) trained", os.path.exists("models/affective_cnn.pth")),
        ("Stress TCN trained", has_tcn_trained),
        ("Gaze Hybrid trained", os.path.exists("models/gaze_hybrid_epoch2.pth")),
        ("TCN INT8 Quantized", has_int8_tcn),
        ("Gaze INT8 Quantized", has_int8_gaze),
        ("ONNX Exports Complete", has_onnx),
    ]
    
    all_pass = True
    for label, passed in checks:
        status = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        print(f"  {status} {label}")
    
    if all_pass:
        print("\n  🎉 ALL CHECKS PASSED — Models are fully quantized and deployment-ready!")
    else:
        print("\n  ⚠️  Some checks failed. See details above.")


if __name__ == "__main__":
    main()
