import os
import time
import numpy as np
import torch

from affective_head import AffectiveHead, TEMPORAL_DIM
from gaze_head import GazeHead

try:
    from tflite_engine import TFLiteAffectiveHead, TFLiteGazeHead
except ImportError:
    pass

def check_variance(fp32_output, tflite_output, name="Model"):
    """Calculates Mean Absolute Error and relative % error."""
    fp32_arr = np.array(fp32_output, dtype=np.float32).flatten()
    tf_arr = np.array(tflite_output, dtype=np.float32).flatten()
    
    mae = np.mean(np.abs(fp32_arr - tf_arr))
    
    # Calculate % variance relative to the range of the FP32 values
    mean_val = np.mean(np.abs(fp32_arr))
    if mean_val == 0.0:
        pct_variance = mae * 100.0 # avoid div by zero
    else:
        pct_variance = (mae / mean_val) * 100.0
        
    print(f"[{name}] Output Variance (MAE): {mae:.5f} | Relative Error: {pct_variance:.2f}%")
    if pct_variance < 5.0:
        print(f"[{name}] ✅ INT8 Quantization variance is WITHIN the 5% margin constraints.")
    else:
        print(f"[{name}] ❌ WARNING: INT8 Quantization exceeds 5% variance baseline.")

def benchmark_inference(model_call, inputs, runs=500):
    """Measures latency of a loaded model over multiple runs."""
    # Warmup
    for _ in range(20):
        model_call(*inputs)
        
    t0 = time.perf_counter()
    for _ in range(runs):
        model_call(*inputs)
    t1 = time.perf_counter()
    
    return ((t1 - t0) * 1000.0) / runs

def main():
    print("=== Validation & Benchmarking: PyTorch FP32 vs TFLite INT8 ===")
    
    # 1. Check if TFLite models actually exist yet
    if not os.path.exists("models/affective_cnn_int8.tflite") or not os.path.exists("models/gaze_hybrid_int8.tflite"):
        print("\n[ERROR] .tflite models not found in models/ directory!")
        print("Please generate the TFLite files from Colab and place them in models/ before benchmarking.")
        return

    # 2. Load PyTorch FP32 Baselines
    print("\nLoading PyTorch FP32 Baselines...")
    pt_device = torch.device("cpu") # Force PyTorch to CPU for fair CPU vs CPU comparison
    
    pt_aff = AffectiveHead()
    pt_gaze = GazeHead()
    
    if pt_aff.feature_extractor: pt_aff.feature_extractor.to(pt_device).eval()
    pt_aff.tcn.to(pt_device).eval()
    pt_gaze.model.to(pt_device).eval()

    # 3. Load TFLite INT8 Engine
    print("\nLoading TFLite INT8 Inference Wrappers...")
    try:
        tf_aff = TFLiteAffectiveHead()
        tf_gaze = TFLiteGazeHead()
    except Exception as e:
        print(f"TFLite Engine failed to load: {e}")
        return

    # 4. Generate identical dummy features matching shapes
    print("\n--- Generating Identical Validation Features ---")
    
    # Bbox/Frame mimic for CNN
    face_input = np.random.randn(48, 48).astype(np.float32) * 255.0
    face_input = np.clip(face_input, 0, 255).astype(np.uint8)
    # PyTorch wants it as (1, 1, 48, 48) tensor normalized
    pt_face_tensor = torch.from_numpy(face_input).float().unsqueeze(0).unsqueeze(0) / 255.0
    pt_face_tensor = (pt_face_tensor - 0.5) / 0.5
    
    # Sequence mapping for TCN (1, 15, 524) and Gaze (1, 15, 10)
    pt_temp_seq = torch.randn(1, 15, pt_aff.embed_dim + TEMPORAL_DIM)
    pt_gaze_seq = torch.randn(1, 15, 10)
    pt_gaze_pose = torch.randn(1, 3)
    
    np_temp_seq = pt_temp_seq.numpy()
    np_gaze_seq = pt_gaze_seq.numpy()
    np_gaze_pose = pt_gaze_pose.numpy()

    # 5. Extract Single Shot Predictions for Output Variance
    print("\n--- Model Variance Analysis ---")
    
    # CNN Variance
    with torch.no_grad():
        pt_cnn_out = pt_aff.feature_extractor(pt_face_tensor).numpy() if pt_aff.feature_extractor else np.zeros((1,512))
    tf_cnn_out = tf_aff.cnn_engine.infer(pt_face_tensor.numpy())[0] if tf_aff.cnn_engine else np.zeros((1,512))
    check_variance(pt_cnn_out, tf_cnn_out, "Affective CNN")
    
    # TCN Variance
    with torch.no_grad():
        pt_tcn_out = pt_aff.tcn(pt_temp_seq).numpy()
    tf_tcn_out = tf_aff.tcn_engine.infer(np_temp_seq)[0] if tf_aff.tcn_engine else np.zeros((1,1))
    check_variance(pt_tcn_out, tf_tcn_out, "Affective TCN")

    # Gaze Variance
    with torch.no_grad(): # PyTorch Gaze reqs transpose logic in forward
        pt_gaze_out = pt_gaze.model(pt_gaze_seq, pt_gaze_pose).numpy()
    tf_gaze_out = tf_gaze.engine.infer(np_gaze_seq, np_gaze_pose)[0] if tf_gaze.engine else np.zeros((1,2))
    check_variance(pt_gaze_out, tf_gaze_out, "Gaze Hybrid Regression")


    # 6. Execute Latency Benchmarks
    print("\n--- Model Latency Benchmarks (CPU) ---")
    
    # Defines the callable logic
    def pt_cnn_call(): pt_aff.feature_extractor(pt_face_tensor)
    def tf_cnn_call(): tf_aff.cnn_engine.infer(pt_face_tensor.numpy())
    
    def pt_tcn_call(): pt_aff.tcn(pt_temp_seq)
    def tf_tcn_call(): tf_aff.tcn_engine.infer(np_temp_seq)
    
    def pt_gaze_call(): pt_gaze.model(pt_gaze_seq, pt_gaze_pose)
    def tf_gaze_call(): tf_gaze.engine.infer(np_gaze_seq, np_gaze_pose)

    pt_cnn_lat = benchmark_inference(pt_cnn_call, []) if pt_aff.feature_extractor else 0
    tf_cnn_lat = benchmark_inference(tf_cnn_call, []) if tf_aff.cnn_engine else 0
    print(f"Affective CNN | FP32 PyTorch: {pt_cnn_lat:.3f} ms --> INT8 TFLite: {tf_cnn_lat:.3f} ms")

    pt_tcn_lat = benchmark_inference(pt_tcn_call, [])
    tf_tcn_lat = benchmark_inference(tf_tcn_call, []) if tf_aff.tcn_engine else 0
    print(f"Affective TCN | FP32 PyTorch: {pt_tcn_lat:.3f} ms --> INT8 TFLite: {tf_tcn_lat:.3f} ms")

    pt_gaze_lat = benchmark_inference(pt_gaze_call, [])
    tf_gaze_lat = benchmark_inference(tf_gaze_call, []) if tf_gaze.engine else 0
    print(f"Gaze Hybrid   | FP32 PyTorch: {pt_gaze_lat:.3f} ms --> INT8 TFLite: {tf_gaze_lat:.3f} ms")


if __name__ == "__main__":
    main()
