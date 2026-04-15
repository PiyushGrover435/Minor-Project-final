import os
import time
import torch
import torch.ao.quantization
from affective_head import AffectiveHead, TEMPORAL_DIM
from gaze_head import GazeHead

# Ensure we evaluate on CPU since INT8 PyTorch quantized engines are CPU-optimized
device = torch.device("cpu")

def benchmark_latency(model, inputs, name="Model", runs=300):
    model.eval()
    # Warmup
    for _ in range(20):
        model(*inputs)
    
    t0 = time.perf_counter()
    for _ in range(runs):
        model(*inputs)
    t1 = time.perf_counter()
    
    mean_ms = ((t1 - t0) * 1000.0) / runs
    return mean_ms

def quantize_and_benchmark():
    os.makedirs("models", exist_ok=True)
    
    print("\nLoading FP32 Models...")
    # Initialize
    aff_head = AffectiveHead()
    gaze_head = GazeHead()
    
    # Move all components to CPU for quantization
    cnn_fp32 = aff_head.feature_extractor.to(device).eval() if aff_head.feature_extractor else None
    tcn_fp32 = aff_head.tcn.to(device).eval()
    gaze_fp32 = gaze_head.model.to(device).eval()

    # Dummy inputs for benchmarking
    dummy_face = torch.randn(1, 1, 48, 48).to(device)
    dummy_aff_seq = torch.randn(1, 15, aff_head.embed_dim + TEMPORAL_DIM).to(device)
    dummy_gaze_seq = torch.randn(1, 15, 10).to(device)
    dummy_gaze_pose = torch.randn(1, 3).to(device)

    print("\n--- Applying INT8 Dynamic Quantization ---")
    
    # 1. Quantize TCN
    # Dynamic Quantization targets nn.Linear layers (which make up the bulk of TCN parameters)
    tcn_int8 = torch.ao.quantization.quantize_dynamic(
        tcn_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(tcn_int8.state_dict(), "models/affective_tcn_int8.pth")
    print("Saved -> models/affective_tcn_int8.pth")

    # 2. Quantize Gaze Hybrid
    gaze_int8 = torch.ao.quantization.quantize_dynamic(
        gaze_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(gaze_int8.state_dict(), "models/gaze_hybrid_int8.pth")
    print("Saved -> models/gaze_hybrid_int8.pth")

    # Benchmarking
    print("\n--- Latency Benchmark (CPU Only) ---")
    
    lat_tcn_fp32 = benchmark_latency(tcn_fp32, (dummy_aff_seq,))
    lat_tcn_int8 = benchmark_latency(tcn_int8, (dummy_aff_seq,))
    print(f"Affective TCN | FP32: {lat_tcn_fp32:.3f} ms | INT8: {lat_tcn_int8:.3f} ms")
    
    lat_gaze_fp32 = benchmark_latency(gaze_fp32, (dummy_gaze_seq, dummy_gaze_pose))
    lat_gaze_int8 = benchmark_latency(gaze_int8, (dummy_gaze_seq, dummy_gaze_pose))
    print(f"Gaze Hybrid   | FP32: {lat_gaze_fp32:.3f} ms | INT8: {lat_gaze_int8:.3f} ms")
    
    print("\nNote: PyTorch Dynamic Quantization primarily accelerates linear layers. Convolutions must be statically quantized, which is skipped here for simplicity.")

if __name__ == "__main__":
    quantize_and_benchmark()
