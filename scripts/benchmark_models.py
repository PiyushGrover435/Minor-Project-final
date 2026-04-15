import os
import time
import torch
import numpy as np
from affective_head import AffectiveHead, TEMPORAL_DIM
from gaze_head import GazeHead

# Disable threading for consistent benchmarking
torch.set_num_threads(1)

def profile_model(model, args_generator, name="Model", runs=500, warmup=50):
    print(f"\n--- Profiling {name} ---")
    
    # Run warmup
    for _ in range(warmup):
        model(*args_generator())
        
    times = []
    
    # Run benchmark
    for _ in range(runs):
        args = args_generator()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        t0 = time.perf_counter()
        model(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        
        times.append((t1 - t0) * 1000.0) # convert to ms
        
    times = np.array(times)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0
    
    print(f"Device:   {next(model.parameters()).device}")
    print(f"Runs:     {runs} (warmup={warmup})")
    print(f"Mean Lat: {mean_ms:.3f} ms ± {std_ms:.3f} ms")
    print(f"Min Lat:  {min_ms:.3f} ms")
    print(f"Max Lat:  {max_ms:.3f} ms")
    print(f"Est. FPS: {fps:.1f}")
    
    return mean_ms, fps

def main():
    print("Initializing FP32 Models...")
    
    # 1. Affective CNN (FER)
    affective_head = AffectiveHead()

    def get_cnn_args():
        dummy_face = torch.randn(1, 1, 48, 48).to(affective_head.device)
        return (dummy_face,)
        
    def get_tcn_args():
        # Seq_len=15, feature_dim = embed_dim + TEMPORAL_DIM
        dummy_seq = torch.randn(1, 15, affective_head.embed_dim + TEMPORAL_DIM).to(affective_head.device)
        return (dummy_seq,)
        
    # 2. Gaze Hybrid Model
    gaze_head = GazeHead()
    gaze_head.model.eval()
    
    def get_gaze_args():
        # Seq_len=15, input_dim=10 (gaze geometry features)
        dummy_seq = torch.randn(1, 15, 10).to(gaze_head.device)
        dummy_pose = torch.randn(1, 3).to(gaze_head.device)
        return (dummy_seq, dummy_pose)
        
    print("\nStarting Benchmarks (FP32)...")
    if affective_head.feature_extractor is not None:
        profile_model(affective_head.feature_extractor, get_cnn_args, name="Affective CNN (Feature Extractor)", runs=1000)
    profile_model(affective_head.tcn, get_tcn_args, name="Affective TCN (Stress Regression)", runs=1000)
    profile_model(gaze_head.model, get_gaze_args, name="Gaze Hybrid Model", runs=1000)

if __name__ == "__main__":
    main()
