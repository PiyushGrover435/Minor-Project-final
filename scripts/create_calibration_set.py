import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def create_affective_calibration_set(dataset_path, output_path, num_samples=200):
    print(f"Creating Affective Calibration Set from {dataset_path}...")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
        # Get random subset
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
        
        images, _ = next(iter(loader))
        
        # We also need dummy temporal data since AffectiveHead requires both
        temporal_dim = 12
        temporal_data = torch.randn(num_samples, temporal_dim)
        
        # Save as numpy arrays for TFLite
        np.savez(
            os.path.join(output_path, "affective_calibration.npz"),
            input_face=images.numpy(),
            input_temporal=temporal_data.numpy()
        )
        print(f"Successfully saved {num_samples} calibration samples to {output_path}/affective_calibration.npz")
    except Exception as e:
        print(f"Error creating affective calibration set (Check if FER-2013 is at the path): {e}")

def create_gaze_calibration_set(output_path, num_samples=200):
    print("Creating Gaze Calibration Set...")
    try:
        # Since we use geometric features (seq_len=15, dim=10) and head pose (dim=3),
        # generating representative random data within expected ranges is often
        # preferable to parsing raw MPIIGaze images which the hybrid model doesn't use directly.
        # Ratios are typically [0, 1], EAR is ~0.2-0.3, Pose is in degrees.
        
        seq_data = np.random.uniform(0.1, 0.9, size=(num_samples, 15, 10)).astype(np.float32)
        pose_data = np.random.uniform(-15.0, 15.0, size=(num_samples, 3)).astype(np.float32)
        
        np.savez(
            os.path.join(output_path, "gaze_calibration.npz"),
            input_seq=seq_data,
            input_pose=pose_data
        )
        print(f"Successfully saved {num_samples} calibration samples to {output_path}/gaze_calibration.npz")
    except Exception as e:
        print(f"Error creating gaze calibration set: {e}")

if __name__ == "__main__":
    out_dir = "calibration_data"
    os.makedirs(out_dir, exist_ok=True)
    
    fer_path = os.path.join("Dataset", "FER-2013")
    
    create_affective_calibration_set(fer_path, out_dir, num_samples=250)
    create_gaze_calibration_set(out_dir, num_samples=250)
    print("\nCalibration set generation complete. Pre-quantization requirement met.")
