import os
import glob
import numpy as np
import scipy.io as sio

# ── Head thresholds (based on implementation plan) ────────────
THETA_THRESH_OFF = 15.0   # vertical absolute angle > 15 deg -> Off-screen
PHI_THRESH_OFF   = 25.0   # horizontal absolute angle > 25 deg -> Off-screen
PHI_THRESH_CEN   = 5.0    # horizontal absolute angle <= 5 deg -> Center
# otherwise Left/Right depending on sign of phi.

def vec_to_angles(gaze_vecs):
    """
    Convert (x,y,z) gaze vector to (theta, phi) in degrees.
    Formula from MPIIGaze ReadMe:
      theta = asin(-y)
      phi = atan2(-x, -z)
    """
    x = gaze_vecs[:, 0]
    y = gaze_vecs[:, 1]
    z = gaze_vecs[:, 2]
    
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    
    return np.degrees(theta), np.degrees(phi)

def get_label(theta, phi):
    """Map angles to 4-class label."""
    if abs(theta) > THETA_THRESH_OFF or abs(phi) > PHI_THRESH_OFF:
        return 'Off-screen'
    if abs(phi) <= PHI_THRESH_CEN:
        return 'Center'
    elif phi < -PHI_THRESH_CEN:
        return 'Left'
    else:
        return 'Right'

def load_mpiigaze_data(base_path, max_samples=None):
    """
    Loads MPIIGaze normalized .mat files.
    Returns:
        X: np.ndarray of shape (N, 36*60)
        y: np.ndarray of shape (N,) containing strings
    """
    X_list = []
    y_list = []
    
    # Iterate over participants
    participants = sorted(glob.glob(os.path.join(base_path, 'Data', 'Normalized', 'p*')))
    
    for p_dir in participants:
        print(f"Loading participant: {os.path.basename(p_dir)}")
        mat_files = sorted(glob.glob(os.path.join(p_dir, '*.mat')))
        
        p_X = []
        p_y = []
        
        for mf in mat_files:
            try:
                mat = sio.loadmat(mf)
                data = mat['data'][0][0]
                
                # Combine left and right eye data
                for eye in ['left', 'right']:
                    eye_data = data[eye][0][0]
                    gaze = eye_data['gaze']
                    images = eye_data['image']
                    
                    thetas, phis = vec_to_angles(gaze)
                    
                    for i in range(len(thetas)):
                        label = get_label(thetas[i], phis[i])
                        img_flat = images[i].flatten()
                        p_X.append(img_flat)
                        p_y.append(label)
            except Exception as e:
                print(f"Error loading {mf}: {e}")
                continue
        
        # Subsample properly if needed to prevent OOM
        if max_samples and len(p_X) > max_samples:
            idx = np.random.choice(len(p_X), max_samples, replace=False)
            p_X = [p_X[i] for i in idx]
            p_y = [p_y[i] for i in idx]
            
        X_list.extend(p_X)
        y_list.extend(p_y)
    
    return np.array(X_list, dtype=np.float32), np.array(y_list)

if __name__ == "__main__":
    # Quick test
    base = os.path.join("Dataset", "MPIIGAZE", "MPIIGaze")
    X, y = load_mpiigaze_data(base, max_samples=100)
    print("X shape:", X.shape)
    print("y length:", len(y))
    print("Label distribution:", np.unique(y, return_counts=True))
