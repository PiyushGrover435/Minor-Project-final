import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class HybridGazeModel(nn.Module):
    """
    Structured Geometry Model for Gaze Estimation with 6-DOF Head Pose correction.
    MLP processes mesh coordinates (EAR + Iris Vectors) -> TCN filters temporal jitter -> 
    Head Pose (Pitch,Yaw,Roll) concatenated -> maps to screen (x, y).
    """
    def __init__(self, seq_len=15, input_dim=6, feature_dim=64, pose_dim=3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Spatial Feature Extractor (MLP on geometric coordinates)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
            nn.ReLU()
        )
        
        # Temporal Convolutional Network (1D Dilated Convolutions)
        # Input shape expected by Conv1d: (Batch, Channels, SeqLen)
        self.tcn = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        
        # Dense Regressor: TCN features (64D) + Head Pose (3D) -> screen (x, y)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + pose_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # (x, y)
            nn.Sigmoid()      # output normalized to [0, 1] screen bounds
        )

    def forward(self, eye_sequence, head_pose=None):
        """
        Parameters
        ----------
        eye_sequence : (Batch, SeqLen, input_dim) - Geometric feature vectors
        head_pose    : (Batch, 3) — Pitch, Yaw, Roll in degrees.  If None, zeros.
        """
        batch_size, seq_len, dim = eye_sequence.shape
        
        # Fold sequence into batch dimension for MLP
        x_flat = eye_sequence.view(-1, dim)
        features = self.feature_extractor(x_flat) # (Batch * SeqLen, feature_dim)
        
        # Reshape for TCN: (Batch, feature_dim, SeqLen)
        features = features.view(batch_size, seq_len, -1).transpose(1, 2)
        
        # Temporal processing
        tcn_out = self.tcn(features) # (Batch, feature_dim, SeqLen)
        
        # Take the feature of the most recent timestep (last element in sequence)
        last_timestep_feature = tcn_out[:, :, -1] # (Batch, feature_dim)
        
        # Fuse with head pose
        if head_pose is None:
            head_pose = torch.zeros(batch_size, 3, device=last_timestep_feature.device)
        fused = torch.cat([last_timestep_feature, head_pose], dim=1) # (Batch, feature_dim + 3)
        
        # Regress to (x, y)
        out = self.regressor(fused)
        return out


class GazeHead:
    """
    ML-based Spatial-Temporal Gaze Head.
    Maintains a temporal buffer of eye patches and evaluates the TCN.
    Provides local on-device fine-tuning via SGD.
    """
    def __init__(self, model_path="models/gaze_hybrid.pth", seq_len=15):
        self.seq_len = seq_len
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridGazeModel(seq_len=seq_len).to(self.device)
        
        # Temporal buffer for the sequence [seq_len, 1, 36, 60]
        # We store NumPy arrays, only converting to Tensor on prediction
        self.patch_buffer = []
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("[GazeHead] ✅ Loaded Spatial-Temporal Hybrid model.")
            except Exception as e:
                print(f"[GazeHead] ⚠️ Failed to load model weights: {e}")
        else:
            print(f"[GazeHead] ⚠️ Model weights not found at {model_path}. Initialized randomly.")
            os.makedirs("models", exist_ok=True)
            
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def _extract_geometric_features(self, keypoints):
        """
        Extract a 1D tensor representing normalized mesh features.
        6 features: Normalized Left/Right Iris Vectors, Left/Right EAR.
        """
        left_inner = np.asarray(keypoints['left_inner'])
        right_inner = np.asarray(keypoints['right_inner'])
        interocular = np.linalg.norm(left_inner - right_inner)
        if interocular < 1e-5:
            interocular = 1.0

        def norm_vec(p1, p0):
            v = np.asarray(p1) - np.asarray(p0)
            return float(v[0]) / interocular, float(v[1]) / interocular

        lx, ly = norm_vec(keypoints['left_iris'], keypoints['left_inner'])
        rx, ry = norm_vec(keypoints['right_iris'], keypoints['right_inner'])

        from analytics import RealtimeAnalyzer
        left_ear = RealtimeAnalyzer._eye_aspect_ratio(keypoints['left_eye_points'])
        right_ear = RealtimeAnalyzer._eye_aspect_ratio(keypoints['right_eye_points'])

        vec = np.array([lx, ly, rx, ry, left_ear, right_ear], dtype=np.float32)
        return vec

    def update_buffer(self, patch):
        self.patch_buffer.append(patch)
        if len(self.patch_buffer) > self.seq_len:
            self.patch_buffer.pop(0)

    def _get_sequence_tensor(self):
        """Returns sequence tensor of shape (1, seq_len, input_dim)."""
        input_dim = 6
        if len(self.patch_buffer) == 0:
            seq = np.zeros((self.seq_len, input_dim), dtype=np.float32)
        elif len(self.patch_buffer) < self.seq_len:
            pad_len = self.seq_len - len(self.patch_buffer)
            pad = [self.patch_buffer[0]] * pad_len
            seq = np.array(pad + self.patch_buffer)
        else:
            seq = np.array(self.patch_buffer)
        
        # Map to PyTorch format (1, Seq_len, input_dim)
        seq = torch.tensor(seq).unsqueeze(0).to(self.device)
        return seq

    def predict(self, keypoints, frame=None):
        """
        Runs ML prediction using TCN if frame is provided.
        Falls back to heuristic for missing frame or early tracking.
        Returns: label, {'left_t', 'right_t'} for backward compatibility with HUD.
        """
        req = ('left_iris', 'right_iris', 'left_inner', 'left_outer', 'right_inner', 'right_outer')
        if not all(k in keypoints for k in req):
            self.patch_buffer.clear()
            return 'Off-screen', {'left_t': -1.0, 'right_t': -1.0}
            
        # Heuristic ratio computation for HUD visualization and Off-screen strict check
        from analytics import _proj_ratio, OFFSCREEN_LO, OFFSCREEN_HI
        left_t  = _proj_ratio(keypoints['left_iris'],  keypoints['left_inner'],  keypoints['left_outer'])
        right_t = _proj_ratio(keypoints['right_iris'],  keypoints['right_inner'], keypoints['right_outer'])
        gaze_vals = {'left_t': left_t, 'right_t': right_t}
        
        if not (OFFSCREEN_LO <= left_t <= OFFSCREEN_HI and OFFSCREEN_LO <= right_t <= OFFSCREEN_HI):
            self.patch_buffer.clear()
            return 'Off-screen', gaze_vals

        if 'left_eye_points' not in keypoints or 'right_eye_points' not in keypoints:
            # Fallback if the full contour points aren't available yet
            avg = (left_t + right_t) * 0.5
            return 'Center' if 0.35 <= avg <= 0.65 else 'Left' if avg < 0.35 else 'Right', gaze_vals

        # Extract structured geometric vector and update temporal dimension
        feature_vec = self._extract_geometric_features(keypoints)
        self.update_buffer(feature_vec)

        # Extract head pose for pose-corrected prediction
        hp = keypoints.get('head_pose', (0.0, 0.0, 0.0))
        pose_tensor = torch.tensor([[hp[0], hp[1], hp[2]]], dtype=torch.float32).to(self.device)

        # Predict screen coordinates
        self.model.eval()
        with torch.no_grad():
            seq_tensor = self._get_sequence_tensor()
            pred = self.model(seq_tensor, head_pose=pose_tensor) # (1, 2)
            x_norm, y_norm = pred[0].cpu().numpy()
            
            # Map normalized x screen coordinate back to generic categorical zones 
            # for smooth Integration Engine compatibility
            if 0.35 <= x_norm <= 0.65:
                label = 'Center'
            elif x_norm < 0.35:
                label = 'Left'
            else:
                label = 'Right'
                
            # Embed the continuous coordinates for external rendering if needed
            gaze_vals['screen_x'] = float(x_norm)
            gaze_vals['screen_y'] = float(y_norm)
            
            # Predict pseudo softmax confidence based on geometric jitter (low variance = high confidence)
            if len(self.patch_buffer) > 1:
                variance = float(np.var(self.patch_buffer, axis=0).mean())
                conf = max(0.1, 1.0 - (variance * 150.0))
            else:
                conf = 0.5
            gaze_vals['confidence'] = min(0.99, conf)
            
            return label, gaze_vals

    def fine_tune(self, keypoints, frame, target_x_norm, target_y_norm):
        """
        Local personalization via Stochastic Gradient Descent.
        Runs a single backward pass mapping the current eye sequence to the known screen point.
        Head Pose is included so the model learns to compensate for head movement.
        """
        req = ('left_iris', 'left_inner', 'left_eye_points', 'right_eye_points')
        if not all(k in keypoints for k in req):
            return 0.0

        feature_vec = self._extract_geometric_features(keypoints)
        self.update_buffer(feature_vec)
        
        hp = keypoints.get('head_pose', (0.0, 0.0, 0.0))
        pose_tensor = torch.tensor([[hp[0], hp[1], hp[2]]], dtype=torch.float32).to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        seq_tensor = self._get_sequence_tensor()
        target_tensor = torch.tensor([[target_x_norm, target_y_norm]], dtype=torch.float32).to(self.device)
        
        pred = self.model(seq_tensor, head_pose=pose_tensor)
        loss = self.criterion(pred, target_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_weights(self):
        """Save the fine-tuned model weights to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[GazeHead] Local weights saved to {self.model_path}")
