import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class HybridGazeModel(nn.Module):
    """
    Spatial-Temporal Hybrid Model for Gaze Estimation.
    CNN processes individual eye patches -> TCN filters temporal jitter over sequences -> maps to screen (x, y).
    """
    def __init__(self, seq_len=15, feature_dim=64):
        super().__init__()
        # Spatial Feature Extractor (Tiny CNN) - input: 1x36x60
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 18x30
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 9x15
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1), # 5x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Global average pooling
            nn.Flatten() # outputs feature_dim
        )
        
        # Temporal Convolutional Network (1D Dilated Convolutions)
        # Input shape expected by Conv1d: (Batch, Channels, SeqLen)
        self.tcn = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        
        # Dense Regressor to continuous (x, y) screen coordinates
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # (x, y)
            nn.Sigmoid()      # output normalized to [0, 1] screen bounds
        )

    def forward(self, eye_sequence):
        # eye_sequence shape: (Batch, SeqLen, 1, 36, 60)
        batch_size, seq_len, c, h, w = eye_sequence.shape
        
        # Fold sequence into batch dimension for CNN
        x_cnn = eye_sequence.view(-1, c, h, w)
        features = self.cnn(x_cnn) # (Batch * SeqLen, feature_dim)
        
        # Reshape for TCN: (Batch, feature_dim, SeqLen)
        features = features.view(batch_size, seq_len, -1).transpose(1, 2)
        
        # Temporal processing
        tcn_out = self.tcn(features) # (Batch, feature_dim, SeqLen)
        
        # Take the feature of the most recent timestep (last element in sequence)
        last_timestep_feature = tcn_out[:, :, -1] # (Batch, feature_dim)
        
        # Regress to (x, y)
        out = self.regressor(last_timestep_feature)
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

    def _extract_eye_patch(self, frame, inner, outer, iris):
        """
        Extracts a 36x60 eye patch from the frame.
        We use the average of both eyes to form a single patch, or just the left eye for simplicity.
        To handle both eyes simultaneously, we'll fuse them horizontally: 36x120 -> resize to 36x60.
        Actually, simpler standard is just tracking the left eye patch, or stacking them. 
        For this prototype, we'll extract left eye, as sequence processing is computationally heavier.
        """
        w = int(abs(outer[0] - inner[0]) * 1.5)
        if w == 0: w = 30
        h = int(w * (36.0 / 60.0))
        
        cx, cy = int(iris[0]), int(iris[1])
        x_min = max(0, cx - w // 2)
        y_min = max(0, cy - h // 2)
        x_max = min(frame.shape[1], x_min + w)
        y_max = min(frame.shape[0], y_min + h)
        
        patch = frame[y_min:y_max, x_min:x_max]
        if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
            return np.zeros((36, 60), dtype=np.float32)
            
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(patch, (60, 36))
        
        # Normalize patch to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        return patch

    def update_buffer(self, patch):
        self.patch_buffer.append(patch)
        if len(self.patch_buffer) > self.seq_len:
            self.patch_buffer.pop(0)

    def _get_sequence_tensor(self):
        """Returns sequence tensor of shape (1, seq_len, 1, 36, 60). Pads if buffer is small."""
        if len(self.patch_buffer) == 0:
            seq = np.zeros((self.seq_len, 36, 60), dtype=np.float32)
        elif len(self.patch_buffer) < self.seq_len:
            pad_len = self.seq_len - len(self.patch_buffer)
            pad = [self.patch_buffer[0]] * pad_len
            seq = np.array(pad + self.patch_buffer)
        else:
            seq = np.array(self.patch_buffer)
        
        # Map to PyTorch format
        seq = torch.tensor(seq).unsqueeze(0).unsqueeze(2).to(self.device) # (1, Seq_len, 1, 36, 60)
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

        if frame is None:
            # Fallback
            avg = (left_t + right_t) * 0.5
            return 'Center' if 0.35 <= avg <= 0.65 else 'Left' if avg < 0.35 else 'Right', gaze_vals

        # Extract patch and update temporal dimension
        patch = self._extract_eye_patch(frame, keypoints['left_inner'], keypoints['left_outer'], keypoints['left_iris'])
        self.update_buffer(patch)

        # Predict screen coordinates
        self.model.eval()
        with torch.no_grad():
            seq_tensor = self._get_sequence_tensor()
            pred = self.model(seq_tensor) # (1, 2)
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
            
            return label, gaze_vals

    def fine_tune(self, keypoints, frame, target_x_norm, target_y_norm):
        """
        Local personalization via Stochastic Gradient Descent.
        Runs a single backward pass mapping the current eye sequence to the known screen point.
        """
        req = ('left_iris', 'left_inner', 'left_outer')
        if not all(k in keypoints for k in req) or frame is None:
            return 0.0

        patch = self._extract_eye_patch(frame, keypoints['left_inner'], keypoints['left_outer'], keypoints['left_iris'])
        self.update_buffer(patch)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        seq_tensor = self._get_sequence_tensor()
        target_tensor = torch.tensor([[target_x_norm, target_y_norm]], dtype=torch.float32).to(self.device)
        
        pred = self.model(seq_tensor)
        loss = self.criterion(pred, target_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_weights(self):
        """Save the fine-tuned model weights to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[GazeHead] Local weights saved to {self.model_path}")
