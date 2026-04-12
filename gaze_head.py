import os
import numpy as np
import torch
import torch.nn as nn
try:
    import torch.ao.quantization
except ImportError:
    pass
import torch.optim as optim

from gaze_geometry import (
    GAZE_FEATURE_DIM,
    extract_gaze_feature_vector,
)


class HybridGazeModel(nn.Module):
    """
    Geometry-driven gaze regression: weighted landmark features -> MLP -> TCN -> head pose fusion -> (x, y) in [0,1].
    """

    def __init__(self, seq_len=15, input_dim=GAZE_FEATURE_DIM, feature_dim=64, pose_dim=3):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
            nn.ReLU(),
        )

        self.tcn = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + pose_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, eye_sequence, head_pose=None):
        batch_size, seq_len, dim = eye_sequence.shape

        x_flat = eye_sequence.view(-1, dim)
        features = self.feature_extractor(x_flat)

        features = features.view(batch_size, seq_len, -1).transpose(1, 2)
        tcn_out = self.tcn(features)
        last_timestep_feature = tcn_out[:, :, -1]

        if head_pose is None:
            head_pose = torch.zeros(batch_size, 3, device=last_timestep_feature.device)
        fused = torch.cat([last_timestep_feature, head_pose], dim=1)
        return self.regressor(fused)


class GazeHead:
    """
    Spatial-temporal gaze regression head with optional local fine-tuning (MSE on screen targets).
    """

    def __init__(self, model_path="models/gaze_hybrid_epoch2.pth", seq_len=15, input_dim=GAZE_FEATURE_DIM):
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.model_path = model_path
        self.int8_path = "models/gaze_hybrid_int8.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridGazeModel(seq_len=self.seq_len, input_dim=self.input_dim).to(self.device)

        self.patch_buffer = []

        if self.device.type == "cpu" and os.path.exists(self.int8_path):
            try:
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                state = torch.load(self.int8_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state, strict=True)
                print("[GazeHead] Loaded INT8 dynamically quantized gaze regression weights.")
            except Exception as e:
                print(f"[GazeHead] Failed to load INT8 gaze model, falling back to FP32: {e}")
                self._load_fp32(model_path)
        elif os.path.exists(model_path):
            self._load_fp32(model_path)
        else:
            print(f"[GazeHead] No weights at {model_path}; random init.")

    def _load_fp32(self, model_path):
        try:
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
            print("[GazeHead] Loaded gaze regression weights (FP32).")
        except Exception as e:
            print(f"[GazeHead] Could not load {model_path} ({e}). Using random init.")
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def _extract_geometric_features(self, keypoints):
        return extract_gaze_feature_vector(keypoints)

    def update_buffer(self, patch):
        self.patch_buffer.append(patch)
        if len(self.patch_buffer) > self.seq_len:
            self.patch_buffer.pop(0)

    def _get_sequence_tensor(self):
        d = self.input_dim
        if len(self.patch_buffer) == 0:
            seq = np.zeros((self.seq_len, d), dtype=np.float32)
        elif len(self.patch_buffer) < self.seq_len:
            pad_len = self.seq_len - len(self.patch_buffer)
            pad = [self.patch_buffer[0]] * pad_len
            seq = np.array(pad + self.patch_buffer, dtype=np.float32)
        else:
            seq = np.array(self.patch_buffer[-self.seq_len :], dtype=np.float32)

        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, keypoints, frame=None):
        """
        Returns categorical label for the integrity engine plus gaze_vals including
        continuous screen_x / screen_y from regression.

        Label is derived from the raw iris projection ratios (geometric signal)
        with head-pose compensation.  The ML model only provides the continuous
        screen_x / screen_y regression for calibration purposes.
        """
        req = (
            "left_iris",
            "right_iris",
            "left_inner",
            "left_outer",
            "right_inner",
            "right_outer",
        )
        if not all(k in keypoints for k in req):
            self.patch_buffer.clear()
            return "Off-screen", {"left_t": -1.0, "right_t": -1.0}

        from analytics import OFFSCREEN_HI, OFFSCREEN_LO, CENTER_LO, CENTER_HI, _proj_ratio

        left_t = _proj_ratio(
            keypoints["left_iris"], keypoints["left_inner"], keypoints["left_outer"]
        )
        right_t = _proj_ratio(
            keypoints["right_iris"], keypoints["right_inner"], keypoints["right_outer"]
        )

        # ── Head-pose compensation (same logic as compute_gaze) ────────
        hp = keypoints.get("head_pose", (0.0, 0.0, 0.0))
        if hp is not None and len(hp) >= 2:
            pitch, yaw = float(hp[0]), float(hp[1])
            yaw_comp = float(np.clip(yaw * 0.006, -0.15, 0.15))
            pitch_comp = float(np.clip(pitch * 0.003, -0.08, 0.08))
            left_t += yaw_comp + pitch_comp
            right_t += yaw_comp + pitch_comp

        gaze_vals = {"left_t": left_t, "right_t": right_t}

        if not (
            OFFSCREEN_LO <= left_t <= OFFSCREEN_HI
            and OFFSCREEN_LO <= right_t <= OFFSCREEN_HI
        ):
            self.patch_buffer.clear()
            return "Off-screen", gaze_vals

        # ── Categorical label from raw geometric ratios (ground truth) ─
        avg = (left_t + right_t) * 0.5
        if CENTER_LO <= avg <= CENTER_HI:
            label = "Center"
        elif avg < CENTER_LO:
            label = "Left"
        else:
            label = "Right"

        if "left_eye_points" not in keypoints or "right_eye_points" not in keypoints:
            return label, gaze_vals

        feature_vec = self._extract_geometric_features(keypoints)
        self.update_buffer(feature_vec)

        pose_tensor = torch.tensor([[hp[0], hp[1], hp[2]]], dtype=torch.float32).to(
            self.device
        )

        self.model.eval()
        with torch.no_grad():
            seq_tensor = self._get_sequence_tensor()
            pred = self.model(seq_tensor, head_pose=pose_tensor)
            x_norm, y_norm = pred[0].cpu().numpy()

            # ML provides continuous coordinates only (not the label)
            gaze_vals["screen_x"] = float(x_norm)
            gaze_vals["screen_y"] = float(y_norm)

            if len(self.patch_buffer) > 1:
                variance = float(np.var(self.patch_buffer, axis=0).mean())
                conf = max(0.1, 1.0 - (variance * 150.0))
            else:
                conf = 0.5
            gaze_vals["confidence"] = min(0.99, conf)

            return label, gaze_vals

    def fine_tune(self, keypoints, frame, target_x_norm, target_y_norm):
        req = (
            "left_iris",
            "left_inner",
            "left_outer",
            "right_inner",
            "right_outer",
            "right_iris",
            "left_eye_points",
            "right_eye_points",
        )
        if not all(k in keypoints for k in req):
            return 0.0

        feature_vec = self._extract_geometric_features(keypoints)
        self.update_buffer(feature_vec)

        hp = keypoints.get("head_pose", (0.0, 0.0, 0.0))
        pose_tensor = torch.tensor([[hp[0], hp[1], hp[2]]], dtype=torch.float32).to(
            self.device
        )

        self.model.train()
        self.optimizer.zero_grad()

        seq_tensor = self._get_sequence_tensor()
        target_tensor = torch.tensor(
            [[target_x_norm, target_y_norm]], dtype=torch.float32
        ).to(self.device)

        pred = self.model(seq_tensor, head_pose=pose_tensor)
        loss = self.criterion(pred, target_tensor)

        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save_weights(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[GazeHead] Saved weights to {self.model_path}")
