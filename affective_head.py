import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from train_affective_head import EmotionCNN, DEFAULT_EMBED_DIM

# Deep embedding + temporal geometry: delta_brow, blink_state, micro_tremor, blink_z, pose×3
TEMPORAL_DIM = 7

DEFAULT_CNN_PATH = os.path.join("models", "affective_cnn.pth")
#  DISFA_CNN_PATH = os.path.join("models", "affective_cnn_disfa.pt") 


def resolve_affective_cnn_path(cnn_path=None):
    """Prefer DISFA-fine-tuned weights when present; else FER baseline."""
    if cnn_path is not None:
        return cnn_path
    if os.path.isfile(DEFAULT_CNN_PATH):
        return DEFAULT_CNN_PATH
    return DEFAULT_CNN_PATH


def _infer_embed_dim_from_state_dict(state: dict) -> int:
    w = state.get("classifier.1.weight")
    if w is not None and hasattr(w, "shape"):
        return int(w.shape[0])
    if isinstance(w, np.ndarray):
        return int(w.shape[0])
    return DEFAULT_EMBED_DIM


class MultiModalStressTCN(nn.Module):
    """
    TCN over [embed || delta_brow || blink || micro_tremor || blink_z || head_pose×3].
    """

    def __init__(self, seq_len=15, embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        feature_dim = embed_dim + TEMPORAL_DIM
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.tcn = nn.Sequential(
            nn.Conv1d(feature_dim, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last_timestep = out[:, :, -1]
        return self.regressor(last_timestep)


class AffectiveHead:
    """
    Multi-modal distress path: 512-D (default) face embedding + landmark-derived temporal cues.
    Legacy 256-D checkpoints still load; TCN is sized to match.
    """

    def __init__(self, cnn_path=None, seq_len=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = int(seq_len)
        self.feature_extractor = None
        self.legacy_model = None
        self.embed_dim = DEFAULT_EMBED_DIM

        cnn_path = resolve_affective_cnn_path(cnn_path)
        self._cnn_path = cnn_path

        self.idx_to_class = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }
        self.stress_emotions = {"angry", "disgust", "fear", "sad"}

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        if os.path.exists(cnn_path):
            try:
                try:
                    state = torch.load(
                        cnn_path, map_location=self.device, weights_only=True
                    )
                except TypeError:
                    state = torch.load(cnn_path, map_location=self.device)
                self.embed_dim = _infer_embed_dim_from_state_dict(state)
                base_model = EmotionCNN(embed_dim=self.embed_dim).to(self.device)
                base_model.load_state_dict(state, strict=True)
                base_model.eval()

                modules = list(base_model.features.children()) + list(
                    base_model.classifier.children()
                )[:-2]
                self.feature_extractor = nn.Sequential(*modules)
                self.feature_extractor.eval()
                self.legacy_model = base_model
                tag = "DISFA AU4" if cnn_path.replace("\\", "/").endswith(
                    "affective_cnn_disfa.pt"
                ) else "FER"
                print(
                    f"[AffectiveHead] Loaded CNN ({tag}) embed_dim={self.embed_dim} "
                    f"from {cnn_path} on {self.device}."
                )
            except Exception as e:
                print(f"[AffectiveHead] Failed to load CNN: {e}")
                self.legacy_model = None
        else:
            print(f"[AffectiveHead] CNN not found at {cnn_path}.")

        tcn_path = "models/stress_tcn.pth"
        self.tcn = MultiModalStressTCN(
            seq_len=self.seq_len, embed_dim=self.embed_dim
        ).to(self.device)
        self.feature_buffer = []

        if os.path.exists(tcn_path):
            try:
                try:
                    tstate = torch.load(
                        tcn_path, map_location=self.device, weights_only=True
                    )
                except TypeError:
                    tstate = torch.load(tcn_path, map_location=self.device)
                self.tcn.load_state_dict(tstate, strict=True)
                print("[AffectiveHead] Loaded stress TCN.")
            except Exception:
                print(
                    "[AffectiveHead] stress_tcn.pth mismatch or missing — using random TCN (retrain recommended)."
                )

    def predict(self, keypoints, frame=None, temporal_geometries=None):
        """
        temporal_geometries: tuple of 7 floats
          (delta_brow_norm, blink_state, micro_tremor_norm, blink_rate_z, rx, ry, rz)
        """
        emotion = "Neutral"
        level = "Low"
        stress_score = 0.0

        if temporal_geometries is None:
            temporal_geometries = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        deep_vector = np.zeros(self.embed_dim, dtype=np.float32)

        if self.feature_extractor is not None and frame is not None and "face_bbox" in keypoints:
            x_min, y_min, x_max, y_max = keypoints["face_bbox"]
            w, h = x_max - x_min, y_max - y_min
            pad_x, pad_y = int(w * 0.2), int(h * 0.2)
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(frame.shape[1], x_max + pad_x)
            y_max = min(frame.shape[0], y_max + pad_y)

            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop.size > 0 and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                if len(face_crop.shape) == 3:
                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_crop = cv2.resize(face_crop, (48, 48))
                tensor = self.transform(face_crop).unsqueeze(0).to(self.device).float()

                with torch.no_grad():
                    feat = self.feature_extractor(tensor)
                    deep_vector = feat[0].cpu().numpy()

                    if self.legacy_model:
                        out = self.legacy_model(tensor)
                        probs = torch.nn.functional.softmax(out, dim=1)[0]
                        emotion = self.idx_to_class[
                            torch.argmax(probs).item()
                        ].capitalize()

        if len(temporal_geometries) != TEMPORAL_DIM:
            temporal_geometries = tuple(temporal_geometries[:TEMPORAL_DIM]) + (
                0.0,
            ) * max(0, TEMPORAL_DIM - len(temporal_geometries))

        combined_vector = np.concatenate(
            [deep_vector, np.asarray(temporal_geometries, dtype=np.float32)]
        ).astype(np.float32)

        self.feature_buffer.append(combined_vector)
        if len(self.feature_buffer) > self.seq_len:
            self.feature_buffer.pop(0)

        if len(self.feature_buffer) > 0:
            pad_len = self.seq_len - len(self.feature_buffer)
            seq = np.array([self.feature_buffer[0]] * pad_len + self.feature_buffer)

            seq_tensor = torch.tensor(seq).unsqueeze(0).to(self.device)

            self.tcn.eval()
            with torch.no_grad():
                stress_res = self.tcn(seq_tensor)
                stress_score = stress_res[0, 0].item()

            if stress_score >= 0.7:
                level = "High"
            elif stress_score >= 0.4:
                level = "Medium"
            else:
                level = "Low"

        return emotion, level, stress_score
