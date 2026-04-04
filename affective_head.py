import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from train_affective_head import EmotionCNN

class MultiModalStressTCN(nn.Module):
    """
    1D TCN that processes the temporal sequence of concatenated features:
    (DeepVector (256D) + BrowMicroTremor (1D) + BlinkState (1D) + HeadPose (3D)) = 261D
    Output: Stress Score [0.0 - 1.0]
    """
    def __init__(self, seq_len=15, feature_dim=261):
        super().__init__()
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
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is (Batch, SeqLen, FeatureDim) -> needs to be (Batch, FeatureDim, SeqLen)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last_timestep = out[:, :, -1]
        stress = self.regressor(last_timestep)
        return stress

class AffectiveHead:
    """
    Multi-Modal ML Affective Head.
    Extracts deep features from a CNN (Privacy First) and fuses with temporal geometries.
    """
    def __init__(self, cnn_path="models/affective_cnn.pth", seq_len=15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = None
        self.seq_len = seq_len
        
        # Mapping from class index to emotion string 
        self.idx_to_class = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        self.stress_emotions = {'angry', 'disgust', 'fear', 'sad'}
        
        # Load visual backbone
        if os.path.exists(cnn_path):
            try:
                base_model = EmotionCNN().to(self.device)
                base_model.load_state_dict(torch.load(cnn_path, map_location=self.device, weights_only=True))
                base_model.eval()
                
                # Monkey-patch to extract 256D continuous vector
                # The classifier is: Flatten, Linear(->256), ReLU, Dropout, Linear(->7)
                modules = list(base_model.features.children()) + list(base_model.classifier.children())[:-2]
                self.feature_extractor = nn.Sequential(*modules)
                self.feature_extractor.eval()
                
                # Keep the original model strictly for legacy emotion prediction backward compatibility
                self.legacy_model = base_model
                print(f"[AffectiveHead] ✅ Loaded Deep Feature Extractor on {self.device}.")
            except Exception as e:
                print(f"[AffectiveHead] ⚠️ Failed to load CNN backbone: {e}")
                self.legacy_model = None
        else:
            print(f"[AffectiveHead] ⚠️ CNN not found at {cnn_path}.")
            self.legacy_model = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Stress TCN Logic
        tcn_path = "models/stress_tcn.pth"
        self.tcn = MultiModalStressTCN(seq_len=seq_len).to(self.device)
        self.feature_buffer = []  # List of 258D numpy arrays
        
        if os.path.exists(tcn_path):
            try:
                self.tcn.load_state_dict(torch.load(tcn_path, map_location=self.device, weights_only=True))
            except Exception:
                pass
        
    def predict(self, keypoints, frame=None, temporal_geometries=None):
        """
        Extracts deep feature vectors, concatenates with temporal_geometries, and evaluates TCN.
        Returns (legacy_emotion, mapped_stress_level, stress_score_float)
        """
        # Defaults
        emotion = 'Neutral'
        level = 'Low'
        stress_score = 0.0
        
        if temporal_geometries is None:
            # Fallback if accessed via direct standalone call without temporal analytics
            temporal_geometries = (0.0, 0.0, 0.0, 0.0, 0.0) # (delta_brow_norm, blink_state, rx, ry, rz)
            
        deep_vector = np.zeros(256, dtype=np.float32)

        if self.feature_extractor is not None and frame is not None and 'face_bbox' in keypoints:
            x_min, y_min, x_max, y_max = keypoints['face_bbox']
            
            # Add padding
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
                    # Extract 256D embedding
                    feat = self.feature_extractor(tensor) # (1, 256)
                    deep_vector = feat[0].cpu().numpy()
                    
                    # Compute legacy emotion text
                    if self.legacy_model:
                        out = self.legacy_model(tensor)
                        probs = torch.nn.functional.softmax(out, dim=1)[0]
                        emotion = self.idx_to_class[torch.argmax(probs).item()].capitalize()
        
        # Multi-modal fusion
        delta_brow_norm, blink_state, rx, ry, rz = temporal_geometries
        # Feature Vector: 261D
        combined_vector = np.concatenate([deep_vector, [delta_brow_norm, blink_state, rx, ry, rz]]).astype(np.float32)
        
        self.feature_buffer.append(combined_vector)
        if len(self.feature_buffer) > self.seq_len:
            self.feature_buffer.pop(0)
            
        # Run TCN inference
        if len(self.feature_buffer) > 0:
            pad_len = self.seq_len - len(self.feature_buffer)
            seq = np.array([self.feature_buffer[0]] * pad_len + self.feature_buffer)
            
            seq_tensor = torch.tensor(seq).unsqueeze(0).to(self.device) # (1, Seq_len, 258)
            
            self.tcn.eval()
            with torch.no_grad():
                stress_res = self.tcn(seq_tensor)
                stress_score = stress_res[0, 0].item()
                
            # Map continuous stress to categorical levels
            if stress_score >= 0.7:
                level = 'High'
            elif stress_score >= 0.4:
                level = 'Medium'
            else:
                level = 'Low'
                
        return emotion, level, stress_score
