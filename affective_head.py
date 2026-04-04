import os
import cv2
import torch
import torchvision.transforms as transforms
from train_affective_head import EmotionCNN

class AffectiveHead:
    """
    ML-based Affective Head using lightweight CNN trained on FER-2013.
    """
    def __init__(self, model_path="models/affective_cnn.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Mapping from class index to emotion string (from load_fer2013.py)
        self.idx_to_class = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        
        # Stress mapping (binary)
        self.stress_emotions = {'angry', 'disgust', 'fear', 'sad'}
        
        if os.path.exists(model_path):
            try:
                self.model = EmotionCNN().to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                print(f"[AffectiveHead] ✅ Loaded CNN model on {self.device}.")
            except Exception as e:
                print(f"[AffectiveHead] ⚠️ Failed to load model: {e}")
        else:
            print(f"[AffectiveHead] ⚠️ Model not found at {model_path}. Will use fallback.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, keypoints, frame=None):
        """
        Runs ML prediction if available.
        Otherwise falls back to the old heuristic.
        """
        # 1. Fallback heuristic
        from analytics import compute_stress
        level, heuristic_score = compute_stress(keypoints)
        
        if self.model is None or frame is None or 'face_bbox' not in keypoints:
            return 'Neutral', level, heuristic_score

        # 2. Extract face crop using MediaPipe bounding box
        x_min, y_min, x_max, y_max = keypoints['face_bbox']
        
        # Add padding (e.g., 20%) to capture full face dynamically
        w = x_max - x_min
        h = y_max - y_min
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(frame.shape[1], x_max + pad_x)
        y_max = min(frame.shape[0], y_max + pad_y)
        
        face_crop = frame[y_min:y_max, x_min:x_max]
        if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            return 'Neutral', level, heuristic_score
            
        # 3. Preprocess for CNN
        if len(face_crop.shape) == 3:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
        face_crop = cv2.resize(face_crop, (48, 48))
        
        # PyTorch requires specific shapes and types mapping
        tensor = self.transform(face_crop).unsqueeze(0).to(self.device).float()
        
        # 4. Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            pred_idx = torch.argmax(probs).item()
            emotion = self.idx_to_class[pred_idx]
            
            # Calculate stress score by summing probabilities of stress-class emotions
            stress_prob = sum(probs[i].item() for i, emo in self.idx_to_class.items() if emo in self.stress_emotions)
            
            # Map back to Low/Medium/High for backward compatibility
            if stress_prob >= 0.7:
                level = 'High'
            elif stress_prob >= 0.4:
                level = 'Medium'
            else:
                level = 'Low'
                
            return emotion.capitalize(), level, stress_prob
