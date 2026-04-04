import os
import cv2
import numpy as np
import joblib

class GazeHead:
    """
    ML-based Gaze Head using Random Forest trained on MPIIGaze.
    Falls back to heuristic if the model is not found.
    """
    def __init__(self, model_path="models/gaze_rf.joblib"):
        self.model = None
        self.classes = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.classes = self.model.classes_
                print("[GazeHead] ✅ Loaded Random Forest model successfully.")
            except Exception as e:
                print(f"[GazeHead] ⚠️ Failed to load model: {e}")
        else:
            print(f"[GazeHead] ⚠️ Model not found at {model_path}. Will use fallback.")

    def _extract_eye_patch(self, frame, inner, outer, iris):
        """
        Extracts a 36x60 eye patch from the frame based on landmarks.
        `inner`, `outer`, `iris` are (x,y) pixel coordinates.
        """
        # Calculate eye width and height (heuristic bounding box)
        w = int(abs(outer[0] - inner[0]) * 1.5)
        h = int(w * (36.0 / 60.0))
        
        # Center around iris
        cx, cy = int(iris[0]), int(iris[1])
        
        x_min = max(0, cx - w // 2)
        y_min = max(0, cy - h // 2)
        x_max = min(frame.shape[1], x_min + w)
        y_max = min(frame.shape[0], y_min + h)
        
        patch = frame[y_min:y_max, x_min:x_max]
        
        if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
            return None
            
        # Convert to grayscale and resize to 36x60 precisely
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(patch, (60, 36)) # width=60, height=36
        
        return patch.flatten().astype(np.float32)

    def predict(self, keypoints, frame=None):
        """
        Runs ML prediction if available and frame is provided.
        Otherwise falls back to the old heuristic.
        """
        # 1. Fallback to heuristic
        req = ('left_iris', 'right_iris', 'left_inner', 'left_outer', 'right_inner', 'right_outer')
        if not all(k in keypoints for k in req):
            return 'Off-screen', {'left_t': -1.0, 'right_t': -1.0}
            
        # Calculate the heuristic ratios for backward compatibility for the HUD
        from analytics import _proj_ratio, OFFSCREEN_LO, OFFSCREEN_HI
        left_t  = _proj_ratio(keypoints['left_iris'],  keypoints['left_inner'],  keypoints['left_outer'])
        right_t = _proj_ratio(keypoints['right_iris'],  keypoints['right_inner'], keypoints['right_outer'])
        gaze_vals = {'left_t': left_t, 'right_t': right_t}
        
        # Override heuristical OFFSCREEN checking
        if not (OFFSCREEN_LO <= left_t <= OFFSCREEN_HI and OFFSCREEN_LO <= right_t <= OFFSCREEN_HI):
            return 'Off-screen', gaze_vals
            
        # If no model or no frame, use heuristics
        if self.model is None or frame is None:
            avg = (left_t + right_t) * 0.5
            if 0.35 <= avg <= 0.65:
                return 'Center', gaze_vals
            elif avg < 0.35:
                return 'Left', gaze_vals
            else:
                return 'Right', gaze_vals

        # 2. Run Random Forest
        left_patch = self._extract_eye_patch(frame, keypoints['left_inner'], keypoints['left_outer'], keypoints['left_iris'])
        right_patch = self._extract_eye_patch(frame, keypoints['right_inner'], keypoints['right_outer'], keypoints['right_iris'])
        
        probs = np.zeros(len(self.classes))
        valid_patches = 0
        
        if left_patch is not None:
            probs += self.model.predict_proba([left_patch])[0]
            valid_patches += 1
        if right_patch is not None:
            probs += self.model.predict_proba([right_patch])[0]
            valid_patches += 1
            
        if valid_patches == 0:
            return 'Center', gaze_vals # default

        probs /= valid_patches
        pred_idx = np.argmax(probs)
        return self.classes[pred_idx], gaze_vals
