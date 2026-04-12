import os
import cv2
import numpy as np

# Attempt to load TFLite runtime or full TensorFlow
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
    HAS_TFLITE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        HAS_TFLITE = True
    except ImportError:
        tflite = None
        HAS_TFLITE = False

class TFLiteInferenceEngine:
    """
    High-Performance wrapper for running INT8 Fully Quantized TFLite models.
    Supports de-quantization for output tensors automatically.
    """
    def __init__(self, model_path):
        if not HAS_TFLITE:
            raise RuntimeError("tflite_runtime or tensorflow not installed.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at {model_path}")
            
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def _quantize_input(self, data: np.ndarray, detail_idx: int) -> np.ndarray:
        """Quantizes an FP32 numpy array to INT8 using the model's IO parameters."""
        details = self.input_details[detail_idx]
        if details['dtype'] == np.int8:
            scale, zero_point = details['quantization']
            if scale == 0.0: 
                return data.astype(np.int8) # safeguard against unquantized dims
            q_data = np.round(data / scale) + zero_point
            q_data = np.clip(q_data, -128, 127).astype(np.int8)
            return q_data
        return data.astype(np.float32)

    def _dequantize_output(self, q_data: np.ndarray, detail_idx: int) -> np.ndarray:
        """De-quantizes an INT8 output back to FP32."""
        details = self.output_details[detail_idx]
        if details['dtype'] == np.int8:
            scale, zero_point = details['quantization']
            if scale == 0.0:
                return q_data.astype(np.float32)
            fp_data = (q_data.astype(np.float32) - zero_point) * scale
            return fp_data
        return q_data.astype(np.float32)

    def infer(self, *inputs) -> list:
        """
        Runs inference handling all quantization and dequantization seamlessly.
        Inputs should match the order expected by the model.
        """
        if len(inputs) != len(self.input_details):
            raise ValueError(f"Expected {len(self.input_details)} inputs but got {len(inputs)}")
            
        for i, in_data in enumerate(inputs):
            in_tensor = self._quantize_input(np.array(in_data), i)
            # Ensure shape matches, reshaping gently if batch dim is missing
            expected_shape = self.input_details[i]['shape']
            if in_tensor.shape != tuple(expected_shape):
                in_tensor = np.reshape(in_tensor, expected_shape)
            self.interpreter.set_tensor(self.input_details[i]['index'], in_tensor)
            
        self.interpreter.invoke()
        
        outputs = []
        for i in range(len(self.output_details)):
            out_tensor = self.interpreter.get_tensor(self.output_details[i]['index'])
            outputs.append(self._dequantize_output(out_tensor, i))
            
        return outputs


# ============================================================================
#  Seamless Compatibility Wrappers for analytics.py
# ============================================================================

from affective_head import TEMPORAL_DIM
from gaze_geometry import extract_gaze_feature_vector

class TFLiteAffectiveHead:
    """Seamless drop-in replacement for PyTorch AffectiveHead using TFLite INT8."""
    def __init__(self, cnn_path="models/affective_cnn_int8.tflite", tcn_path="models/affective_tcn_int8.tflite"):
        self.cnn_engine = TFLiteInferenceEngine(cnn_path) if os.path.exists(cnn_path) else None
        self.tcn_engine = TFLiteInferenceEngine(tcn_path) if os.path.exists(tcn_path) else None
        
        self.idx_to_class = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        self.feature_buffer = []
        self.seq_len = 15
        
        # Dynamically determine embed_dim from the TFLite model output shape
        self.embed_dim = 256
        if self.cnn_engine is not None and len(self.cnn_engine.output_details) > 0:
            out_shape = self.cnn_engine.output_details[0]['shape']
            if len(out_shape) > 1:
                self.embed_dim = int(out_shape[-1])
        
        print(f"[TFLiteInferenceEngine] Initialized AffectiveHead (INT8) with embed_dim={self.embed_dim}.")
        
    def predict(self, keypoints, frame=None, temporal_geometries=None):
        emotion = 'Neutral'
        level = 'Low'
        stress_score = 0.0
        
        # We need the frame to extract CNN features
        if frame is not None and self.cnn_engine is not None and 'bbox' in keypoints:
            x, y, w, h = keypoints['bbox']
            # Safeguard bounds
            x = max(0, x); y = max(0, y)
            w = min(w, frame.shape[1] - x); h = min(h, frame.shape[0] - y)
            
            if w > 10 and h > 10:
                face_img = frame[y:y+h, x:x+w]
                # Preprocess matches PyTorch transforms
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_tensor = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
                face_tensor = np.reshape(face_tensor, (1, 1, 48, 48))
                
                # Execute INT8 inference returning both emotion probs and the embedding
                outputs = self.cnn_engine.infer(face_tensor)
                
                # Depending on TF internal tensor ordering, shape [7] is probs, [embed_dim] is the feature
                if outputs[0].size == 7:
                    probs, cnn_out = outputs[0][0], outputs[1][0]
                else:
                    cnn_out, probs = outputs[0][0], outputs[1][0]
                
                # Apply softmax equivalent and extract emotion label
                emotion_idx = int(np.argmax(probs))
                emotion = self.idx_to_class.get(emotion_idx, 'Neutral').capitalize()
                
                # Ensure it's correctly shaped as (1, 256)
                cnn_out = cnn_out.reshape(1, self.embed_dim)
            else:
                cnn_out = np.zeros((1, self.embed_dim), dtype=np.float32)
        else:
            cnn_out = np.zeros((1, self.embed_dim), dtype=np.float32)
            
        temp_data = temporal_geometries if temporal_geometries else (0.0,) * TEMPORAL_DIM
        temp_arr = np.array(temp_data, dtype=np.float32).reshape(1, TEMPORAL_DIM)
        
        fused = np.concatenate([cnn_out, temp_arr], axis=1) # (1, 512 + 12)
        self.feature_buffer.append(fused)
        if len(self.feature_buffer) > self.seq_len:
            self.feature_buffer.pop(0)
            
        if len(self.feature_buffer) == self.seq_len and self.tcn_engine is not None:
            # seq_tensor shape (1, 15, dim)
            seq_tensor = np.stack(self.feature_buffer, axis=1)
            # Execute INT8 TCN
            tcn_out = self.tcn_engine.infer(seq_tensor)[0]
            stress_score = float(tcn_out[0][0])
            
            if stress_score > 0.7:
                level = 'High'
            elif stress_score > 0.4:
                level = 'Medium'
        
        return emotion, level, stress_score


class TFLiteGazeHead:
    """Seamless drop-in replacement for PyTorch GazeHead using TFLite INT8."""
    def __init__(self, model_path="models/gaze_hybrid_int8.tflite", seq_len=15):
        self.engine = TFLiteInferenceEngine(model_path) if os.path.exists(model_path) else None
        self.seq_len = seq_len
        self.feature_buffer = []
        self.patch_buffer = [] # For backwards compatibility with variance logic
        
        if self.engine:
            print("[TFLiteInferenceEngine] Initialized GazeHead (INT8).")
        
    def predict(self, keypoints, frame=None):
        label, gaze_vals = self._get_geom_label(keypoints)
        
        if "left_eye_points" not in keypoints or "right_eye_points" not in keypoints:
            return label, gaze_vals
            
        feature_vec = extract_gaze_feature_vector(keypoints)
        self.feature_buffer.append(feature_vec)
        if len(self.feature_buffer) > self.seq_len:
            self.feature_buffer.pop(0)
            
        if self.engine is not None and len(self.feature_buffer) == self.seq_len:
            hp = keypoints.get('head_pose', (0.0, 0.0, 0.0))
            seq_arr = np.array(self.feature_buffer, dtype=np.float32).reshape(1, self.seq_len, 10)
            pose_arr = np.array(hp, dtype=np.float32).reshape(1, 3)
            # Execute INT8 Hybrid Inference
            # TF Lite often reorders inputs alphabetically or by graph logic. We map by shape.
            if self.engine.input_details[0]['shape'][-1] == 3:
                out = self.engine.infer(pose_arr, seq_arr)[0]
            else:
                out = self.engine.infer(seq_arr, pose_arr)[0]
            gaze_vals['screen_x'] = float(out[0][0])
            gaze_vals['screen_y'] = float(out[0][1])
            gaze_vals['confidence'] = 0.95
            
        return label, gaze_vals

    def _get_geom_label(self, keypoints):
        # Mirror the exact logic of the PyTorch predict() for the categorical label using geometric ratios
        req = ("left_iris", "right_iris", "left_inner", "left_outer", "right_inner", "right_outer")
        if not all(k in keypoints for k in req):
            return 'Off-screen', {'left_t': -1.0, 'right_t': -1.0}
            
        from analytics import OFFSCREEN_LO, OFFSCREEN_HI, CENTER_LO, CENTER_HI, _proj_ratio
        
        left_t = _proj_ratio(keypoints['left_iris'], keypoints['left_inner'], keypoints['left_outer'])
        right_t = _proj_ratio(keypoints['right_iris'], keypoints['right_inner'], keypoints['right_outer'])
        
        hp = keypoints.get('head_pose', (0.0, 0.0, 0.0))
        if hp is not None and len(hp) >= 2:
            yaw_comp = float(np.clip(hp[1] * 0.006, -0.15, 0.15))
            pitch_comp = float(np.clip(hp[0] * 0.003, -0.08, 0.08))
            left_t += yaw_comp + pitch_comp
            right_t += yaw_comp + pitch_comp
            
        gaze_vals = {'left_t': left_t, 'right_t': right_t}
        if not (OFFSCREEN_LO <= left_t <= OFFSCREEN_HI and OFFSCREEN_LO <= right_t <= OFFSCREEN_HI):
            return 'Off-screen', gaze_vals
            
        avg = (left_t + right_t) * 0.5
        if CENTER_LO <= avg <= CENTER_HI: label = 'Center'
        elif avg < CENTER_LO: label = 'Left'
        else: label = 'Right'
        
        return label, gaze_vals
