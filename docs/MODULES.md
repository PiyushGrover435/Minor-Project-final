# Module Reference

API documentation for all source modules in Sentin-Edge AI.

---

## `main.py` — Application Entry Point

The primary real-time loop. Opens the webcam, processes frames through the pipeline, and renders the HUD overlay.

### Key Functions

| Function | Description |
|----------|-------------|
| `main()` | Opens `VideoCapture(0)`, initializes all engines, runs the frame loop |
| `draw_overlay(frame, kp, result, fps)` | Renders the full analytics HUD onto the frame |
| `_integrity_colour(score)` | Returns a BGR colour based on integrity health (Green/Yellow/Red) |

### Controls
- **`q`** — Quit the application
- **`c`** — Start gaze calibration phase

---

## `vision_engine.py` — Landmark Extraction

MediaPipe Face Mesh wrapper that extracts 478 3D landmarks and computes 6-DOF head pose via `solvePnP`.

### Class: `VisionEngine`

```python
engine = VisionEngine(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `process(frame)` | `dict` or `None` | Extract keypoints from a BGR frame. Returns `None` if no face detected |
| `close()` | `None` | Release MediaPipe resources |

### Keypoints Dictionary

The returned dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `landmarks` | `np.ndarray (N,2)` | All 478 landmarks in pixel coordinates |
| `left_iris` / `right_iris` | `np.ndarray (2,)` | Iris centre (mean of 4 iris landmarks) |
| `face_bbox` | `tuple (x_min,y_min,x_max,y_max)` | Bounding box of detected face |
| `left_eye_points` / `right_eye_points` | `tuple` of 6 points | Eye contour for EAR calculation |
| `left_iris_pts` / `right_iris_pts` | `tuple` of 4 points | Raw iris landmark points |
| `head_pose` | `tuple (pitch, yaw, roll)` | Head orientation in degrees |
| `head_translation` | `tuple (tx, ty, tz)` | Head position in camera space |
| `left_inner`, `right_outer`, etc. | `np.ndarray (2,)` | Named landmarks from `IDX` map |

---

## `analytics.py` — Gaze, Stress & Integrity Engine

Core analytics module containing gaze classification, stress computation, the integrity engine, and the `RealtimeAnalyzer` state manager.

### Standalone Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_gaze(keypoints)` | → `(label, ratios)` | Heuristic gaze classification with head-pose compensation |
| `compute_stress(keypoints)` | → `(level, score)` | Brow-eyelid distance stress heuristic |
| `compute_integrity(prev_score, ...)` | → `float` | Multi-signal integrity fusion (7 layers) |
| `_proj_ratio(pt, a, b)` | → `float` | Project point onto segment, returns normalised ratio |

### Class: `RealtimeAnalyzer`

```python
analyzer = RealtimeAnalyzer(window=15, calib_frames=30, gaze_seq_len=15)
result = analyzer.update(keypoints, frame)
```

The `update()` method returns a comprehensive result dictionary:

| Key | Type | Description |
|-----|------|-------------|
| `gaze_label` | `str` | 'Center', 'Left', 'Right', 'Off-screen', or 'Calibrating' |
| `gaze_vals` | `dict` | `{left_t, right_t, screen_x, screen_y, confidence}` |
| `stress_level` | `str` | 'Low', 'Medium', 'High' |
| `stress_score` | `float` | Continuous stress score in [0, 1] |
| `emotion` | `str` | Detected emotion label (e.g., 'Happy', 'Angry') |
| `integrity` | `float` | Integrity score in [0, 100] |
| `blink_bpm` | `float` | Estimated blinks per minute |
| `blink_baseline_bpm` | `float or None` | Calibrated baseline blink rate |
| `blink_zscore` | `float` | Z-score of current blink rate vs baseline |
| `micro_tremor` | `float` | Normalised eyebrow micro-tremor [0, 1] |
| `emotional_distress` | `bool` | Composite distress flag |
| `valence` / `arousal` | `float` | Heuristic V-A proxy values |
| `au4_velocity` / `au12_velocity` | `float` | Action unit temporal velocities |
| `stress_spike` | `bool` | Micro-expression onset detected |
| `fixation_dur_ms` | `float` | Current fixation duration in ms |
| `saccade_rate` | `float` | Saccades per second |

---

## `gaze_head.py` — ML Gaze Regression Head

PyTorch-based hybrid gaze model combining geometric features with a temporal convolutional network.

### Class: `HybridGazeModel(nn.Module)`

Architecture: `10-D features → MLP → TCN (dilated Conv1d) → Head-pose fusion → Sigmoid → (x, y)`

### Class: `GazeHead`

| Method | Description |
|--------|-------------|
| `predict(keypoints, frame)` | Returns `(label, gaze_vals)` with categorical label and continuous coordinates |
| `fine_tune(keypoints, frame, target_x, target_y)` | SGD step toward a known screen target (used during calibration) |
| `save_weights()` | Persist model to `models/gaze_hybrid_epoch2.pth` |

---

## `affective_head.py` — Emotion & Stress Head

Multi-modal distress detection: face crop → EmotionCNN (256-D embedding) + temporal geometry → Stress TCN → stress score.

### Class: `AffectiveHead`

| Method | Description |
|--------|-------------|
| `predict(keypoints, frame, temporal_geometries)` | Returns `(emotion, level, stress_score)` |

### Temporal Geometry Vector (12-D)

`(delta_brow, blink_state, micro_tremor, blink_z, pitch, yaw, roll, au4_vel, au4_acc, au12_vel, au12_acc, stress_spike)`

---

## `gaze_geometry.py` — Shared Gaze Utilities

Shared feature extraction and head-pose compensation used by both PyTorch and TFLite paths.

### Key Functions

| Function | Description |
|----------|-------------|
| `extract_gaze_feature_vector(keypoints)` | Weighted 10-D gaze geometry vector |
| `compensate_head_pose(left_t, right_t, head_pose)` | Yaw/pitch compensation for iris ratios |
| `raw_gaze_geometry_vector(keypoints)` | Unweighted 10-D vector |
| `eye_aspect_ratio(eye_pts)` | EAR from 6-point contour |

---

## `landmark_indices.py` — Canonical Landmark Map

Single source of truth for all MediaPipe 478-point face mesh indices used across the pipeline.

Exports: `LEFT_IRIS`, `RIGHT_IRIS`, `IDX` (dict), `LEFT_EYE_POINTS`, `RIGHT_EYE_POINTS`, AU indices, etc.

---

## `landmark_smoother.py` — Kalman Temporal Smoothing

Per-landmark 1D Kalman filter with constant-velocity model for stabilising noisy webcam coordinates.

### Class: `KeypointSmoother`

| Method | Description |
|--------|-------------|
| `smooth(keypoints)` | Returns smoothed copy of keypoints dict |
| `reset()` | Clear filters when face tracking is lost |

---

## `landmark_amplifier.py` — Signal Amplification

Magnifies subtle frame-to-frame coordinate shifts in ROI landmarks for micro-expression detection.

### Class: `LandmarkAmplifier`

| Method | Description |
|--------|-------------|
| `amplify(lm, interocular)` | Returns landmarks with subtle ROI motions magnified |

Amplification is bounded: motions below `NOISE_FLOOR` are denoised, motions in the subtle range are magnified, motions above `SUBTLE_CEIL` pass through unmodified.

---

## `calibration.py` — Gaze Calibration Engine

5-point calibration phase that fine-tunes the GazeHead TCN using known screen targets.

### Class: `CalibrationEngine`

| Method | Description |
|--------|-------------|
| `start()` | Begin the calibration sequence |
| `stop()` | End calibration and save weights |
| `update_and_draw(frame_size, kp, frame, gaze_head)` | Drive SGD + render calibration UI |

---

## `tflite_engine.py` — TFLite INT8 Inference

High-performance wrappers for running fully-quantized TFLite models as drop-in replacements for the PyTorch heads.

### Classes

| Class | Replaces | Description |
|-------|----------|-------------|
| `TFLiteInferenceEngine` | — | Core INT8 inference with automatic quantize/dequantize |
| `TFLiteAffectiveHead` | `AffectiveHead` | TFLite-based emotion + stress inference |
| `TFLiteGazeHead` | `GazeHead` | TFLite-based gaze regression |

---

## `infer_from_landmarks.py` — Offline Inference

CLI tool to run the analytics pipeline on preprocessed landmark CSVs (from `preprocess_data.py`).

```bash
python infer_from_landmarks.py --input_csv data.csv --output_jsonl results.jsonl
```
