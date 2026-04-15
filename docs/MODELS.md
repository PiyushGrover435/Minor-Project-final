# Models & Training

Inventory of all model files and documentation for the training scripts.

---

## Model Inventory

All models live in the `models/` directory:

### Affective Head (Emotion + Stress)

| File | Format | Size | Description |
|------|--------|------|-------------|
| `affective_cnn.pth` | PyTorch FP32 | ~5.4 MB | EmotionCNN trained on FER-2013 (7-class emotions, 256-D embedding) |
| `affective_tcn_int8.pth` | PyTorch INT8 | ~500 KB | Stress TCN with dynamic quantization |
| `stress_tcn.pth` | PyTorch FP32 | ~500 KB | Stress TCN (full precision) |
| `affective_cnn.onnx` + `.onnx.data` | ONNX | ~5.4 MB | ONNX export of EmotionCNN |
| `affective_tcn.onnx` + `.onnx.data` | ONNX | ~500 KB | ONNX export of Stress TCN |
| `affective_cnn_int8.tflite` | TFLite INT8 | ~1.4 MB | Fully quantized EmotionCNN for edge deployment |
| `affective_tcn_int8.tflite` | TFLite INT8 | ~140 KB | Fully quantized Stress TCN for edge deployment |

### Gaze Head (Gaze Regression)

| File | Format | Size | Description |
|------|--------|------|-------------|
| `gaze_hybrid_epoch2.pth` | PyTorch FP32 | ~120 KB | Hybrid gaze model (MLP + TCN + head-pose fusion) |
| `gaze_hybrid_int8.pth` | PyTorch INT8 | ~110 KB | Dynamically quantized gaze model |
| `gaze_hybrid.onnx` + `.onnx.data` | ONNX | ~120 KB | ONNX export |
| `gaze_hybrid_int8.tflite` | TFLite INT8 | ~58 KB | Fully quantized gaze model for edge deployment |

### MediaPipe

| File | Format | Size | Description |
|------|--------|------|-------------|
| `face_landmarker.task` | MediaPipe Task | ~3.6 MB | MediaPipe Face Landmarker model (Tasks API fallback) |

---

## Git Tracking Strategy

Per `.gitignore`, only lightweight TFLite models are tracked in git:

```
# Tracked (pushed to GitHub)
models/*.tflite
face_landmarker.task

# Ignored (local only — too large for git)
models/*.pth
models/*.pt
models/*.onnx
models/*_saved_model/
```

This means after a fresh clone:
- The TFLite engine will work immediately (`.tflite` files are present)
- The PyTorch engine will use random initialization until you either:
  - Download the `.pth` files separately, or
  - Retrain using the scripts below

---

## Training Scripts

All training scripts are in the `scripts/` directory:

### Data Loading

| Script | Description |
|--------|-------------|
| `load_fer2013.py` | Load FER-2013 emotion dataset for EmotionCNN training |
| `load_mpiigaze.py` | Load MPIIGaze dataset for gaze model training |
| `load_disfa.py` | Load DISFA AU dataset for fine-tuning |
| `preprocess_data.py` | Extract MediaPipe landmarks from image datasets → CSV |

### Model Training

| Script | Description |
|--------|-------------|
| `train_affective_head.py` | Train the EmotionCNN (7-class FER-2013). Defines `EmotionCNN` class and `DEFAULT_EMBED_DIM` |
| `train_gaze_head.py` | Train the HybridGazeModel on Eye Gaze Detection + MPIIGaze data |
| `train_stress_tcn.py` | Train the MultiModalStressTCN on temporal landmark sequences |
| `train_disfa_finetune.py` | Fine-tune the EmotionCNN on DISFA AU4 annotations |

### Quantization & Export

| Script | Description |
|--------|-------------|
| `export_models.py` | Export PyTorch models to ONNX format |
| `quantize_pytorch.py` | Apply INT8 dynamic quantization to PyTorch models |
| `convert_tflite_full_integer.py` | Convert ONNX → TFLite with full-integer quantization (INT8 weights + activations) |
| `create_calibration_set.py` | Generate calibration data for TFLite quantization |

### Verification

| Script | Description |
|--------|-------------|
| `verify_quantization.py` | Compare FP32 vs INT8 vs TFLite outputs for accuracy validation |
| `benchmark_models.py` | Benchmark inference speed across model formats |
| `benchmark_tflite_vs_pytorch.py` | Detailed latency comparison: PyTorch FP32 vs INT8 vs TFLite |

---

## Training Workflow

### EmotionCNN (Affective Head)

```bash
# 1. Download FER-2013 dataset to Dataset/FER-2013/
# 2. Train the base model
python scripts/train_affective_head.py

# 3. (Optional) Fine-tune on DISFA for AU4 sensitivity
python scripts/train_disfa_finetune.py
```

### HybridGazeModel (Gaze Head)

```bash
# 1. Download Eye Gaze Detection dataset to Dataset/Eye Gaze Detection/
# 2. Preprocess landmarks
python scripts/preprocess_data.py

# 3. Train the hybrid model
python scripts/train_gaze_head.py
```

### Stress TCN

```bash
# Train on recorded landmark sequences
python scripts/train_stress_tcn.py
```

### Full Quantization Pipeline

```bash
# 1. Export to ONNX
python scripts/export_models.py

# 2. PyTorch INT8 dynamic quantization
python scripts/quantize_pytorch.py

# 3. Create calibration data for TFLite
python scripts/create_calibration_set.py

# 4. Full-integer TFLite conversion
python scripts/convert_tflite_full_integer.py

# 5. Verify accuracy
python scripts/verify_quantization.py

# 6. Benchmark
python scripts/benchmark_tflite_vs_pytorch.py
```

---

## Model Architecture Details

### EmotionCNN

```
Input: (1, 1, 48, 48) grayscale face crop
├── Conv2d(1, 32, 3) → ReLU → MaxPool
├── Conv2d(32, 64, 3) → ReLU → MaxPool
├── Conv2d(64, 128, 3) → ReLU → MaxPool
├── Flatten
├── Linear → ReLU → Dropout(0.5)     ← 256-D embedding extracted here
└── Linear(256, 7) → Softmax          ← 7-class emotion output
```

### HybridGazeModel

```
Input: (batch, 15, 10) gaze features + (batch, 3) head pose
├── MLP: Linear(10, 32) → ReLU → Linear(32, 64) → ReLU
├── TCN: Conv1d(64, 64, k=3, d=2) → ReLU → Conv1d(64, 64, k=3, d=4) → ReLU
├── Take last timestep → concat with head pose
└── Regressor: Linear(67, 32) → ReLU → Linear(32, 2) → Sigmoid → (x, y)
```

### MultiModalStressTCN

```
Input: (batch, 15, 268) [256-D CNN embed + 12-D temporal geometry]
├── TCN: Conv1d(268, 128, k=3, d=2) → ReLU → Conv1d(128, 64, k=3, d=4) → ReLU
├── Take last timestep
└── Regressor: Linear(64, 32) → ReLU → Linear(32, 1) → Sigmoid → stress ∈ [0, 1]
```
