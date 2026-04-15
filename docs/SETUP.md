# Setup Guide

Comprehensive setup instructions for Sentin-Edge AI, including troubleshooting for common issues.

---

## Requirements

| Component | Version |
|-----------|---------|
| **Python** | 3.11 recommended (MediaPipe wheels available for 3.8–3.11) |
| **OS** | Windows, Linux, or macOS |
| **Camera** | Any 720p/1080p webcam |
| **GPU** | Not required — optimised for CPU inference |

## Quick Start

### Option 1: Automated Setup (PowerShell)

```powershell
# From the project root:
.\setup_env.ps1

# Then activate and run:
.\.venv\Scripts\Activate.ps1
python main.py
```

### Option 2: Manual Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### Option 3: Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

---

## Dependencies

The full dependency list in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `mediapipe>=0.10.5` | Face mesh landmark extraction (478 3D points) |
| `opencv-python>=4.5.5` | Frame capture, image processing, HUD rendering |
| `numpy>=1.21` | Vector math for all analytics |
| `torch>=2.0.0` | PyTorch runtime for CNN/TCN inference |
| `torchvision>=0.15.0` | Image transforms for face crop preprocessing |
| `scikit-learn>=1.3.0` | Performance benchmarking utilities |
| `scipy>=1.10.0` | Scientific computing utilities |
| `joblib>=1.3.0` | Model serialization support |
| `tqdm>=4.64.0` | Progress bars for training scripts |

---

## Calibration

Sentin-Edge AI includes a 5-point gaze calibration system that fine-tunes the gaze regression model to your specific camera setup and face geometry.

### How to Calibrate

1. Start the application: `python main.py`
2. Press **`c`** to begin calibration
3. Follow the pulsing red dot through 5 screen positions (Center, TopLeft, TopRight, BottomLeft, BottomRight)
4. Each point is shown for 3 seconds — keep your gaze fixed on it
5. After completion, calibrated weights are saved to `models/gaze_hybrid_epoch2.pth`

> **Note:** Calibrated model weights (`.pth` files) are git-ignored by design.
> They are local to your machine and camera setup. After cloning the repo
> fresh, you will need to recalibrate. The application works without
> calibration using the pre-trained weights, but accuracy improves
> significantly after calibration.

---

## Troubleshooting

### Camera Access Issues

**Symptom:** `Unable to open camera` or frames fail to read.

**Causes & Fixes:**
- **Another app is using the camera** (Zoom, Teams, etc.) — close those apps first
- **Windows Privacy Settings** block Python — go to *Settings → Privacy → Camera* and ensure Python is allowed
- **Wrong camera index** — if you have multiple cameras, edit `cv2.VideoCapture(0)` in `main.py` to use index `1` or `2`

### MediaPipe Version Errors

**Symptom:** `ModuleNotFoundError` or import errors with MediaPipe.

**Fix:** Ensure you're using Python 3.11 and `mediapipe>=0.10.5`. Older versions (0.9.x) are not compatible with Python 3.11+.

### TFLite Not Available

**Symptom:** App falls back to PyTorch instead of using TFLite.

This is normal and expected. The app gracefully falls back:
1. TFLite Runtime (fastest) → 2. TensorFlow Lite → 3. PyTorch INT8 → 4. PyTorch FP32

To install TFLite for maximum performance:
```bash
pip install tflite-runtime
```

### Calibration Crash (AttributeError: 'optimizer')

This was a known bug where the INT8 quantized model path skipped optimizer initialization. It has been fixed — ensure you have the latest `gaze_head.py`.

---

## Environment Notes

- The application creates **no files** during normal operation (privacy-first design)
- Calibrated weights are saved **only** when you explicitly calibrate (`c` key)
- The `calibration_data/` directory is used by training scripts, not the live app
- All `.pth`, `.onnx`, and `_saved_model/` files are git-ignored — only `.tflite` files are tracked in git
