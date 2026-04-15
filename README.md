# Sentin-Edge AI

**Privacy-Preserving Gaze and Affective State Analysis for Secure Digital Environments**

Sentin-Edge AI is a privacy-first, CPU-friendly prototype for real-time gaze, stress, and integrity estimation from webcam input. The application runs entirely on-device and keeps all processing in memory. No image or frame data is written to disk by the app.

---

## Features

- **Real-time face/iris landmark extraction** using MediaPipe Face Mesh (478 3D points with iris refinement)
- **Gaze direction classification**: Center, Left, Right, Off-screen — with head-pose compensation
- **Emotion detection**: 7-class CNN (angry, disgust, fear, happy, neutral, sad, surprise)
- **Stress estimation**: Multi-modal TCN fusing deep face embeddings with temporal geometry
- **Blink rate monitoring**: EAR-based detection with calibrated baseline and z-score distress
- **Micro-expression onset detection**: AU4/AU12 velocity + acceleration spike analysis
- **Fixation & saccade tracking**: Biometric gaze stability metrics
- **Continuous integrity scoring**: 7-signal fusion engine with adaptive penalties and rewards
- **Edge-optimized inference**: INT8 dynamic quantization (PyTorch) and full-integer TFLite
- **OpenCV HUD overlay** with live metrics, confidence heatmap, and eye mesh visualisation
- **5-point gaze calibration** for per-user fine-tuning via screen targets

---

## Project Structure

```
Sentin-Edge AI/
├── main.py                  # Application entry point and HUD overlay
├── vision_engine.py         # MediaPipe landmark extraction + 6-DOF head pose
├── analytics.py             # Gaze, stress, integrity scoring engine
├── gaze_head.py             # PyTorch hybrid gaze regression (MLP + TCN)
├── affective_head.py        # EmotionCNN + Stress TCN (multi-modal)
├── gaze_geometry.py         # Shared gaze feature extraction + head-pose compensation
├── tflite_engine.py         # TFLite INT8 drop-in replacements for PyTorch heads
├── calibration.py           # 5-point gaze calibration engine
├── landmark_indices.py      # Canonical MediaPipe landmark index map
├── landmark_smoother.py     # Per-landmark Kalman temporal filter
├── landmark_amplifier.py    # Micro-expression signal amplification
├── infer_from_landmarks.py  # Offline CSV inference CLI tool
├── face_landmarker.task     # MediaPipe Tasks API model (fallback)
├── requirements.txt         # Python dependencies
├── setup_env.ps1            # Windows PowerShell setup helper
├── models/                  # Trained model weights (see docs/MODELS.md)
├── scripts/                 # Training, quantization, and benchmark scripts
└── docs/                    # Project documentation wiki
    ├── ARCHITECTURE.md      # System architecture and design
    ├── MODULES.md           # Per-module API reference
    ├── SETUP.md             # Expanded setup and troubleshooting guide
    ├── PIPELINE.md          # Frame-by-frame pipeline walkthrough
    └── MODELS.md            # Model inventory and training guide
```

---

## Requirements

- Windows, Linux, or macOS
- Python 3.11 recommended (MediaPipe wheels available for 3.8–3.11)

---

## Quick Start

### Option 1: Automated setup (PowerShell)

```powershell
./setup_env.ps1
.\.venv\Scripts\Activate.ps1
python main.py
```

### Option 2: Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Press `q` to quit. Press `c` to start gaze calibration.

---

## Calibration

The application includes a 5-point gaze calibration system:

1. Press **`c`** during normal operation
2. Follow the pulsing red dot through 5 screen positions
3. Calibrated weights are saved locally to `models/gaze_hybrid_epoch2.pth`

> **Note:** Calibration weights are git-ignored (local to your camera setup). The app works without calibration using pre-trained weights, but accuracy improves significantly after calibrating.

For detailed calibration and troubleshooting info, see **[docs/SETUP.md](docs/SETUP.md)**.

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Architecture](docs/ARCHITECTURE.md)** | System architecture, module dependency graph, design principles |
| **[Modules](docs/MODULES.md)** | Per-module API reference for all source files |
| **[Setup Guide](docs/SETUP.md)** | Expanded setup, dependencies, and troubleshooting |
| **[Pipeline](docs/PIPELINE.md)** | Step-by-step frame processing walkthrough |
| **[Models](docs/MODELS.md)** | Model inventory, training scripts, and quantization guide |

---

## Privacy

- Processing is performed entirely in memory
- The application does not save camera frames
- No data is transmitted over the network
- If you add logging, recording, or telemetry, document it explicitly

---

## Development Notes

- This is a prototype with heuristic thresholds
- Thresholds should be calibrated for camera setup, lighting, and user variability before production use
- See [CONTRIBUTING.md](CONTRIBUTING.md) for pull request guidelines

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
