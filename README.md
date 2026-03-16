# Sentin-Edge AI

Sentin-Edge AI is a privacy-first, CPU-friendly prototype for real-time gaze, stress, and integrity estimation from webcam input.

The application runs entirely on-device and keeps processing in memory. No image or frame data is written to disk by the app.

## Features

- Real-time face/iris landmark extraction using MediaPipe (with a fallback face detector path).
- Gaze direction classification: Center, Left, Right, Off-screen.
- Lightweight stress heuristic from eyebrow and eyelid geometry.
- Continuous integrity score with smoothing.
- OpenCV overlay UI with live metrics.

## Project Structure

- `main.py`: Primary real-time app loop and overlays.
- `vision_engine.py`: Landmark extraction backend and keypoint mapping.
- `analytics.py`: Gaze, stress, and integrity scoring logic.
- `sentin_edge.py`: Minimal prototype runner.
- `setup_env.ps1`: Windows PowerShell setup helper for `.venv`.
- `requirements.txt`: Python dependencies.

## Requirements

- Windows, Linux, or macOS.
- Python 3.11 recommended.

Note: MediaPipe wheels are typically available for Python 3.8 through 3.11.

## Quick Start

### Option 1: Automated setup (PowerShell)

From the project root:

```powershell
./setup_env.ps1
```

Then activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Option 2: Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Press `q` to quit.

## Privacy

- Processing is performed in memory.
- The application does not intentionally save camera frames.
- If you add logging, recording, or telemetry, document it explicitly.

## Publishing To GitHub

After creating a new GitHub repository, from this folder run:

```powershell
git init
git add .
git commit -m "Initial commit: Sentin-Edge AI"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Development Notes

- This is a prototype with heuristic thresholds.
- Thresholds should be calibrated for camera setup, lighting, and user variability before production use.

## License

This project is licensed under the MIT License. See `LICENSE`.
