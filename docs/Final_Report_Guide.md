# Project Finalization Guide

This guide outlines exactly how to test your quantized `.tflite` models on an actual edge device (like a Raspberry Pi or Jetson Nano) and provides a structured outline for your Final Project Report.

---

## Part 1: Testing on an Edge Device

To truly demonstrate the value of this project, testing on an edge device (like a Raspberry Pi 4/5, Nvidia Jetson Nano, or even an older laptop acting as an edge node) is highly recommended.

### 1. Preparing the Edge Environment
Edge devices usually have limited resources. You do not need the full PyTorch heavy libraries.

1. **Install minimal dependencies:**
   ```bash
   # On your Raspberry Pi or edge device terminal:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy opencv-python mediapipe
   
   # IMPORTANT: Only install the TFLite runtime, not full TensorFlow!
   pip install tflite-runtime
   ```

### 2. What Files to Transfer
You only need a subset of the codebase on the edge device to run inference:
*   `main.py`, `vision_engine.py`, `analytics.py`, `tflite_engine.py`, `gaze_geometry.py`, `landmark_*.py`
*   The `.tflite` models inside the `models/` directory (these are tiny!).
*   `face_landmarker.task`

*(You can omit the `scripts/` directory and `.pth` files as no training happens on the edge).*

### 3. Measuring Performance (Benchmarking)
Before running the full HUD, run the benchmarking script on the edge device to gather hard data for your report:

```bash
# Transfer the benchmarking script and run it
python scripts/benchmark_tflite_vs_pytorch.py
```
> [!TIP]
> **Record these metrics for your report:**
> *   Inference Latency (ms) for the CNN and TCN.
> *   Overall FPS of the application.
> *   CPU Utilization and RAM usage (using `htop` or equivalent).

---

## Part 2: Final Report Preparation

Your final report should highlight the engineering challenges you solved, particularly around **privacy-first edge computing** and **multi-modal integrity fusion**. Here is a recommended structure tailored to your exact implementation:

### 1. Abstract & Introduction
*   **Problem Statement:** The need for non-invasive, privacy-preserving behavioral analysis for continuous authentication/integrity monitoring. Data privacy concerns with cloud-based inference.
*   **Proposed Solution:** A fully offline, edge-capable CPU-first architecture integrating gaze tracking, micro-expression analysis, and stress estimation.

### 2. System Architecture
*(You can directly reuse the Mermaid diagrams from `docs/ARCHITECTURE.md`)*
*   **MediaPipe Vision Engine:** 478-point 3D mesh extraction and 6-DOF head-pose estimation (`solvePnP`).
*   **Dual-Head Multi-Task Learning (MTL):**
    *   **Affective Head:** EmotionCNN (spatial) + TCN (temporal) for stress scoring.
    *   **Gaze Head:** Hybrid geometric + TCN model for robust gaze classification.
*   **Integrity Engine:** How the 7 biometric signals (Fixation, Saccade, Tremor, Blinks, AU spikes, Stress, Gaze) fuse into a 0-100 score. Mention the penalty caps and "death spiral" prevention logic.

### 3. Edge Optimization & Quantization (Crucial Section)
*   **The Pipeline:** Explain the conversion process: PyTorch (FP32) $\rightarrow$ ONNX $\rightarrow$ TFLite (INT8).
*   **Dynamic vs. Full-Integer Quantization:** Explain why you chose full INT8 quantization for TFLite to leverage the XNNPACK delegate for maximum CPU speed without needing a GPU.

### 4. Implementation Challenges Solved
Highlight the impressive engineering feats you tackled:
*   **Head-Pose Compensation:** Preventing false "off-screen" cheating alerts when the user merely rotates their head.
*   **Micro-tremor amplification & smoothing:** Using Kalman filters to stabilize webcam noise while amplifying subtle somatic stress markers.
*   **Point-Chasing Calibration:** Creating an in-app SGD fine-tuning loop to personalize the gaze model.

### 5. Results & Evaluation 
*   **Latency & FPS:** Present a table comparing PyTorch FP32 vs. TFLite INT8 latency. Show that INT8 made real-time processing accessible on CPUs.
*   **HUD Visuals:** Include screenshots of the UI in different scenarios.
    *   *Scenario 1:* Normal State (Integrity 100).
    *   *Scenario 2:* Stress Spike / Cheating State (Integrity dropping).

### 6. Conclusion & Future Scope
*   **Summary:** Achieved a stable, lightweight framework.
*   **Future Scope:** Moving to NPU hardware accelerators, porting the UI to a native mobile app, or adding audio-based stress analysis.

---

> [!IMPORTANT]
> **Actionable Next Steps:**
> 1. Run `python scripts/benchmark_tflite_vs_pytorch.py` on your laptop to get baseline metrics to put in the report right now.
> 2. If you have an edge device (Raspberry Pi), set it up using Step 1 above and run the same benchmark. The delta between PyTorch and TFLite will be even more dramatic on weak hardware!
