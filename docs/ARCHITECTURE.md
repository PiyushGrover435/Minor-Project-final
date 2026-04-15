# Architecture

Sentin-Edge AI is a **privacy-first, edge-deployed** system for real-time gaze tracking, affective state estimation, and behavioral integrity scoring. All processing runs on-device using the local CPU — no frames leave the machine.

## High-Level Architecture

```mermaid
graph TD
    subgraph Input
        CAM["Webcam (720p/1080p)"]
    end

    subgraph "VisionEngine (vision_engine.py)"
        MP["MediaPipe Face Mesh<br/>478 3D Landmarks"]
        AMP["LandmarkAmplifier<br/>Micro-expression magnification"]
        POSE["solvePnP<br/>6-DOF Head Pose"]
    end

    subgraph "Temporal Smoothing"
        KF["KeypointSmoother<br/>Per-landmark Kalman Filter"]
    end

    subgraph "Dual-Head MTL Pipeline"
        direction TB
        GH["GazeHead<br/>Hybrid TCN Regression"]
        AH["AffectiveHead<br/>EmotionCNN + Stress TCN"]
    end

    subgraph "Analytics Engine (analytics.py)"
        BLINK["Blink Detector<br/>EAR-based"]
        AU["AU4/AU12<br/>Temporal Dynamics"]
        FIX["Fixation & Saccade<br/>Biometric Gaze"]
        STRESS["Stress Fusion<br/>Multi-signal"]
    end

    subgraph "Integrity Engine"
        IE["7-Signal Fusion<br/>Adaptive Penalties & Rewards"]
    end

    subgraph "Output"
        HUD["OpenCV HUD Overlay"]
    end

    CAM --> MP
    MP --> AMP --> POSE
    POSE --> KF
    KF --> GH
    KF --> AH
    KF --> BLINK
    KF --> AU
    KF --> FIX
    GH --> STRESS
    AH --> STRESS
    BLINK --> STRESS
    AU --> STRESS
    FIX --> STRESS
    STRESS --> IE
    IE --> HUD
```

## Module Dependency Graph

```mermaid
graph LR
    main --> vision_engine
    main --> analytics
    main --> calibration
    main --> landmark_smoother

    vision_engine --> landmark_indices
    vision_engine --> landmark_amplifier

    analytics --> gaze_head
    analytics --> affective_head
    analytics --> gaze_geometry
    analytics --> tflite_engine

    gaze_head --> gaze_geometry

    affective_head --> train_affective_head["scripts/train_affective_head"]

    tflite_engine --> affective_head
    tflite_engine --> gaze_geometry

    calibration --> gaze_head
```

## Design Principles

### Privacy-First Architecture
- **No frame persistence**: All video frames are processed in-memory and immediately discarded.
- **Edge-only inference**: Models run locally via PyTorch (INT8 dynamic quantization) or TFLite (full integer quantization).
- **No network calls**: The application never transmits data externally.

### Dual-Head Multi-Task Learning
The system runs two classification heads simultaneously on every frame:

| Head | Input | Model | Output |
|------|-------|-------|--------|
| **Gaze Head** | 10-D weighted iris geometry + 6-DOF pose | MLP → TCN → Regressor | Screen (x, y) + categorical label |
| **Affective Head** | 48×48 grayscale face crop + 12-D temporal geometry | EmotionCNN (256-D embed) → Stress TCN | Emotion label + stress score (0-1) |

### Multi-Signal Integrity Engine
The Integrity Engine fuses **7 independent signals** into a single 0-100 score:

1. **Gaze direction** — on-screen vs off-screen
2. **Stress level** — continuous CNN+TCN score
3. **Fixation duration** — gaze stability indicator
4. **Saccade rate** — gaze instability indicator
5. **AU micro-expression spikes** — temporal velocity/acceleration
6. **Micro-tremor** — somatic stress marker (eyebrow Y-variance)
7. **Blink-rate distress** — autonomic stress marker vs calibrated baseline

Penalties compound when multiple signals fire simultaneously; recovery rewards scale when no distress signals are active.

## Quantization Strategy

```mermaid
graph LR
    FP32["FP32 Training<br/>(PyTorch)"]
    INT8_DYN["INT8 Dynamic<br/>(torch.ao.quantization)"]
    ONNX["ONNX Export"]
    TFLITE["TFLite Full-Integer<br/>(INT8 weights + activations)"]

    FP32 --> INT8_DYN
    FP32 --> ONNX --> TFLITE
```

- **Development**: FP32 PyTorch models for training and experimentation
- **CPU Optimized**: INT8 dynamic quantization (`.pth`) for PyTorch runtime
- **Edge Deployment**: Full-integer TFLite (`.tflite`) for maximum CPU throughput
