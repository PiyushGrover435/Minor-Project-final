"""
main.py

OpenCV visualisation loop that runs the VisionEngine and analytics in real-time.
Privacy-first: no frames are saved to disk; all processing is in-memory.

Press 'q' to quit.
"""
import time
import cv2
import numpy as np
from vision_engine import VisionEngine
from analytics import compute_integrity, RealtimeAnalyzer
from calibration import CalibrationEngine
from landmark_smoother import KeypointSmoother


# ── Overlay colours ─────────────────────────────────────────────────
COL_WHITE  = (255, 255, 255)
COL_GREEN  = (0, 200, 0)
COL_YELLOW = (0, 180, 200)
COL_RED    = (0, 60, 200)
COL_CYAN   = (200, 200, 0)
COL_IRIS   = (0, 255, 0)
COL_CORNER = (255, 0, 0)


def _integrity_colour(score):
    """Return a BGR colour reflecting integrity health."""
    if score > 70:
        return COL_GREEN
    if score > 40:
        return COL_YELLOW
    return COL_RED


def draw_overlay(frame, kp, result, fps):
    """Render analytics HUD onto the frame."""
    y = 30

    def put(text, col=COL_WHITE, scale=0.7, thickness=2):
        nonlocal y
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness, cv2.LINE_AA)
        y += 26

    gaze_label   = result['gaze_label']
    gaze_vals    = result['gaze_vals']
    stress_level = result['stress_level']
    stress_score = result['stress_score']
    integrity    = result['integrity']

    # ── Calibration progress bar ────────────────────────────────────
    if result.get('calibrating'):
        remaining = result.get('calib_remaining', 0)
        total     = 30  # default calib_frames
        progress  = max(0.0, 1.0 - remaining / max(total, 1))
        bar_w     = 200
        bar_h     = 16
        x0, y0    = 10, y
        # Background
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
        # Fill
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), COL_CYAN, -1)
        # Border
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), COL_WHITE, 1)
        # Label
        cv2.putText(frame, f'Calibrating... {remaining} frames left',
                    (x0 + bar_w + 10, y0 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1, cv2.LINE_AA)
        y += bar_h + 12
        return  # skip normal HUD during calibration

    # ── Normal HUD ──────────────────────────────────────────────────
    put(f'Gaze: {gaze_label}')
    put(f"Gaze L:{gaze_vals['left_t']:.2f}  R:{gaze_vals['right_t']:.2f}")
    
    # Head Pose display (6-DOF solvePnP)
    hp = result.get('head_pose', (0.0, 0.0, 0.0))
    put(f'Pose P:{hp[0]:+.1f} Y:{hp[1]:+.1f} R:{hp[2]:+.1f}', col=COL_CYAN, scale=0.5, thickness=1)
    
    emotion = result.get('emotion', 'Neutral')
    put(f'Emotion: {emotion}')
    put(f'Stress: {stress_level} ({stress_score:.2f})')
    put(f'Integrity: {int(integrity)}', col=_integrity_colour(integrity))
    put(f'FPS: {fps:.0f}', col=COL_CYAN, scale=0.5, thickness=1)

    # ── Draw Eye & Iris Structural Geometries (Mesh Contours) ───────────
    draw_keys = ('left_eye_points', 'right_eye_points', 'left_iris_pts', 'right_iris_pts', 'left_iris', 'right_iris')
    if all(k in kp for k in draw_keys):
        # 1. Draw solid center dot for tracked gaze anchors
        for name in ('left_iris', 'right_iris'):
            pt = tuple(kp[name].astype(int))
            cv2.circle(frame, pt, 2, COL_WHITE, -1)
            
        # 2. Draw outer eye meshing (Red)
        for name in ('left_eye_points', 'right_eye_points'):
            pts = np.array([pt for pt in kp[name]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=COL_RED, thickness=1, lineType=cv2.LINE_AA)
            for pt in kp[name]:
                cv2.circle(frame, tuple(int(c) for c in pt), 1, COL_RED, -1)
                
        # 3. Draw inner Iris meshing (Green)
        for name in ('left_iris_pts', 'right_iris_pts'):
            pts = np.array([pt for pt in kp[name]], np.int32).reshape((-1, 1, 2))
            # MediaPipe's 4 iris points define a distinct circle when fitted or bounded
            cv2.polylines(frame, [pts], isClosed=True, color=COL_IRIS, thickness=1, lineType=cv2.LINE_AA)
            for pt in kp[name]:
                cv2.circle(frame, tuple(int(c) for c in pt), 1, COL_IRIS, -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Unable to open camera')
        return

    engine   = VisionEngine()
    analyzer = RealtimeAnalyzer(window=12, calib_frames=30)
    integrity = analyzer.prev_integrity
    calibrator = CalibrationEngine()
    smoother  = KeypointSmoother()

    # FPS tracking
    prev_time = time.time()
    fps       = 0.0
    fps_alpha = 0.1  # EMA smoothing for FPS display

    try:
        # Check the first frame outside the loop or carefully inside to print the error
        first_frame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                if first_frame:
                    print('\n[ERROR] Camera opened (VideoCapture(0) succeeded), but failed to read any frames.')
                    print('        This is a common Windows issue when another app (like Zoom/Teams) is using the camera,')
                    print('        or Windows Privacy settings are blocking Python from accessing the camera.')
                break
            first_frame = False

            # ── FPS calculation ─────────────────────────────────────
            now     = time.time()
            dt      = now - prev_time
            prev_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = fps * (1.0 - fps_alpha) + instant_fps * fps_alpha

            kp = engine.process(frame)

            # ── Kalman temporal smoothing ───────────────────────────
            if kp is None:
                smoother.reset()
            else:
                kp = smoother.smooth(kp)
            
            # Create a display frame to manipulate (calibration UI overlays on this)
            display_frame = frame.copy()

            if kp is None:
                # No face detected — decay integrity
                integrity = compute_integrity(integrity, 'Off-screen', 'Medium', 0.3)
                result = {
                    'gaze_label':   'Off-screen',
                    'gaze_vals':    {'left_t': -1.0, 'right_t': -1.0},
                    'stress_level': 'Medium',
                    'stress_score': 0.3,
                    'emotion':      'Neutral',
                    'integrity':    integrity,
                    'calibrating':  False,
                }
                # If calibrating, keep showing calibration UI even if face blinks out
                display_frame, is_calib = calibrator.update_and_draw((frame.shape[0], frame.shape[1]), None, frame, analyzer.gaze_head)
                if not is_calib:
                    draw_overlay(display_frame, {}, result, fps)
            else:
                # ── Run analyser ────────────────────────────────────────
                result = analyzer.update(kp, frame)
                integrity = result['integrity']

                # Update calibration and check if it took over the screen
                display_frame, is_calib = calibrator.update_and_draw((frame.shape[0], frame.shape[1]), kp, frame, analyzer.gaze_head)
                if not is_calib:
                    draw_overlay(display_frame, kp, result, fps)
                    cv2.putText(display_frame, "Press 'c' to Calibrate Gaze", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            cv2.imshow('Sentin-Edge AI', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and not calibrator.active:
                calibrator.start()

    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
