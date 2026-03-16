"""
main.py

OpenCV visualization loop that runs the VisionEngine and analytics in real-time.
Privacy-first: no frames are saved to disk; all processing is in-memory.

Press 'q' to quit.
"""
import time
import cv2
import numpy as np
from vision_engine import VisionEngine
from analytics import compute_gaze, compute_stress, compute_integrity


def draw_overlay(frame, kp, gaze_label, gaze_vals, stress_level, stress_score, integrity_score):
    h, w = frame.shape[:2]

    # Text overlay
    y = 30
    def put(s, col=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        y += 26

    put(f'Gaze: {gaze_label}')
    put(f"Gaze L:{gaze_vals['left_t']:.2f} R:{gaze_vals['right_t']:.2f}")
    put(f'Stress: {stress_level} ({stress_score:.2f})')
    # Integrity with color indicating health
    color = (0, 200, 0) if integrity_score > 70 else (0, 180, 200) if integrity_score > 40 else (0, 60, 200)
    put(f'Integrity: {int(integrity_score)}', col=color)

    # Draw iris centers and eye corners
    try:
        li = tuple(kp['left_iris'].astype(int))
        ri = tuple(kp['right_iris'].astype(int))
        cv2.circle(frame, li, 4, (0, 255, 0), -1)
        cv2.circle(frame, ri, 4, (0, 255, 0), -1)
        for name in ('left_outer', 'left_inner', 'right_outer', 'right_inner'):
            p = tuple(kp[name].astype(int))
            cv2.circle(frame, p, 2, (255, 0, 0), -1)
    except Exception:
        pass


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Unable to open camera')
        return

    engine = VisionEngine()
    integrity = 100.0
    last_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            kp = engine.process(frame)
            if kp is None:
                # No face: decay integrity slowly
                integrity = compute_integrity(integrity, 'Off-screen', 'Medium', 0.3)
                draw_overlay(frame, {}, 'Off-screen', {'left_t': -1.0, 'right_t': -1.0}, 'Medium', 0.3, integrity)
                cv2.imshow('Sentin-Edge AI', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            gaze_label, gaze_vals = compute_gaze(kp)
            stress_level, stress_score = compute_stress(kp)
            integrity = compute_integrity(integrity, gaze_label, stress_level, stress_score)

            draw_overlay(frame, kp, gaze_label, gaze_vals, stress_level, stress_score, integrity)

            cv2.imshow('Sentin-Edge AI', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # tiny sleep to ease CPU usage on some devices
            time.sleep(0.001)

    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
