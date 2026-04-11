"""
Quick gaze diagnostic — prints raw iris ratios, head pose,
and the ML model's prediction for 60 frames then exits.
"""
import cv2
import numpy as np
from vision_engine import VisionEngine
from gaze_head import GazeHead
from analytics import _proj_ratio, OFFSCREEN_LO, OFFSCREEN_HI

engine = VisionEngine()
gaze = GazeHead(seq_len=15)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

print(f"{'frame':>5}  {'left_t':>7} {'right_t':>7}  {'avg':>6}  {'label':>10}  {'pitch':>6} {'yaw':>6} {'roll':>6}  {'scr_x':>6} {'scr_y':>6}  {'conf':>5}")
print("-" * 110)

for i in range(90):
    ret, frame = cap.read()
    if not ret:
        break
    kp = engine.process(frame)
    if kp is None:
        print(f"{i:5d}  -- no face --")
        continue

    # Raw projection ratios (before any ML or compensation)
    left_t = _proj_ratio(kp['left_iris'], kp['left_inner'], kp['left_outer'])
    right_t = _proj_ratio(kp['right_iris'], kp['right_inner'], kp['right_outer'])
    avg = (left_t + right_t) * 0.5

    hp = kp.get('head_pose', (0, 0, 0))

    # ML prediction  
    label, gv = gaze.predict(kp, frame)
    sx = gv.get('screen_x', -1)
    sy = gv.get('screen_y', -1)
    conf = gv.get('confidence', 0)

    print(f"{i:5d}  {left_t:7.3f} {right_t:7.3f}  {avg:6.3f}  {label:>10}  {hp[0]:6.1f} {hp[1]:6.1f} {hp[2]:6.1f}  {sx:6.3f} {sy:6.3f}  {conf:5.2f}")

cap.release()
engine.close()
print("\nDone. Look at left_t/right_t: they should be ~0.5 when looking at camera,")
print("<0.35 when looking left, >0.65 when looking right.")
print("If avg is always stuck near one value, the iris detection or projection is off.")
