"""
main.py

OpenCV visualisation loop that runs the VisionEngine and analytics in real-time.
Privacy-first: no frames are saved to disk; all processing is in-memory.

Press 'q' to quit.  Press 'c' to calibrate gaze.
"""
import time
import cv2
import numpy as np
from vision_engine import VisionEngine
from analytics import compute_integrity, RealtimeAnalyzer
from calibration import CalibrationEngine
from landmark_smoother import KeypointSmoother


# ── HUD Design System ──────────────────────────────────────────────
# Palette (BGR format)
_WHITE      = (240, 240, 245)
_GRAY       = (140, 140, 145)
_DIM        = (90, 90, 95)
_GREEN      = (100, 220, 100)
_YELLOW     = (60, 200, 230)
_RED        = (70, 70, 220)
_CYAN       = (220, 200, 80)
_ORANGE     = (60, 140, 240)
_IRIS_COL   = (80, 255, 120)
_EYE_COL    = (140, 100, 255)
_PANEL_BG   = (20, 20, 25)
_BAR_BG     = (45, 45, 50)


def _lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _integrity_color(score):
    """Return a BGR colour reflecting integrity health with smooth gradient."""
    if score > 80:
        return _GREEN
    if score > 55:
        return _lerp_color(_YELLOW, _GREEN, (score - 55) / 25)
    if score > 30:
        return _lerp_color(_RED, _YELLOW, (score - 30) / 25)
    return _RED


def _draw_panel(frame, x, y, w, h, alpha=0.70):
    """Draw a semi-transparent dark panel for grouping HUD elements."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), _PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_bar(frame, x, y, w, h, value, max_val=1.0,
              fill_color=_GREEN, bg_color=_BAR_BG, border=True):
    """Draw a horizontal progress/meter bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(w * max(0.0, min(1.0, value / max_val)))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_color, -1)
    if border:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 75), 1)


def _put(frame, text, x, y, scale=0.48, color=_WHITE, thickness=1):
    """Convenience text renderer with LINE_AA."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ── Overlay Layout ──────────────────────────────────────────────────

def draw_overlay(frame, kp, result, fps):
    """Render the enhanced analytics HUD with grouped panels."""
    fh, fw = frame.shape[:2]
    gaze_label   = result['gaze_label']
    gaze_vals    = result['gaze_vals']
    stress_level = result['stress_level']
    stress_score = result['stress_score']
    integrity    = result['integrity']

    # ── Auto-calibration progress ───────────────────────────────────
    if result.get('calibrating'):
        remaining = result.get('calib_remaining', 0)
        total     = 30
        progress  = max(0.0, 1.0 - remaining / max(total, 1))
        _draw_panel(frame, 5, 5, 340, 36)
        _draw_bar(frame, 10, 10, 200, 14, progress)
        _put(frame, f'Auto-calibrating... {remaining} frames', 220, 22,
             scale=0.4, color=_CYAN)
        return

    # ── Left column: Gaze & Emotion Panel ──────────────────────────
    panel_x, panel_y = 8, 8
    panel_w = 260
    _draw_panel(frame, panel_x, panel_y, panel_w, 162)

    # Section title
    _put(frame, 'GAZE', panel_x + 8, panel_y + 18, scale=0.42, color=_CYAN, thickness=1)

    # Gaze label (large)
    gaze_col = _GREEN if gaze_label == 'Center' else (_YELLOW if gaze_label in ('Left','Right') else _RED)
    _put(frame, gaze_label, panel_x + 55, panel_y + 20, scale=0.5, color=gaze_col, thickness=2)

    # Gaze ratios
    lt, rt = gaze_vals.get('left_t', 0), gaze_vals.get('right_t', 0)
    _put(frame, f'L:{lt:.2f}  R:{rt:.2f}', panel_x + 8, panel_y + 42, color=_GRAY)

    # Head Pose
    hp = result.get('head_pose', (0.0, 0.0, 0.0))
    _put(frame, f'Pose  P:{hp[0]:+.1f}  Y:{hp[1]:+.1f}  R:{hp[2]:+.1f}',
         panel_x + 8, panel_y + 60, color=_DIM, scale=0.38)

    # Divider
    cv2.line(frame, (panel_x + 8, panel_y + 68), (panel_x + panel_w - 8, panel_y + 68), (50,50,55), 1)

    # Emotion
    emotion = result.get('emotion', 'Neutral')
    _put(frame, 'EMOTION', panel_x + 8, panel_y + 86, scale=0.38, color=_CYAN)
    _put(frame, emotion, panel_x + 75, panel_y + 86, scale=0.48, color=_WHITE, thickness=1)

    # Stress bar
    _put(frame, 'STRESS', panel_x + 8, panel_y + 108, scale=0.38, color=_CYAN)
    stress_col = _GREEN if stress_level == 'Low' else (_YELLOW if stress_level == 'Medium' else _RED)
    _put(frame, f'{stress_level}', panel_x + 75, panel_y + 108, scale=0.48, color=stress_col, thickness=1)
    _draw_bar(frame, panel_x + 140, panel_y + 96, 110, 10, stress_score, fill_color=stress_col)
    _put(frame, f'{stress_score:.2f}', panel_x + 140, panel_y + 122, scale=0.35, color=_GRAY)

    # Fixation & Saccade
    fix_ms = result.get('fixation_dur_ms', 0.0)
    sacc_r = result.get('saccade_rate', 0.0)
    fix_col = _GREEN if fix_ms >= 180 else _RED
    _put(frame, f'Fix:{fix_ms:.0f}ms', panel_x + 8, panel_y + 143, scale=0.38, color=fix_col)
    _put(frame, f'Sacc:{sacc_r:.1f}/s', panel_x + 105, panel_y + 143, scale=0.38, color=_GRAY)

    # Conf readout
    conf = gaze_vals.get('confidence', 0.5)
    _put(frame, f'Conf:{conf*100:.0f}%', panel_x + 190, panel_y + 143, scale=0.38, color=_GRAY)

    # ── Left column: Biometrics Panel ──────────────────────────────
    bio_y = panel_y + 175
    _draw_panel(frame, panel_x, bio_y, panel_w, 120)

    _put(frame, 'BIOMETRICS', panel_x + 8, bio_y + 18, scale=0.38, color=_CYAN)

    bb = result.get('blink_bpm', 0.0)
    bbl = result.get('blink_baseline_bpm')
    bz = result.get('blink_zscore', 0.0)
    blink_str = f'Blink: {bb:.1f}/min  z={bz:.2f}'
    if bbl is not None:
        blink_str += f'  base~{bbl:.1f}'
    _put(frame, blink_str, panel_x + 8, bio_y + 38, scale=0.36, color=_GRAY)

    mt = result.get('micro_tremor', 0.0)
    v_val = result.get('valence', 0.0)
    a_val = result.get('arousal', 0.0)
    _put(frame, f'Tremor: {mt:.2f}', panel_x + 8, bio_y + 58, scale=0.36, color=_GRAY)
    _put(frame, f'V={v_val:+.2f}  A={a_val:.2f}', panel_x + 120, bio_y + 58, scale=0.36, color=_DIM)

    au4v = result.get('au4_velocity', 0.0)
    au12v = result.get('au12_velocity', 0.0)
    _put(frame, f'AU4v:{au4v:+.3f}  AU12v:{au12v:+.3f}', panel_x + 8, bio_y + 78, scale=0.36, color=_DIM)

    spike = result.get('stress_spike', False)
    if spike:
        _put(frame, 'STRESS SPIKE', panel_x + 8, bio_y + 100, scale=0.42, color=_RED, thickness=2)
    else:
        _put(frame, 'Spike: ---', panel_x + 8, bio_y + 100, scale=0.36, color=(70, 70, 75))

    distress = result.get('emotional_distress', False)
    d_label = 'DISTRESS' if distress else 'Stable'
    d_col = _RED if distress else _GREEN
    _put(frame, d_label, panel_x + 160, bio_y + 100, scale=0.42, color=d_col, thickness=1)

    # ── Right side: Integrity Meter ────────────────────────────────
    int_w = 160
    int_h = 100
    int_x = fw - int_w - 10
    int_y = 8
    _draw_panel(frame, int_x, int_y, int_w, int_h)

    _put(frame, 'INTEGRITY', int_x + 8, int_y + 18, scale=0.38, color=_CYAN)

    # Large score number
    score_str = f'{int(integrity)}'
    i_col = _integrity_color(integrity)
    _put(frame, score_str, int_x + 30, int_y + 62, scale=1.4, color=i_col, thickness=3)

    # Integrity bar
    _draw_bar(frame, int_x + 8, int_y + 76, int_w - 16, 12, integrity, max_val=100, fill_color=i_col)

    # ── Right side: Confidence Circle ──────────────────────────────
    conf_cx = int_x + int_w // 2
    conf_cy = int_y + int_h + 30
    conf_r = 22
    _draw_panel(frame, int_x, int_y + int_h + 4, int_w, 56)

    # Confidence arc
    conf_angle = int(360 * conf)
    cv2.ellipse(frame, (conf_cx, conf_cy), (conf_r, conf_r), -90, 0, 360, (50, 50, 55), 2, cv2.LINE_AA)
    conf_col = _GREEN if conf > 0.7 else (_YELLOW if conf > 0.4 else _RED)
    if conf_angle > 0:
        cv2.ellipse(frame, (conf_cx, conf_cy), (conf_r, conf_r), -90, 0, conf_angle, conf_col, 3, cv2.LINE_AA)
    _put(frame, f'{conf*100:.0f}%', conf_cx - 16, conf_cy + 5, scale=0.4, color=_WHITE)
    _put(frame, 'CONFIDENCE', int_x + 8, conf_cy + conf_r + 14, scale=0.32, color=_DIM)

    # ── Bottom bar: FPS + Environment ──────────────────────────────
    bar_h = 28
    bar_y = fh - bar_h
    _draw_panel(frame, 0, bar_y, fw, bar_h, alpha=0.75)

    # FPS
    _put(frame, f'FPS: {fps:.0f}', 10, bar_y + 18, scale=0.38, color=_CYAN)

    # Environment quality
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_bright = gray.mean()
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    _put(frame, f'Bright: {mean_bright:.0f}', 90, bar_y + 18, scale=0.35, color=_GRAY)
    _put(frame, f'Focus: {lap_var:.0f}', 190, bar_y + 18, scale=0.35, color=_GRAY)

    # Warnings
    if mean_bright < 40.0:
        _put(frame, 'LOW LIGHT', 290, bar_y + 18, scale=0.38, color=_RED, thickness=1)
    elif lap_var < 50.0:
        _put(frame, 'BLURRY', 290, bar_y + 18, scale=0.38, color=_RED, thickness=1)

    # Calibrate hint
    _put(frame, "[C] Calibrate  [Q] Quit", fw - 200, bar_y + 18, scale=0.35, color=_DIM)

    # ── Eye & Iris Mesh Overlays ───────────────────────────────────
    draw_keys = ('left_eye_points', 'right_eye_points', 'left_iris_pts',
                 'right_iris_pts', 'left_iris', 'right_iris')
    if all(k in kp for k in draw_keys):
        # Iris centres
        for name in ('left_iris', 'right_iris'):
            pt = tuple(kp[name].astype(int))
            cv2.circle(frame, pt, 2, _WHITE, -1, cv2.LINE_AA)

        # Eye contour polylines
        for name in ('left_eye_points', 'right_eye_points'):
            pts = np.array([pt for pt in kp[name]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, _EYE_COL, 1, cv2.LINE_AA)

        # Iris contour polylines
        for name in ('left_iris_pts', 'right_iris_pts'):
            pts = np.array([pt for pt in kp[name]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, _IRIS_COL, 1, cv2.LINE_AA)


# ── Main Loop ───────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Unable to open camera')
        return

    engine     = VisionEngine()
    analyzer   = RealtimeAnalyzer(window=12, calib_frames=30, gaze_seq_len=15)
    integrity  = analyzer.prev_integrity
    calibrator = CalibrationEngine()
    smoother   = KeypointSmoother()

    # FPS tracking
    prev_time = time.time()
    fps       = 0.0
    fps_alpha = 0.1

    try:
        first_frame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                if first_frame:
                    print('\n[ERROR] Camera opened but failed to read frames.')
                    print('        Check if another app is using the camera or Windows Privacy settings.')
                break
            first_frame = False

            # ── FPS calculation ─────────────────────────────────────
            now       = time.time()
            dt        = now - prev_time
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
                    'blink_bpm': 0.0,
                    'blink_baseline_bpm': None,
                    'blink_zscore': 0.0,
                    'micro_tremor': 0.0,
                    'emotional_distress': False,
                    'valence': 0.0,
                    'arousal': 0.35,
                    'fixation_dur_ms': 0.0,
                    'saccade_rate': 0.0,
                }
                display_frame, is_calib = calibrator.update_and_draw(
                    (frame.shape[0], frame.shape[1]), None, frame, analyzer.gaze_head)
                if not is_calib:
                    draw_overlay(display_frame, {}, result, fps)
            else:
                result    = analyzer.update(kp, frame)
                integrity = result['integrity']

                display_frame, is_calib = calibrator.update_and_draw(
                    (frame.shape[0], frame.shape[1]), kp, frame, analyzer.gaze_head)
                if not is_calib:
                    draw_overlay(display_frame, kp, result, fps)

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
