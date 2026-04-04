"""
analytics.py

Gaze and stress math utilities.
Optimized with NumPy vector ops and small, deterministic heuristics suitable for on-device use.
"""
import numpy as np
from collections import deque
import math

# Small epsilon to avoid division-by-zero
EPS = 1e-6

def _proj_ratio(pt, a, b):
    """Project point pt onto segment a->b and return normalized ratio t where a corresponds to 0 and b to 1."""
    # Ensure inputs are numpy arrays (works if lists/tuples provided)
    pt = np.asarray(pt, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= EPS:
        return 0.5
    t = float(np.dot(pt - a, ab) / denom)
    return t


def compute_gaze(keypoints):
    """
    Compute gaze direction label and numeric ratios.
    Returns: (label, dict)
      label: 'Center'|'Left'|'Right'|'Off-screen'
      dict: { 'left_t': ..., 'right_t': ... }
    """
    # Defensive: ensure required keys exist
    try:
        li = keypoints['left_iris']
        ri = keypoints['right_iris']
        la = keypoints['left_inner']
        lb = keypoints['left_outer']
        ra = keypoints['right_inner']
        rb = keypoints['right_outer']
    except Exception:
        # If any key missing, treat as off-screen
        return 'Off-screen', {'left_t': -1.0, 'right_t': -1.0}

    left_t = _proj_ratio(li, la, lb)
    right_t = _proj_ratio(ri, ra, rb)

    # Off-screen detection thresholds (allow small margin)
    if left_t < -0.15 or left_t > 1.15 or right_t < -0.15 or right_t > 1.15:
        return 'Off-screen', {'left_t': left_t, 'right_t': right_t}

    # Combine
    avg = (left_t + right_t) / 2.0
    if 0.35 <= avg <= 0.65:
        label = 'Center'
    elif avg < 0.35:
        label = 'Left'
    else:
        label = 'Right'

    return label, {'left_t': left_t, 'right_t': right_t}


def compute_stress(keypoints):
    """
    Compute a simple stress indicator based on eyebrow-to-eyelid distance.
    Returns: (level, score)
      level: 'Low'|'Medium'|'High'
      score: 0..1 (higher means more stressed)
    """
    # Use vertical distances (y coordinates) normalized by interocular distance
    # Defensive fetching and array conversion
    try:
        left_eb = np.asarray(keypoints['left_eyebrow'], dtype=np.float32)
        right_eb = np.asarray(keypoints['right_eyebrow'], dtype=np.float32)
        left_up = np.asarray(keypoints['left_upper_eyelid'], dtype=np.float32)
        right_up = np.asarray(keypoints['right_upper_eyelid'], dtype=np.float32)
        left_inner = np.asarray(keypoints['left_inner'], dtype=np.float32)
        right_inner = np.asarray(keypoints['right_inner'], dtype=np.float32)
    except Exception:
        # If any data missing, return neutral/low stress
        return 'Low', 0.0

    # interocular distance (inner corners)
    interocular = float(np.linalg.norm(left_inner - right_inner))
    if interocular <= EPS:
        interocular = 1.0

    # Use vertical (y) coordinate differences
    left_dist = float((left_eb[1] - left_up[1]) / interocular)
    right_dist = float((right_eb[1] - right_up[1]) / interocular)
    avg_dist = (left_dist + right_dist) / 2.0

    # Lower distance -> higher stress (eyebrows pulled down)
    # Map distances to a 0..1 stress score (clamp)
    # typical neutral might be ~0.03-0.06 depending on camera; calibrate on-device if needed
    score = np.clip((0.06 - avg_dist) / 0.05, 0.0, 1.0)
    score = float(score)
    if score >= 0.7:
        level = 'High'
    elif score >= 0.35:
        level = 'Medium'
    else:
        level = 'Low'

    return level, score


def compute_integrity(prev_score, gaze_label, stress_level, stress_score):
    """
    Real-time Integrity Score logic.
    - prev_score: previous score in 0..100 (or None to start at 100)
    - big penalty when gaze is Off-screen and stress is High
    - small penalties otherwise
    Returns new_score (0..100)
    """
    if prev_score is None:
        prev_score = 100.0

    # Start from previous and apply small decay/penalties
    score = float(prev_score)
    score -= 0.05  # tiny continuous decay
    score -= float(stress_score) * 5.0

    if gaze_label == 'Off-screen':
        score -= 10.0

    # Big penalty when both conditions occur
    if gaze_label == 'Off-screen' and stress_level == 'High':
        score -= 40.0

    # Clip to 0..100
    score = max(0.0, min(100.0, score))

    # Smooth with a small EMA to reduce jitter
    alpha = 0.12
    new_score = float(prev_score) * (1.0 - alpha) + score * alpha
    return new_score


class RealtimeAnalyzer:
    """Realtime analyzer with simple on-device calibration and temporal smoothing.

    Usage:
      a = RealtimeAnalyzer(window=15, calib_frames=30)
      # feed keypoints each frame:
      gaze_label, gaze_vals, stress_level, stress_score, integrity = a.update(keypoints)
    """

    def __init__(self, window=15, calib_frames=30):
        self.window = int(window)
        self.calib_frames = int(calib_frames)
        self.stress_buf = deque(maxlen=self.window)
        self.gaze_buf = deque(maxlen=self.window)
        self.prev_integrity = 100.0

        # calibration state for neutral eyebrow/eyelid distance
        self._calib_vals = []
        self._baseline_dist = None
        self._calibrated = False
        # blink detection buffers and suppression
        self.blink_buf = deque(maxlen=6)
        self.blink_suppress = 0
        self.blink_threshold = 0.015  # normalized opening below which is considered blink
        self.blink_suppress_frames = 4

    def _raw_eyebrow_eyelid_dist(self, keypoints):
        try:
            left_eb = np.asarray(keypoints['left_eyebrow'], dtype=np.float32)
            right_eb = np.asarray(keypoints['right_eyebrow'], dtype=np.float32)
            left_up = np.asarray(keypoints['left_upper_eyelid'], dtype=np.float32)
            right_up = np.asarray(keypoints['right_upper_eyelid'], dtype=np.float32)
            left_inner = np.asarray(keypoints['left_inner'], dtype=np.float32)
            right_inner = np.asarray(keypoints['right_inner'], dtype=np.float32)
        except Exception:
            return None

        interocular = float(np.linalg.norm(left_inner - right_inner))
        if interocular <= EPS:
            interocular = 1.0

        left_dist = float((left_eb[1] - left_up[1]) / interocular)
        right_dist = float((right_eb[1] - right_up[1]) / interocular)
        return (left_dist + right_dist) / 2.0

    def _eye_opening_ratio(self, keypoints):
        """Estimate eye opening normalized by interocular distance using upper/lower eyelid points."""
        try:
            left_up = np.asarray(keypoints['left_upper_eyelid'], dtype=np.float32)
            left_low = np.asarray(keypoints['left_lower_eyelid'], dtype=np.float32)
            right_up = np.asarray(keypoints['right_upper_eyelid'], dtype=np.float32)
            right_low = np.asarray(keypoints['right_lower_eyelid'], dtype=np.float32)
            left_inner = np.asarray(keypoints['left_inner'], dtype=np.float32)
            right_inner = np.asarray(keypoints['right_inner'], dtype=np.float32)
        except Exception:
            return None

        interocular = float(np.linalg.norm(left_inner - right_inner))
        if interocular <= EPS:
            interocular = 1.0

        left_open = float((left_low[1] - left_up[1]) / interocular)
        right_open = float((right_low[1] - right_up[1]) / interocular)
        return (left_open + right_open) / 2.0

    def update(self, keypoints):
        # compute gaze
        gaze_label, gaze_vals = compute_gaze(keypoints)

        # compute raw distance
        raw = self._raw_eyebrow_eyelid_dist(keypoints)
        if raw is None:
            # cannot compute stress; return defaults
            return gaze_label, gaze_vals, 'Low', 0.0, self.prev_integrity

        # calibration collection
        if not self._calibrated:
            self._calib_vals.append(raw)
            if len(self._calib_vals) >= self.calib_frames:
                # use median baseline to be robust to outliers
                self._baseline_dist = float(np.median(np.array(self._calib_vals)))
                # ensure baseline isn't zero
                if self._baseline_dist <= EPS:
                    self._baseline_dist = 0.04
                self._calibrated = True
            # still calibrating
            # return a special calibrating label so caller can show UI
            remaining = max(0, self.calib_frames - len(self._calib_vals))
            return 'Calibrating', gaze_vals, 'Low', 0.0, self.prev_integrity

        # compute delta from baseline: positive delta means eyebrows moved down (stress)
        delta = self._baseline_dist - raw
        # normalize by baseline magnitude to make it scale-invariant
        norm = delta / max(self._baseline_dist, EPS)
        # map to 0..1 with a conservative scale factor
        score = float(np.clip(norm / 0.6, 0.0, 1.0))

        # temporal smoothing
        self.stress_buf.append(score)
        smooth = float(np.mean(self.stress_buf)) if len(self.stress_buf) > 0 else score

        # Blink detection: compute eye opening and track recent values
        eye_open = self._eye_opening_ratio(keypoints)
        blink_detected = False
        if eye_open is not None:
            self.blink_buf.append(eye_open)
            # consider blink if recent median is below threshold
            try:
                med = float(np.median(np.array(self.blink_buf)))
                if med < self.blink_threshold:
                    blink_detected = True
                    self.blink_suppress = self.blink_suppress_frames
            except Exception:
                pass

        # If suppression active, reduce stress influence and avoid spikes
        if self.blink_suppress > 0:
            self.blink_suppress -= 1
            # do not append this frame's score to smoothing buffer to avoid spike
            if len(self.stress_buf) > 0:
                smooth = float(np.mean(self.stress_buf))
            else:
                smooth = 0.0
            level = 'Low'
            integrity = compute_integrity(self.prev_integrity, gaze_label, level, smooth)
            self.prev_integrity = integrity
            return gaze_label, gaze_vals, level, smooth, integrity

        # thresholds for levels (adjustable)
        if smooth >= 0.6:
            level = 'High'
        elif smooth >= 0.25:
            level = 'Medium'
        else:
            level = 'Low'

        # update integrity using previous value
        integrity = compute_integrity(self.prev_integrity, gaze_label, level, smooth)
        self.prev_integrity = integrity

        return gaze_label, gaze_vals, level, smooth, integrity
