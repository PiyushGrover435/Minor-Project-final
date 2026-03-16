"""
analytics.py

Gaze and stress math utilities.
Optimized with NumPy vector ops and small, deterministic heuristics suitable for on-device use.
"""
import numpy as np

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
