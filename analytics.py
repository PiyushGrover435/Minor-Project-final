"""
analytics.py

Gaze and stress math utilities.
Optimized with NumPy vector ops and small, deterministic heuristics suitable for on-device use.
"""
import numpy as np
from collections import deque

# Small epsilon to avoid division-by-zero
EPS = 1e-6

# ── Gaze thresholds ─────────────────────────────────────────────────
OFFSCREEN_LO = -0.15
OFFSCREEN_HI =  1.15
CENTER_LO    =  0.35
CENTER_HI    =  0.65

# ── Integrity tuning knobs ──────────────────────────────────────────
INTEGRITY_DECAY       =  0.05   # tiny continuous decay per frame
STRESS_WEIGHT         =  5.0    # penalty multiplier for stress score
OFFSCREEN_PENALTY     = 10.0    # penalty when gaze is off-screen
COMBO_PENALTY         = 40.0    # extra penalty: off-screen + high stress
RECOVERY_REWARD       =  2.0    # reward when gaze is Center + stress Low
INTEGRITY_EMA_ALPHA   =  0.12   # EMA smoothing for integrity score


def _proj_ratio(pt, a, b):
    """Project point *pt* onto segment a→b; returns normalised ratio t (a=0, b=1)."""
    pt = np.asarray(pt, dtype=np.float32)
    a  = np.asarray(a,  dtype=np.float32)
    b  = np.asarray(b,  dtype=np.float32)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= EPS:
        return 0.5
    return float(np.dot(pt - a, ab) / denom)


# ────────────────────────────────────────────────────────────────────
#  Gaze Head
# ────────────────────────────────────────────────────────────────────

def compute_gaze(keypoints):
    """
    Compute gaze direction label and numeric ratios.

    Returns
    -------
    label : str
        'Center' | 'Left' | 'Right' | 'Off-screen'
    ratios : dict
        {'left_t': float, 'right_t': float}
    """
    req = ('left_iris', 'right_iris', 'left_inner', 'left_outer',
           'right_inner', 'right_outer')
    if not all(k in keypoints for k in req):
        return 'Off-screen', {'left_t': -1.0, 'right_t': -1.0}

    left_t  = _proj_ratio(keypoints['left_iris'],  keypoints['left_inner'],  keypoints['left_outer'])
    right_t = _proj_ratio(keypoints['right_iris'],  keypoints['right_inner'], keypoints['right_outer'])

    # Off-screen when iris projects well beyond eye corners
    if not (OFFSCREEN_LO <= left_t <= OFFSCREEN_HI and
            OFFSCREEN_LO <= right_t <= OFFSCREEN_HI):
        return 'Off-screen', {'left_t': left_t, 'right_t': right_t}

    avg = (left_t + right_t) * 0.5
    if CENTER_LO <= avg <= CENTER_HI:
        label = 'Center'
    elif avg < CENTER_LO:
        label = 'Left'
    else:
        label = 'Right'

    return label, {'left_t': left_t, 'right_t': right_t}


# ────────────────────────────────────────────────────────────────────
#  Affective Head  (stand-alone, uncalibrated — prefer RealtimeAnalyzer)
# ────────────────────────────────────────────────────────────────────

def _interocular(keypoints):
    """Return the inter-ocular distance (inner eye corners) for normalisation."""
    left_inner  = np.asarray(keypoints['left_inner'],  dtype=np.float32)
    right_inner = np.asarray(keypoints['right_inner'], dtype=np.float32)
    d = float(np.linalg.norm(left_inner - right_inner))
    return d if d > EPS else 1.0


def _brow_eyelid_dist(keypoints, interocular):
    """Normalised mean eyebrow-to-upper-eyelid vertical distance."""
    left_eb  = np.asarray(keypoints['left_eyebrow'],      dtype=np.float32)
    right_eb = np.asarray(keypoints['right_eyebrow'],      dtype=np.float32)
    left_up  = np.asarray(keypoints['left_upper_eyelid'],  dtype=np.float32)
    right_up = np.asarray(keypoints['right_upper_eyelid'], dtype=np.float32)

    left_d  = float((left_eb[1]  - left_up[1])  / interocular)
    right_d = float((right_eb[1] - right_up[1]) / interocular)
    return (left_d + right_d) * 0.5


_STRESS_KEYS = ('left_eyebrow', 'right_eyebrow',
                'left_upper_eyelid', 'right_upper_eyelid',
                'left_inner', 'right_inner')


def compute_stress(keypoints):
    """
    Compute a simple stress indicator based on eyebrow-to-eyelid distance.

    Returns
    -------
    level : str   – 'Low' | 'Medium' | 'High'
    score : float – 0..1 (higher = more stressed)
    """
    if not all(k in keypoints for k in _STRESS_KEYS):
        return 'Low', 0.0

    interocular = _interocular(keypoints)
    avg_dist    = _brow_eyelid_dist(keypoints, interocular)

    # Lower distance → higher stress (eyebrows pulled down)
    score = float(np.clip((0.06 - avg_dist) / 0.05, 0.0, 1.0))

    if score >= 0.7:
        level = 'High'
    elif score >= 0.35:
        level = 'Medium'
    else:
        level = 'Low'
    return level, score


# ────────────────────────────────────────────────────────────────────
#  Integrity Engine
# ────────────────────────────────────────────────────────────────────

def compute_integrity(prev_score, gaze_label, stress_level, stress_score):
    """
    Real-time Integrity Score logic.

    Parameters
    ----------
    prev_score   : float | None – previous score in 0..100 (None → start at 100)
    gaze_label   : str
    stress_level : str
    stress_score : float

    Returns
    -------
    new_score : float – smoothed integrity in [0, 100]
    """
    if prev_score is None:
        prev_score = 100.0

    score = float(prev_score)
    score -= INTEGRITY_DECAY
    score -= float(stress_score) * STRESS_WEIGHT

    if gaze_label == 'Off-screen':
        score -= OFFSCREEN_PENALTY

    # Large penalty when both conditions fire simultaneously
    if gaze_label == 'Off-screen' and stress_level == 'High':
        score -= COMBO_PENALTY

    # Recovery reward: gaze on-screen and relaxed → slowly regain points
    if gaze_label == 'Center' and stress_level == 'Low':
        score += RECOVERY_REWARD

    score = max(0.0, min(100.0, score))

    # Smooth with EMA
    return float(prev_score) * (1.0 - INTEGRITY_EMA_ALPHA) + score * INTEGRITY_EMA_ALPHA


# ────────────────────────────────────────────────────────────────────
#  RealtimeAnalyzer  (temporal state manager — Layer 4)
# ────────────────────────────────────────────────────────────────────

from gaze_head import GazeHead
from affective_head import AffectiveHead

class RealtimeAnalyzer:
    """Realtime analyzer with temporal smoothing and ML heads fallback."""

    def __init__(self, window=15, calib_frames=30):
        self.window       = int(window)
        self.calib_frames = int(calib_frames)

        self.stress_buf = deque(maxlen=self.window)
        self.gaze_buf   = deque(maxlen=self.window)
        self.prev_integrity = 100.0
        
        # Load ML heads (graceful fallback inside if models missing)
        self.gaze_head = GazeHead()
        self.affective_head = AffectiveHead()

        # Calibration state
        self._calib_vals          = []
        self._calib_eye_open_vals = []
        self._baseline_dist       = None
        self._baseline_eye_open   = None
        self._calibrated          = False

        # Blink detection
        self.blink_buf            = deque(maxlen=6)
        self.blink_suppress       = 0
        self.blink_threshold      = 0.015   # fallback; updated after calibration
        self.blink_suppress_frames = 4

    # ── Helper: raw brow-eyelid distance ────────────────────────────

    def _raw_brow_eyelid_dist(self, keypoints):
        """Compute normalised brow-eyelid distance (reuses shared helpers)."""
        if not all(k in keypoints for k in _STRESS_KEYS):
            return None
        interocular = _interocular(keypoints)
        return _brow_eyelid_dist(keypoints, interocular)

    # ── Helper: eye-opening ratio ───────────────────────────────────

    @staticmethod
    def _eye_aspect_ratio(eye_pts):
        """Compute the Eye Aspect Ratio for a 6-point eye contour."""
        p1, p2, p3, p4, p5, p6 = [np.asarray(pt, dtype=np.float32) for pt in eye_pts]
        dist_vert1 = np.linalg.norm(p2 - p6)
        dist_vert2 = np.linalg.norm(p3 - p5)
        dist_horiz = np.linalg.norm(p1 - p4)
        if dist_horiz < EPS:
            return 0.0
        return float((dist_vert1 + dist_vert2) / (2.0 * dist_horiz))

    def _eye_opening_ratio(self, keypoints):
        """Estimate eye opening using formal Eye Aspect Ratio (EAR)."""
        if 'left_eye_points' not in keypoints or 'right_eye_points' not in keypoints:
            return None
        
        left_ear = self._eye_aspect_ratio(keypoints['left_eye_points'])
        right_ear = self._eye_aspect_ratio(keypoints['right_eye_points'])
        return (left_ear + right_ear) * 0.5

    # ── Main update ─────────────────────────────────────────────────

    def _make_result(self, gaze_label, gaze_vals, stress_level, stress_score,
                     integrity, calibrating=False, calib_remaining=0, emotion='Neutral',
                     head_pose=(0.0, 0.0, 0.0)):
        """Uniform result dict for all return paths."""
        return {
            'gaze_label':      gaze_label,
            'gaze_vals':       gaze_vals,
            'stress_level':    stress_level,
            'stress_score':    stress_score,
            'emotion':         emotion,
            'integrity':       integrity,
            'calibrating':     calibrating,
            'calib_remaining': calib_remaining,
            'head_pose':       head_pose,
        }

    def update(self, keypoints, frame=None):
        """
        Feed one frame's keypoints and receive back the full analytics result.

        Returns
        -------
        dict with keys: gaze_label, gaze_vals, stress_level, stress_score, emotion,
                        integrity, calibrating, calib_remaining, head_pose
        """
        hp = keypoints.get('head_pose', (0.0, 0.0, 0.0))
        
        # ML Gaze Head prediction
        gaze_label, gaze_vals = self.gaze_head.predict(keypoints, frame)
        
        raw = self._raw_brow_eyelid_dist(keypoints)
        if raw is None:
            # Fallback if no face
            emotion, ml_level, ml_score = self.affective_head.predict(keypoints, frame)
            return self._make_result(gaze_label, gaze_vals, 'Low', 0.0,
                                     self.prev_integrity, emotion=emotion, head_pose=hp)

        eye_open = self._eye_opening_ratio(keypoints)
        
        # Phase 1: Micro-tremor / blink state / posture extraction for Affective Multi-Modal TCN
        delta_brow_norm = 0.0
        # If calibrated, measure deviation from baseline as a robust micro-tremor indicator
        if hasattr(self, '_baseline_dist') and self._baseline_dist is not None:
            delta_brow_norm = self._baseline_dist - raw
            
        blink_state = 0.0
        if hasattr(self, 'blink_buf') and len(self.blink_buf) > 0:
            try:
                if float(np.median(self.blink_buf)) < self.blink_threshold:
                    blink_state = 1.0
            except Exception:
                pass
                
        rx, ry, rz = keypoints.get('head_pose', (0.0, 0.0, 0.0))
                
        # ML Affective Head prediction (Multi-Modal TCN)
        emotion, ml_level, ml_score = self.affective_head.predict(
            keypoints, frame, temporal_geometries=(delta_brow_norm, blink_state, rx, ry, rz)
        )

        # ── Calibration phase ───────────────────────────────────────
        if not self._calibrated:
            self._calib_vals.append(raw)
            if eye_open is not None:
                self._calib_eye_open_vals.append(eye_open)

            remaining = max(0, self.calib_frames - len(self._calib_vals))

            if len(self._calib_vals) >= self.calib_frames:
                self._baseline_dist = float(np.median(self._calib_vals))
                if self._baseline_dist <= EPS:
                    self._baseline_dist = 0.04

                if self._calib_eye_open_vals:
                    self._baseline_eye_open = float(np.median(self._calib_eye_open_vals))
                    self.blink_threshold = self._baseline_eye_open * 0.7

                self._calibrated = True

            return self._make_result('Calibrating', gaze_vals, 'Low', 0.0,
                                     self.prev_integrity,
                                     calibrating=True, calib_remaining=remaining, emotion=emotion,
                                     head_pose=hp)

        # ── Compute stress delta from calibrated baseline ───────────
        delta = self._baseline_dist - raw
        norm  = delta / max(self._baseline_dist, EPS)
        score = float(np.clip(norm / 0.6, 0.0, 1.0))

        # Temporal smoothing
        self.stress_buf.append(score)
        smooth = float(np.mean(self.stress_buf))

        # ── Blink detection & suppression ───────────────────────────
        if eye_open is not None:
            self.blink_buf.append(eye_open)
            try:
                if float(np.median(self.blink_buf)) < self.blink_threshold:
                    self.blink_suppress = self.blink_suppress_frames
            except Exception:
                pass

        if self.blink_suppress > 0:
            self.blink_suppress -= 1
            smooth = float(np.mean(self.stress_buf)) if self.stress_buf else ml_score
            integrity = compute_integrity(self.prev_integrity, gaze_label, 'Low', smooth)
            self.prev_integrity = integrity
            return self._make_result(gaze_label, gaze_vals, 'Low', smooth, integrity, emotion=emotion, head_pose=hp)

        # Override if ML model provides higher confidence structure
        if hasattr(self.affective_head, 'tcn') and self.affective_head.tcn is not None:
            smooth = ml_score
            level = ml_level
        else:
            # Classification based on heuristic
            if smooth >= 0.6:
                level = 'High'
            elif smooth >= 0.25:
                level = 'Medium'
            else:
                level = 'Low'

        integrity = compute_integrity(self.prev_integrity, gaze_label, level, smooth)
        self.prev_integrity = integrity

        return self._make_result(gaze_label, gaze_vals, level, smooth, integrity, emotion=emotion, head_pose=hp)
