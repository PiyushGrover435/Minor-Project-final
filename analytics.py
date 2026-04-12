"""
analytics.py

Gaze and stress math utilities.
Optimized with NumPy vector ops and small, deterministic heuristics suitable for on-device use.

Emotional-distress cues: blink rate vs post-calibration baseline (1.5σ), eyebrow y-variance
(micro-tremor proxy), AU4/AU12 temporal velocity/acceleration (micro-expression onset detection),
fused with the affective CNN+TCN score.

All processing is privacy-first and performed entirely in-memory.
"""
import os
import numpy as np
from collections import deque

# Small epsilon to avoid division-by-zero
EPS = 1e-6

# ── Blink rate → distress (resting ~15–20 / min; stress often ~2×) ───
BLINK_RING_MAXLEN = 900
ASSUMED_FPS = 30.0
BLINK_BASELINE_FRAMES = 150
BLINK_SIGMA_MULT = 1.5
MIN_BLINK_BPM_SIGMA = 2.5

# ── Brow micro-tremor (y-variance, normalised by interocular scale) ─
BROW_Y_RING_MAXLEN = 24
MICRO_TREMOR_VAR_SCALE = 2.8e-4
MICRO_TREMOR_DISTRESS_THRESH = 0.62

# ── AU Temporal Dynamics (velocity / acceleration spike detection) ──
AU_HISTORY_LEN        = 20     # rolling window of AU activations
AU4_VEL_SPIKE_THRESH  = 0.08   # normalised velocity threshold for AU4 spike
AU12_VEL_SPIKE_THRESH = 0.06   # normalised velocity threshold for AU12 spike
AU_ACCEL_SPIKE_THRESH = 0.04   # acceleration threshold for micro-expression onset
SPIKE_COOLDOWN_FRAMES = 10     # min frames between consecutive spike detections

# ── Gaze thresholds ─────────────────────────────────────────────────
OFFSCREEN_LO = -0.15
OFFSCREEN_HI =  1.15
CENTER_LO    =  0.35
CENTER_HI    =  0.65

# ── Biometric Gaze Engine (Fixation & Saccade) ──────────────────────
# Fixation: consecutive frames where gaze variance stays within range
FIXATION_VARIANCE_THRESH  = 0.08    # max gaze_avg variance for fixation (lenient)
FIXATION_RING_MAXLEN      = 30      # 1-second window at 30 FPS
FIXATION_MIN_DURATION_MS  = 180.0   # below this → unstable gaze → integrity penalty
# Saccade: rapid eye jump between screen zones
SACCADE_VEL_THRESH        = 0.12    # frame-to-frame gaze_avg delta threshold
SACCADE_RING_MAXLEN       = 60      # 2-second window for saccade rate (at 30 FPS)
SACCADE_RATE_PENALTY_THRESH = 4.0   # saccades/sec above which integrity is penalised
# Integrity tuning for fixation & saccade
FIX_SHORT_PENALTY         = 0.5     # mild penalty when fixation duration < 180 ms
SACCADE_STRESS_PENALTY    = 1.5     # penalty when high saccade rate during stress spike

# ── Integrity tuning knobs ──────────────────────────────────────────
INTEGRITY_DECAY       =  0.015  # tiny continuous decay per frame
STRESS_WEIGHT         =  2.0    # penalty multiplier for stress score
OFFSCREEN_PENALTY     =  5.0    # penalty when gaze is off-screen
COMBO_PENALTY         = 12.0    # extra penalty: off-screen + high stress
RECOVERY_REWARD       =  4.0    # reward when gaze is Center + stress Low
RECOVERY_ONSCREEN     =  1.5    # smaller reward when gaze is on-screen (any stress)
INTEGRITY_EMA_ALPHA   =  0.06   # EMA smoothing for integrity score (slower = more stable)

# ── Advanced Integrity: Multi-Signal Fusion weights ─────────────────
INTEGRITY_AU_SPIKE_PENALTY     = 2.0    # penalty when AU stress spike is detected
INTEGRITY_TREMOR_PENALTY       = 1.5    # penalty for elevated micro-tremor
INTEGRITY_BLINK_DISTRESS_PENALTY = 1.0  # penalty for abnormal blink rate
INTEGRITY_FIXATION_REWARD_SCALE = 0.8   # reward scale for sustained fixation
INTEGRITY_CONFIDENCE_WINDOW    = 30     # frames to build confidence in stable state


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
#  Gaze Head  (heuristic fallback — used when ML model unavailable)
# ────────────────────────────────────────────────────────────────────

# Head-pose compensation: degrees of yaw/pitch that shift the gaze ratio
# toward center by this fraction per degree, preventing false cheating alerts.
_YAW_COMP_PER_DEG    = 0.006   # ~0.6% per degree of yaw
_PITCH_COMP_PER_DEG  = 0.003   # ~0.3% per degree of pitch
_MAX_YAW_COMP        = 0.15    # cap total yaw compensation
_MAX_PITCH_COMP      = 0.08    # cap total pitch compensation


def compute_gaze(keypoints):
    """
    Compute gaze direction label and numeric ratios, with optional
    head-pose compensation to prevent false-positive off-screen alerts
    when the user merely shifts their head.

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

    # ── Head-pose compensation (6-DOF solvePnP) ────────────────────
    # When the head rotates (yaw/pitch), the iris projection shifts
    # even when the user is still fixating on the screen.  Compensate
    # by nudging the ratio back toward 0.5 (centre) proportionally.
    hp = keypoints.get('head_pose')
    if hp is not None and len(hp) >= 2:
        pitch, yaw = float(hp[0]), float(hp[1])

        # Yaw compensation: positive yaw → head turned right → iris appears left
        yaw_comp = float(np.clip(yaw * _YAW_COMP_PER_DEG,
                                 -_MAX_YAW_COMP, _MAX_YAW_COMP))
        # Pitch compensation: positive pitch → head tilted up → iris appears higher
        pitch_comp = float(np.clip(pitch * _PITCH_COMP_PER_DEG,
                                   -_MAX_PITCH_COMP, _MAX_PITCH_COMP))

        left_t  += yaw_comp + pitch_comp
        right_t += yaw_comp + pitch_comp

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

def compute_integrity(prev_score, gaze_label, stress_level, stress_score,
                      fixation_dur_ms=300.0, saccade_rate=0.0,
                      stress_spike=False, micro_tremor=0.0,
                      blink_distress=False, au4_vel=0.0, au12_vel=0.0):
    """
    Advanced Multi-Signal Integrity Engine with adaptive penalty scaling.

    Fuses 7 independent signals into a single integrity score:
      1. Gaze direction (on-screen vs off-screen)
      2. Stress level & continuous score
      3. Fixation duration (gaze stability)
      4. Saccade rate (gaze instability)
      5. AU micro-expression spikes (temporal dynamics)
      6. Micro-tremor magnitude (somatic stress marker)
      7. Blink-rate distress (autonomic stress marker)

    Adaptive scaling: penalties intensify when multiple signals fire
    simultaneously (compounding distress), while recovery rewards
    scale with confidence (sustained calm state).

    Parameters
    ----------
    prev_score       : float | None – previous score in 0..100
    gaze_label       : str
    stress_level     : str
    stress_score     : float
    fixation_dur_ms  : float – current fixation duration in milliseconds
    saccade_rate     : float – saccades per second
    stress_spike     : bool  – micro-expression onset detected
    micro_tremor     : float – normalised micro-tremor magnitude [0, 1]
    blink_distress   : bool  – blink rate exceeds baseline by 1.5σ
    au4_vel          : float – AU4 velocity (brow lowerer)
    au12_vel         : float – AU12 velocity (lip corner puller)

    Returns
    -------
    new_score : float – smoothed integrity in [0, 100]
    """
    if prev_score is None:
        prev_score = 100.0

    score = float(prev_score)

    # ── Layer 1: Base decay ─────────────────────────────────────────
    score -= INTEGRITY_DECAY

    # ── Layer 2: Stress-driven penalty (continuous) ─────────────────
    score -= float(stress_score) * STRESS_WEIGHT

    # ── Layer 3: Gaze-driven penalties ──────────────────────────────
    if gaze_label == 'Off-screen':
        score -= OFFSCREEN_PENALTY

    # Combo penalty: off-screen AND high stress simultaneously
    if gaze_label == 'Off-screen' and stress_level == 'High':
        score -= COMBO_PENALTY

    # ── Layer 4: Biometric gaze penalties ───────────────────────────
    # Short fixation: gaze instability → penalise
    if fixation_dur_ms < FIXATION_MIN_DURATION_MS:
        # Adaptive: penalty scales with how far below the threshold we are
        fix_deficit = 1.0 - (fixation_dur_ms / max(FIXATION_MIN_DURATION_MS, 1.0))
        score -= FIX_SHORT_PENALTY * (1.0 + fix_deficit)

    # High saccade rate during stress spike → compounding penalty
    if saccade_rate > SACCADE_RATE_PENALTY_THRESH:
        saccade_excess = (saccade_rate - SACCADE_RATE_PENALTY_THRESH) / SACCADE_RATE_PENALTY_THRESH
        base_saccade_pen = SACCADE_STRESS_PENALTY * (1.0 + min(saccade_excess, 2.0))
        if stress_spike:
            base_saccade_pen *= 1.5  # compounding when spike is active
        score -= base_saccade_pen

    # ── Layer 5: AU micro-expression onset penalty ──────────────────
    if stress_spike:
        score -= INTEGRITY_AU_SPIKE_PENALTY

    # Rapid AU movement (even without full spike) contributes mild penalty
    au_velocity_mag = abs(au4_vel) + abs(au12_vel)
    if au_velocity_mag > 0.1:
        score -= min(au_velocity_mag * 3.0, 2.0)

    # ── Layer 6: Somatic / autonomic stress markers ─────────────────
    if micro_tremor > MICRO_TREMOR_DISTRESS_THRESH:
        score -= INTEGRITY_TREMOR_PENALTY * (micro_tremor / max(MICRO_TREMOR_DISTRESS_THRESH, 0.01))

    if blink_distress:
        score -= INTEGRITY_BLINK_DISTRESS_PENALTY

    # ── Layer 7: Recovery rewards ───────────────────────────────────
    # Count active distress signals for adaptive recovery
    distress_signals = sum([
        gaze_label == 'Off-screen',
        stress_level in ('Medium', 'High'),
        stress_spike,
        micro_tremor > MICRO_TREMOR_DISTRESS_THRESH,
        blink_distress,
        saccade_rate > SACCADE_RATE_PENALTY_THRESH,
    ])

    if distress_signals == 0:
        # Full recovery: no distress signals at all
        if gaze_label == 'Center':
            score += RECOVERY_REWARD
        elif gaze_label in ('Left', 'Right'):
            score += RECOVERY_ONSCREEN
    elif distress_signals <= 1:
        # Partial recovery: mostly calm
        if gaze_label in ('Center', 'Left', 'Right'):
            score += RECOVERY_ONSCREEN * 0.5

    # Sustained fixation bonus: reward stable focused gaze
    if fixation_dur_ms > 500.0 and gaze_label == 'Center':
        fix_bonus = min((fixation_dur_ms - 500.0) / 2000.0, 1.0) * INTEGRITY_FIXATION_REWARD_SCALE
        score += fix_bonus

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

    def __init__(self, window=15, calib_frames=30, gaze_seq_len=15):
        self.window       = int(window)
        self.calib_frames = int(calib_frames)

        self.stress_buf = deque(maxlen=self.window)
        self.gaze_buf   = deque(maxlen=self.window)
        self.prev_integrity = 100.0
        
        # Load ML heads (graceful fallback inside if models missing)
        try:
            import tflite_engine
            if os.path.exists("models/affective_cnn_int8.tflite") and tflite_engine.HAS_TFLITE:
                self.affective_head = tflite_engine.TFLiteAffectiveHead()
                self.gaze_head = tflite_engine.TFLiteGazeHead(seq_len=int(gaze_seq_len))
                print("[RealtimeAnalyzer] Successfully hooked High-Performance INT8 TFLite Engine.")
            else:
                self.gaze_head = GazeHead(seq_len=int(gaze_seq_len))
                self.affective_head = AffectiveHead()
        except ImportError:
            self.gaze_head = GazeHead(seq_len=int(gaze_seq_len))
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
        self._prev_eye_closed     = False

        # Blink-rate ring (1 = blink onset this frame) → estimated BPM
        self._blink_ring          = deque(maxlen=BLINK_RING_MAXLEN)
        self._blink_bpm_samples   = []
        self._blink_baseline_mu   = None
        self._blink_baseline_sig  = None
        self._blink_baseline_done = False
        self._post_calib_frames   = 0

        # Eyebrow vertical micro-movement (high-frequency variance proxy)
        self._brow_y_ring         = deque(maxlen=BROW_Y_RING_MAXLEN)

        # ── AU Temporal Dynamics (20-frame rolling buffer) ──────────
        self._au4_history  = deque(maxlen=AU_HISTORY_LEN)   # normalised AU4 activation per frame
        self._au12_history = deque(maxlen=AU_HISTORY_LEN)   # normalised AU12 activation per frame
        self._au4_vel_history  = deque(maxlen=AU_HISTORY_LEN)
        self._au12_vel_history = deque(maxlen=AU_HISTORY_LEN)
        self._spike_cooldown   = 0
        self._last_stress_spike = False

        # ── Biometric Gaze Engine (Fixation & Saccade) ─────────────
        self._gaze_avg_ring    = deque(maxlen=FIXATION_RING_MAXLEN)   # rolling gaze avg
        self._fixation_frames  = 0        # consecutive frames within variance threshold
        self._fixation_dur_ms  = 0.0      # current fixation duration in ms
        self._prev_gaze_avg    = None     # previous frame's gaze avg (for saccade vel)
        self._saccade_ring     = deque(maxlen=SACCADE_RING_MAXLEN)    # 1=saccade this frame
        self._saccade_rate     = 0.0      # saccades per second (recent window)
        self._prev_gaze_zone   = None     # previous gaze zone label

    # ── AU4 / AU12 Activation Extraction ────────────────────────────

    @staticmethod
    def _compute_au4_activation(keypoints):
        """
        AU4 (Brow Lowerer): measures inner eyebrow depression relative to
        nose bridge, normalised by inter-ocular distance.

        Higher value → brows pulled *down* more → greater AU4 activation.
        All computation is in-memory vector math.
        """
        req = ('left_inner_brow', 'right_inner_brow', 'nose_bridge',
               'left_inner', 'right_inner')
        if not all(k in keypoints for k in req):
            return None

        io = _interocular(keypoints)
        nose_y = float(np.asarray(keypoints['nose_bridge'], dtype=np.float32)[1])
        left_brow_y  = float(np.asarray(keypoints['left_inner_brow'],  dtype=np.float32)[1])
        right_brow_y = float(np.asarray(keypoints['right_inner_brow'], dtype=np.float32)[1])

        # In screen coords y increases downward; brows *lowered* = larger y = closer to nose
        # Activation = how close inner brows are to the nose bridge (normalised)
        mean_dist = ((nose_y - left_brow_y) + (nose_y - right_brow_y)) * 0.5
        return float(mean_dist / max(io, EPS))

    @staticmethod
    def _compute_au12_activation(keypoints):
        """
        AU12 (Lip Corner Puller): measures how far lip corners are
        pulled up-and-outward relative to the upper lip centre,
        normalised by inter-ocular distance.

        Higher value → more smile / lip tension → greater AU12 activation.
        All computation is in-memory vector math.
        """
        req = ('left_mouth_corner', 'right_mouth_corner', 'upper_lip_center',
               'nose_tip', 'left_inner', 'right_inner')
        if not all(k in keypoints for k in req):
            return None

        io = _interocular(keypoints)
        lmc = np.asarray(keypoints['left_mouth_corner'],  dtype=np.float32)
        rmc = np.asarray(keypoints['right_mouth_corner'], dtype=np.float32)
        ulc = np.asarray(keypoints['upper_lip_center'],   dtype=np.float32)

        # Vertical pull: lip corners rise above the upper lip centre (in screen y)
        vert_pull = ((ulc[1] - lmc[1]) + (ulc[1] - rmc[1])) * 0.5
        # Lateral spread: horizontal distance between mouth corners
        lat_spread = float(np.linalg.norm(lmc[:2] - rmc[:2]))

        activation = (vert_pull + lat_spread * 0.3) / max(io, EPS)
        return float(activation)

    # ── Velocity & Acceleration (finite differences over deque) ─────

    def _compute_au_velocity(self, history):
        """
        First-order finite difference: velocity = activation[t] - activation[t-1].
        Returns 0.0 if fewer than 2 samples.
        """
        if len(history) < 2:
            return 0.0
        return float(history[-1] - history[-2])

    def _compute_au_acceleration(self, vel_history):
        """
        Second-order finite difference: acceleration = velocity[t] - velocity[t-1].
        Returns 0.0 if fewer than 2 velocity samples.
        """
        if len(vel_history) < 2:
            return 0.0
        return float(vel_history[-1] - vel_history[-2])

    def _detect_stress_spike(self, au4_vel, au4_acc, au12_vel, au12_acc):
        """
        Detect a 'Stress Spike' — the rapid onset phase of a micro-expression.

        A spike fires when:
          • AU4 velocity exceeds threshold (brows slamming down) AND acceleration
            confirms it's an *onset* (not a sustained position), OR
          • AU12 shows rapid change (lip tension snap).

        Cooldown prevents spurious re-triggers from a single expression event.
        """
        if self._spike_cooldown > 0:
            self._spike_cooldown -= 1
            return False

        au4_spike = (
            abs(au4_vel) > AU4_VEL_SPIKE_THRESH
            and abs(au4_acc) > AU_ACCEL_SPIKE_THRESH
        )
        au12_spike = (
            abs(au12_vel) > AU12_VEL_SPIKE_THRESH
            and abs(au12_acc) > AU_ACCEL_SPIKE_THRESH
        )

        if au4_spike or au12_spike:
            self._spike_cooldown = SPIKE_COOLDOWN_FRAMES
            return True
        return False

    def _update_au_temporal(self, keypoints):
        """
        Update the AU temporal buffers and compute velocities, accelerations,
        and stress spike for the current frame.

        Returns
        -------
        au4_vel, au4_acc, au12_vel, au12_acc : float
            Current velocity and acceleration for each AU.
        stress_spike : bool
            Whether a micro-expression onset was detected this frame.
        """
        au4  = self._compute_au4_activation(keypoints)
        au12 = self._compute_au12_activation(keypoints)

        # Default values when landmarks are missing
        au4_val  = au4  if au4  is not None else (self._au4_history[-1]  if self._au4_history  else 0.0)
        au12_val = au12 if au12 is not None else (self._au12_history[-1] if self._au12_history else 0.0)

        self._au4_history.append(au4_val)
        self._au12_history.append(au12_val)

        au4_vel  = self._compute_au_velocity(self._au4_history)
        au12_vel = self._compute_au_velocity(self._au12_history)

        self._au4_vel_history.append(au4_vel)
        self._au12_vel_history.append(au12_vel)

        au4_acc  = self._compute_au_acceleration(self._au4_vel_history)
        au12_acc = self._compute_au_acceleration(self._au12_vel_history)

        spike = self._detect_stress_spike(au4_vel, au4_acc, au12_vel, au12_acc)
        self._last_stress_spike = spike

        return au4_vel, au4_acc, au12_vel, au12_acc, spike

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

    # ── Biometric Gaze: Fixation & Saccade Tracking ─────────────────

    def _update_fixation_saccade(self, gaze_vals, gaze_label):
        """
        Update fixation timer and saccade detector from current gaze ratios.

        Fixation: consecutive frames where avg gaze stays within a
        FIXATION_VARIANCE_THRESH corridor.  Duration in ms uses ASSUMED_FPS.

        Saccade: a frame where gaze_avg *velocity* exceeds SACCADE_VEL_THRESH
        AND the gaze zone has changed (Left→Right, Center→Off-screen, etc.).

        All math is NumPy scalar ops → negligible overhead.

        Returns
        -------
        fixation_dur_ms : float  – current fixation duration
        saccade_rate    : float  – saccades per second in the recent window
        """
        lt = float(gaze_vals.get('left_t', 0.5))
        rt = float(gaze_vals.get('right_t', 0.5))

        # Clamp off-screen sentinels to 0.5 so they don't blow up variance
        if lt < OFFSCREEN_LO or lt > OFFSCREEN_HI:
            lt = 0.5
        if rt < OFFSCREEN_LO or rt > OFFSCREEN_HI:
            rt = 0.5

        avg = (lt + rt) * 0.5
        self._gaze_avg_ring.append(avg)

        # ── Saccade detection ──────────────────────────────────────
        saccade_this_frame = False
        if self._prev_gaze_avg is not None:
            vel = abs(avg - self._prev_gaze_avg)
            zone_changed = (gaze_label != self._prev_gaze_zone
                            and self._prev_gaze_zone is not None)
            if vel > SACCADE_VEL_THRESH and zone_changed:
                saccade_this_frame = True
        self._prev_gaze_avg = avg
        self._prev_gaze_zone = gaze_label

        self._saccade_ring.append(1 if saccade_this_frame else 0)
        n = len(self._saccade_ring)
        if n >= 10:
            self._saccade_rate = float(sum(self._saccade_ring)) * ASSUMED_FPS / float(n)
        else:
            self._saccade_rate = 0.0

        # ── Fixation tracking ──────────────────────────────────────
        if len(self._gaze_avg_ring) >= 3:
            recent = np.asarray(self._gaze_avg_ring, dtype=np.float64)
            # Use a slightly smaller window to compute instantaneous variance
            var = float(np.var(recent[-min(len(recent), 10):])) 
            if var <= FIXATION_VARIANCE_THRESH:
                self._fixation_frames += 1
            else:
                self._fixation_frames = 0
        else:
            self._fixation_frames += 1  # not enough data yet — be generous

        current_fix_ms = float(self._fixation_frames) * (1000.0 / ASSUMED_FPS)
        
        # Keep track of the *maximum* fixation achieved recently.
        # This prevents penalizing natural eye movements (saccades) where
        # instantaneous fixation drops to 0. We slowly decay the max duration.
        if current_fix_ms > self._fixation_dur_ms:
            self._fixation_dur_ms = current_fix_ms
        else:
            # Decay max fixation by ~100ms per second so it gradually drops 
            # if the user becomes truly unstable, rather than instantly tanking.
            self._fixation_dur_ms = max(0.0, self._fixation_dur_ms - (100.0 / ASSUMED_FPS))

        return self._fixation_dur_ms, self._saccade_rate

    # ── Main update ─────────────────────────────────────────────────

    def _make_result(self, gaze_label, gaze_vals, stress_level, stress_score,
                     integrity, calibrating=False, calib_remaining=0, emotion='Neutral',
                     head_pose=(0.0, 0.0, 0.0), blink_bpm=0.0, blink_baseline_bpm=None,
                     blink_zscore=0.0, micro_tremor=0.0, emotional_distress=False,
                     valence=0.0, arousal=0.0,
                     au4_vel=0.0, au4_acc=0.0, au12_vel=0.0, au12_acc=0.0,
                     stress_spike=False,
                     fixation_dur_ms=0.0, saccade_rate=0.0):
        """Uniform result dict for all return paths."""
        return {
            'gaze_label':           gaze_label,
            'gaze_vals':            gaze_vals,
            'stress_level':         stress_level,
            'stress_score':         stress_score,
            'emotion':              emotion,
            'integrity':            integrity,
            'calibrating':          calibrating,
            'calib_remaining':      calib_remaining,
            'head_pose':            head_pose,
            'blink_bpm':            float(blink_bpm),
            'blink_baseline_bpm':   None if blink_baseline_bpm is None else float(blink_baseline_bpm),
            'blink_zscore':         float(blink_zscore),
            'micro_tremor':         float(micro_tremor),
            'emotional_distress':   bool(emotional_distress),
            'valence':              float(valence),
            'arousal':              float(arousal),
            'au4_velocity':         float(au4_vel),
            'au4_acceleration':     float(au4_acc),
            'au12_velocity':        float(au12_vel),
            'au12_acceleration':    float(au12_acc),
            'stress_spike':         bool(stress_spike),
            'fixation_dur_ms':      float(fixation_dur_ms),
            'saccade_rate':         float(saccade_rate),
        }

    def _estimate_blink_bpm(self):
        """Blinks per minute from recent blink-onset counts in the ring."""
        n = len(self._blink_ring)
        if n < 30:
            return 0.0
        s = float(sum(self._blink_ring))
        return s * 60.0 * ASSUMED_FPS / float(n)

    def _update_blink_rate_baseline(self, bpm):
        """After calibration, collect BPM samples then fix μ, σ for distress tests."""
        if self._blink_baseline_done:
            return
        self._post_calib_frames += 1
        self._blink_bpm_samples.append(bpm)
        if len(self._blink_bpm_samples) >= BLINK_BASELINE_FRAMES:
            arr = np.asarray(self._blink_bpm_samples, dtype=np.float64)
            self._blink_baseline_mu = float(np.median(arr))
            self._blink_baseline_sig = float(max(np.std(arr), MIN_BLINK_BPM_SIGMA))
            self._blink_baseline_done = True

    def _blink_rate_zscore(self, bpm):
        if not self._blink_baseline_done or self._blink_baseline_sig is None:
            return 0.0
        z = (bpm - self._blink_baseline_mu) / max(self._blink_baseline_sig, EPS)
        return float(max(0.0, z))

    def _blink_distress(self, bpm):
        if not self._blink_baseline_done or self._blink_baseline_sig is None:
            return False
        return bpm > self._blink_baseline_mu + BLINK_SIGMA_MULT * self._blink_baseline_sig

    def _micro_tremor_norm(self, keypoints):
        """Normalised variance of mean eyebrow y (screen coords / interocular)."""
        if not all(k in keypoints for k in ('left_eyebrow', 'right_eyebrow', 'left_inner', 'right_inner')):
            return 0.0
        io = _interocular(keypoints)
        if io < EPS:
            io = 1.0
        ly = float(keypoints['left_eyebrow'][1]) / io
        ry = float(keypoints['right_eyebrow'][1]) / io
        self._brow_y_ring.append(0.5 * (ly + ry))
        if len(self._brow_y_ring) < 5:
            return 0.0
        var_y = float(np.var(np.asarray(self._brow_y_ring, dtype=np.float64)))
        return float(np.clip(var_y / MICRO_TREMOR_VAR_SCALE, 0.0, 1.0))

    @staticmethod
    def _valence_arousal_proxy(emotion: str, stress_score: float, micro_tremor: float, blink_z: float):
        """Rough V–A mapping (AU / distress literature–inspired heuristic, not DISFA-trained)."""
        e = emotion.lower()
        neg = {'angry', 'disgust', 'fear', 'sad'}
        pos = {'happy'}
        if e in neg:
            v = -0.65
        elif e in pos:
            v = 0.55
        elif e == 'surprise':
            v = 0.1
        else:
            v = 0.0
        a = float(np.clip(0.55 * stress_score + 0.25 * micro_tremor + 0.2 * min(blink_z / 3.0, 1.0), 0.0, 1.0))
        return v, a

    def update(self, keypoints, frame=None):
        """
        Feed one frame's keypoints and receive back the full analytics result.

        Returns
        -------
        dict including gaze, stress, emotion, integrity, blink_bpm, micro_tremor,
        AU4/AU12 velocity & acceleration, stress_spike, emotional_distress,
        valence, arousal (V–A are heuristic proxies).
        """
        hp = keypoints.get('head_pose', (0.0, 0.0, 0.0))
        rx, ry, rz = hp

        gaze_label, gaze_vals = self.gaze_head.predict(keypoints, frame)

        # ── Biometric Gaze: fixation & saccade (every frame) ────────
        fix_dur, sacc_rate = self._update_fixation_saccade(gaze_vals, gaze_label)

        # ── AU Temporal Dynamics (run every frame, even pre-calibration) ─
        au4_vel, au4_acc, au12_vel, au12_acc, stress_spike = \
            self._update_au_temporal(keypoints)

        raw = self._raw_brow_eyelid_dist(keypoints)
        if raw is None:
            emotion, ml_level, ml_score = self.affective_head.predict(keypoints, frame)
            return self._make_result(
                gaze_label, gaze_vals, 'Low', 0.0, self.prev_integrity,
                emotion=emotion, head_pose=hp,
                au4_vel=au4_vel, au4_acc=au4_acc,
                au12_vel=au12_vel, au12_acc=au12_acc,
                stress_spike=stress_spike,
                fixation_dur_ms=fix_dur, saccade_rate=sacc_rate,
            )

        eye_open = self._eye_opening_ratio(keypoints)
        if eye_open is not None:
            self.blink_buf.append(eye_open)

        closed_now = False
        if len(self.blink_buf) >= 3:
            try:
                closed_now = float(np.median(self.blink_buf)) < self.blink_threshold
            except Exception:
                closed_now = False

        if closed_now and not self._prev_eye_closed:
            self._blink_ring.append(1)
        else:
            self._blink_ring.append(0)
        self._prev_eye_closed = closed_now

        blink_state = 1.0 if closed_now else 0.0
        micro_tremor = self._micro_tremor_norm(keypoints)
        bpm = self._estimate_blink_bpm()

        delta_brow_norm = 0.0
        if self._baseline_dist is not None:
            delta_brow_norm = self._baseline_dist - raw

        blink_z = 0.0
        blink_z_feat = 0.0
        if self._calibrated:
            self._update_blink_rate_baseline(bpm)
            blink_z = self._blink_rate_zscore(bpm)
            blink_z_feat = float(np.clip(blink_z / 3.0, 0.0, 1.0))

        # ── Temporal geometry vector (expanded with AU dynamics) ─────
        spike_feat = 1.0 if stress_spike else 0.0
        temporal = (
            float(delta_brow_norm),
            float(blink_state),
            float(micro_tremor),
            float(blink_z_feat),
            float(rx),
            float(ry),
            float(rz),
            float(au4_vel),
            float(au4_acc),
            float(au12_vel),
            float(au12_acc),
            float(spike_feat),
        )

        emotion, ml_level, ml_score = self.affective_head.predict(
            keypoints, frame, temporal_geometries=temporal
        )

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
                self._blink_baseline_done = False
                self._blink_bpm_samples = []
                self._post_calib_frames = 0

            return self._make_result(
                'Calibrating', gaze_vals, 'Low', 0.0, self.prev_integrity,
                calibrating=True, calib_remaining=remaining, emotion=emotion,
                head_pose=hp, blink_bpm=bpm, micro_tremor=micro_tremor,
                au4_vel=au4_vel, au4_acc=au4_acc,
                au12_vel=au12_vel, au12_acc=au12_acc,
                stress_spike=stress_spike,
                fixation_dur_ms=fix_dur, saccade_rate=sacc_rate,
            )

        delta = self._baseline_dist - raw
        norm = delta / max(self._baseline_dist, EPS)
        score = float(np.clip(norm / 0.6, 0.0, 1.0))

        self.stress_buf.append(score)
        smooth = float(np.mean(self.stress_buf))

        if eye_open is not None:
            try:
                if float(np.median(self.blink_buf)) < self.blink_threshold:
                    self.blink_suppress = self.blink_suppress_frames
            except Exception:
                pass

        blink_distress = self._blink_distress(bpm)
        tremor_distress = micro_tremor >= MICRO_TREMOR_DISTRESS_THRESH

        if self.blink_suppress > 0:
            self.blink_suppress -= 1
            smooth = float(np.mean(self.stress_buf)) if self.stress_buf else ml_score
            integrity = compute_integrity(
                self.prev_integrity, gaze_label, 'Low', smooth,
                fixation_dur_ms=fix_dur, saccade_rate=sacc_rate,
                stress_spike=stress_spike, micro_tremor=micro_tremor,
                blink_distress=blink_distress,
                au4_vel=au4_vel, au12_vel=au12_vel,
            )
            self.prev_integrity = integrity
            v, a = self._valence_arousal_proxy(emotion, smooth, micro_tremor, blink_z)
            distress = blink_distress or tremor_distress or stress_spike or (smooth >= 0.65)
            return self._make_result(
                gaze_label, gaze_vals, 'Low', smooth, integrity,
                emotion=emotion, head_pose=hp, blink_bpm=bpm,
                blink_baseline_bpm=self._blink_baseline_mu,
                blink_zscore=blink_z, micro_tremor=micro_tremor,
                emotional_distress=distress, valence=v, arousal=a,
                au4_vel=au4_vel, au4_acc=au4_acc,
                au12_vel=au12_vel, au12_acc=au12_acc,
                stress_spike=stress_spike,
                fixation_dur_ms=fix_dur, saccade_rate=sacc_rate,
            )

        if hasattr(self.affective_head, 'feature_extractor') and self.affective_head.feature_extractor is not None:
            smooth = ml_score
            level = ml_level
        else:
            if smooth >= 0.6:
                level = 'High'
            elif smooth >= 0.25:
                level = 'Medium'
            else:
                level = 'Low'

        # ── Stress spike boost: temporarily elevate stress on micro-expression onset
        if stress_spike:
            smooth = float(np.clip(smooth + 0.20, 0.0, 1.0))
            if level == 'Low':
                level = 'Medium'

        integrity = compute_integrity(
            self.prev_integrity, gaze_label, level, smooth,
            fixation_dur_ms=fix_dur, saccade_rate=sacc_rate,
            stress_spike=stress_spike, micro_tremor=micro_tremor,
            blink_distress=blink_distress,
            au4_vel=au4_vel, au12_vel=au12_vel,
        )
        self.prev_integrity = integrity

        v, a = self._valence_arousal_proxy(emotion, smooth, micro_tremor, blink_z)
        distress = (
            blink_distress
            or tremor_distress
            or stress_spike
            or (level == 'High')
            or (smooth >= 0.68)
            or (fix_dur < FIXATION_MIN_DURATION_MS and sacc_rate > SACCADE_RATE_PENALTY_THRESH)
        )

        return self._make_result(
            gaze_label, gaze_vals, level, smooth, integrity,
            emotion=emotion, head_pose=hp, blink_bpm=bpm,
            blink_baseline_bpm=self._blink_baseline_mu,
            blink_zscore=blink_z, micro_tremor=micro_tremor,
            emotional_distress=distress, valence=v, arousal=a,
            au4_vel=au4_vel, au4_acc=au4_acc,
            au12_vel=au12_vel, au12_acc=au12_acc,
            stress_spike=stress_spike,
            fixation_dur_ms=fix_dur, saccade_rate=sacc_rate,
        )
