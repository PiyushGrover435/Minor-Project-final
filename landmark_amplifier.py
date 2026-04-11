"""
landmark_amplifier.py

Signal Amplification wrapper for micro-expression detection.

Magnifies subtle coordinate shifts in Region-of-Interest (ROI) facial landmarks
so that the downstream analytics and CNN can detect micro-expressions that would
otherwise be lost in low-resolution webcam noise.

Reference: MDPI 2024 — micro-expression magnification via linear amplification
of sub-pixel motion in ROI zones.

Privacy-first: all processing is in-memory NumPy vector math; no frames are saved.
"""
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
# Amplification factor applied to subtle changes
AMP_FACTOR_LO = 1.5   # lower bound of amplification range
AMP_FACTOR_HI = 2.0   # upper bound — applied to smallest motions

# Motion thresholds (normalised by inter-ocular distance)
NOISE_FLOOR   = 0.001  # below this → noise, do not amplify
SUBTLE_CEIL   = 0.01   # above this → already visible, no amplification needed

# ── ROI landmark indices (MediaPipe 478-point refined mesh) ────────────
# These are the landmarks most relevant to Action Units and micro-expressions.
ROI_INDICES = {
    # Eyebrows (AU1 inner raise, AU2 outer raise, AU4 lowerer)
    'left_brow':   [46, 53, 52, 65, 55, 107, 105],
    'right_brow':  [276, 283, 282, 295, 285, 336, 334],
    # Eyes (AU5 upper lid raise, AU6 cheek raise, AU7 lid tighten)
    'left_eye':    [33, 160, 158, 133, 153, 144, 159, 145],
    'right_eye':   [362, 385, 387, 263, 373, 380, 386, 374],
    # Mouth (AU10 upper lip, AU12 lip corner, AU15 lip depressor,
    #         AU17 chin raise, AU20 lip stretch, AU25 lips part)
    'mouth':       [61, 291, 13, 14, 78, 308, 82, 312, 87, 317,
                    0, 267, 269, 270, 37, 39, 40, 185, 409],
    # Nose bridge (AU9 nose wrinkle) — stable reference
    'nose':        [1, 6, 4, 5],
}

# Flat set of all ROI indices for fast lookup
_ROI_SET = set()
for _group in ROI_INDICES.values():
    _ROI_SET.update(_group)
ROI_FLAT = sorted(_ROI_SET)


class LandmarkAmplifier:
    """
    Amplifies subtle frame-to-frame coordinate shifts in ROI landmarks.

    Usage::

        amp = LandmarkAmplifier()
        amplified_lm = amp.amplify(lm, interocular_dist)
        # pass amplified_lm downstream instead of raw lm

    The amplification is *linear* and bounded:
      - Motions below NOISE_FLOOR (normalised) are zeroed (denoised).
      - Motions between NOISE_FLOOR and SUBTLE_CEIL are magnified by a factor
        that interpolates linearly from AMP_FACTOR_HI (tiny motions) down to
        AMP_FACTOR_LO (near the ceiling).
      - Motions above SUBTLE_CEIL pass through unmodified.
    """

    def __init__(self):
        self._prev_lm = None
        self._roi_mask = None  # lazily built boolean mask over N landmarks

    def _build_roi_mask(self, n_landmarks):
        """Build a boolean array of shape (n_landmarks,) marking ROI points."""
        mask = np.zeros(n_landmarks, dtype=bool)
        for idx in ROI_FLAT:
            if idx < n_landmarks:
                mask[idx] = True
        self._roi_mask = mask
        return mask

    def amplify(self, lm, interocular=None):
        """
        Apply linear signal amplification to ROI landmarks.

        Parameters
        ----------
        lm : np.ndarray, shape (N, 2)
            Raw pixel-coordinate landmarks from VisionEngine.
        interocular : float or None
            Inter-ocular distance for normalisation.  If None, uses
            the Euclidean distance between landmarks 133 and 362.

        Returns
        -------
        amplified : np.ndarray, shape (N, 2)
            Landmarks with subtle ROI motions magnified in-place.
        """
        n = lm.shape[0]

        # First frame: nothing to diff against
        if self._prev_lm is None or self._prev_lm.shape != lm.shape:
            self._prev_lm = lm.copy()
            return lm.copy()

        # Build mask on first call or if landmark count changed
        if self._roi_mask is None or len(self._roi_mask) != n:
            self._build_roi_mask(n)

        # Compute normalisation scale
        if interocular is None or interocular < 1.0:
            # Fallback: left_inner (133) ↔ right_inner (362)
            if n > 362:
                interocular = float(np.linalg.norm(lm[133] - lm[362]))
            if interocular is None or interocular < 1.0:
                interocular = 1.0

        # ── Per-landmark displacement ──────────────────────────────────
        delta = lm - self._prev_lm                      # (N, 2)
        dist = np.linalg.norm(delta, axis=1)             # (N,)
        norm_dist = dist / interocular                   # normalised

        # ── Amplification logic (vectorised NumPy) ─────────────────────
        amplified = lm.copy()

        # Only process ROI landmarks
        roi = self._roi_mask
        nd = norm_dist[roi]
        d  = delta[roi]

        # Mask: motions in the subtle range (above noise, below ceiling)
        subtle = (nd > NOISE_FLOOR) & (nd < SUBTLE_CEIL)

        if np.any(subtle):
            # Interpolate amplification factor: smaller motion → higher factor
            # ratio ∈ (0, 1) where 0 = at noise floor, 1 = at ceiling
            ratio = (nd[subtle] - NOISE_FLOOR) / (SUBTLE_CEIL - NOISE_FLOOR)
            factors = AMP_FACTOR_HI - ratio * (AMP_FACTOR_HI - AMP_FACTOR_LO)

            # Apply amplification to the delta, then reconstruct position
            amp_delta = d[subtle] * factors[:, np.newaxis]
            amplified_roi = self._prev_lm[roi][subtle] + amp_delta
            # Write back into the full array
            roi_indices = np.where(roi)[0]
            subtle_indices = roi_indices[subtle]
            amplified[subtle_indices] = amplified_roi

        # Denoise: zero out sub-noise-floor motions in ROI
        noise = nd <= NOISE_FLOOR
        if np.any(noise):
            roi_indices = np.where(roi)[0]
            noise_indices = roi_indices[noise]
            amplified[noise_indices] = self._prev_lm[noise_indices]

        # Update stored previous frame
        self._prev_lm = lm.copy()

        return amplified
