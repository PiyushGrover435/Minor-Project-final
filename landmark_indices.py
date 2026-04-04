"""
landmark_indices.py

Canonical MediaPipe Face Mesh landmark indices used across the pipeline.
Single source of truth — all modules import from here.

Reference: MediaPipe 478-point refined face mesh
  - 0–467:   face mesh
  - 468–477:  iris landmarks (refined mesh only)
"""

# ── Iris clusters ──────────────────────────────────────────────────
LEFT_IRIS  = [468, 469, 470, 471]   # 4 points around left iris
RIGHT_IRIS = [473, 474, 475, 476]   # 4 points around right iris

# ── Eye corners ────────────────────────────────────────────────────
LEFT_OUTER  = 33
LEFT_INNER  = 133
RIGHT_INNER = 362
RIGHT_OUTER = 263

# ── Eyelids ────────────────────────────────────────────────────────
LEFT_UPPER_EYELID  = 159
LEFT_LOWER_EYELID  = 145
RIGHT_UPPER_EYELID = 386
RIGHT_LOWER_EYELID = 374

# ── Full Eye Contours (for EAR) ────────────────────────────────────
# Ordered as: p1 (corner), p2, p3 (uppers), p4 (corner), p5, p6 (lowers)
LEFT_EYE_POINTS  = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_POINTS = (362, 385, 387, 263, 373, 380)

# ── Eyebrows (mid-point of brow arch) ─────────────────────────────
LEFT_EYEBROW  = 105
RIGHT_EYEBROW = 334

# ── Convenience dict for VisionEngine keypoint extraction ──────────
IDX = {
    'left_outer':          LEFT_OUTER,
    'left_inner':          LEFT_INNER,
    'right_outer':         RIGHT_OUTER,
    'right_inner':         RIGHT_INNER,
    'left_upper_eyelid':   LEFT_UPPER_EYELID,
    'left_lower_eyelid':   LEFT_LOWER_EYELID,
    'right_upper_eyelid':  RIGHT_UPPER_EYELID,
    'right_lower_eyelid':  RIGHT_LOWER_EYELID,
    'left_eyebrow':        LEFT_EYEBROW,
    'right_eyebrow':       RIGHT_EYEBROW,
}

# Total expected landmarks from refined face mesh
NUM_LANDMARKS = 478
