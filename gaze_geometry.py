"""
gaze_geometry.py

Shared gaze feature extraction for GazeHead and offline training.
Uses weighted geometry: iris + inner-corner channels are emphasised vs outer-eye and EAR.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

EPS = 1e-6

# ── Head-pose compensation constants ───────────────────────────────────
# Degrees of yaw/pitch that shift the gaze ratio toward center per degree,
# preventing false off-screen alerts when the user merely shifts their head.
_YAW_COMP_PER_DEG   = 0.006   # ~0.6% per degree of yaw
_PITCH_COMP_PER_DEG = 0.003   # ~0.3% per degree of pitch
_MAX_YAW_COMP       = 0.15    # cap total yaw compensation
_MAX_PITCH_COMP     = 0.08    # cap total pitch compensation


def compensate_head_pose(left_t: float, right_t: float,
                         head_pose: tuple) -> Tuple[float, float]:
    """Apply yaw/pitch compensation to iris projection ratios.

    When the head rotates, the iris projection shifts even when the user is
    still fixating on the screen.  This nudges the ratio back toward centre
    proportionally.

    Parameters
    ----------
    left_t, right_t : float
        Raw iris projection ratios for left and right eyes.
    head_pose : tuple
        (pitch, yaw, roll) in degrees from solvePnP.

    Returns
    -------
    (left_t, right_t) : tuple[float, float]
        Compensated projection ratios.
    """
    if head_pose is not None and len(head_pose) >= 2:
        pitch, yaw = float(head_pose[0]), float(head_pose[1])
        yaw_comp = float(np.clip(yaw * _YAW_COMP_PER_DEG,
                                 -_MAX_YAW_COMP, _MAX_YAW_COMP))
        pitch_comp = float(np.clip(pitch * _PITCH_COMP_PER_DEG,
                                   -_MAX_PITCH_COMP, _MAX_PITCH_COMP))
        left_t += yaw_comp + pitch_comp
        right_t += yaw_comp + pitch_comp
    return left_t, right_t

# Per-channel multipliers on the raw 10-D vector (iris-from-inner, iris-from-outer, EARs).
DEFAULT_SPATIAL_WEIGHTS = np.array(
    [1.6, 1.6, 1.6, 1.6, 0.55, 0.55, 0.55, 0.55, 0.85, 0.85], dtype=np.float32
)
GAZE_FEATURE_DIM = 10

# Normalised screen targets for the Eye Gaze Detection dataset zone folders (3×3 grid minus centre cell).
ZONE_TARGET_XY: Dict[str, Tuple[float, float]] = {
    "TopLeft": (0.15, 0.15),
    "TopCenter": (0.5, 0.15),
    "TopRight": (0.85, 0.15),
    "MiddleLeft": (0.15, 0.5),
    "MiddleRight": (0.85, 0.5),
    "BottomLeft": (0.15, 0.85),
    "BottomCenter": (0.5, 0.85),
    "BottomRight": (0.85, 0.85),
}


def eye_aspect_ratio(eye_pts: Tuple[Any, ...]) -> float:
    """Eye aspect ratio from six contour points (same ordering as landmark_indices)."""
    p1, p2, p3, p4, p5, p6 = [np.asarray(pt, dtype=np.float32) for pt in eye_pts]
    dist_vert1 = float(np.linalg.norm(p2 - p6))
    dist_vert2 = float(np.linalg.norm(p3 - p5))
    dist_horiz = float(np.linalg.norm(p1 - p4))
    if dist_horiz < EPS:
        return 0.0
    return (dist_vert1 + dist_vert2) / (2.0 * dist_horiz)


def _norm_vec_xy(p1: np.ndarray, p0: np.ndarray, interocular: float) -> Tuple[float, float]:
    v = np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32)
    d = float(interocular) if float(interocular) > EPS else 1.0
    return float(v[0]) / d, float(v[1]) / d


def raw_gaze_geometry_vector(keypoints: dict) -> np.ndarray:
    """
    Raw (unweighted) 10-D vector:
      [L iris-inner(2), R iris-inner(2), L iris-outer(2), R iris-outer(2), L EAR, R EAR]
    """
    left_inner = np.asarray(keypoints["left_inner"], dtype=np.float32)
    right_inner = np.asarray(keypoints["right_inner"], dtype=np.float32)
    interocular = float(np.linalg.norm(left_inner - right_inner))
    if interocular < EPS:
        interocular = 1.0

    lix, liy = _norm_vec_xy(keypoints["left_iris"], left_inner, interocular)
    rix, riy = _norm_vec_xy(keypoints["right_iris"], right_inner, interocular)
    lox, loy = _norm_vec_xy(keypoints["left_iris"], keypoints["left_outer"], interocular)
    rox, roy = _norm_vec_xy(keypoints["right_iris"], keypoints["right_outer"], interocular)

    left_ear = eye_aspect_ratio(keypoints["left_eye_points"])
    right_ear = eye_aspect_ratio(keypoints["right_eye_points"])

    return np.array(
        [lix, liy, rix, riy, lox, loy, rox, roy, left_ear, right_ear],
        dtype=np.float32,
    )


def apply_spatial_weights(
    raw: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    w = DEFAULT_SPATIAL_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float32)
    if w.shape[0] != raw.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != feature dim {raw.shape[0]}")
    return (raw * w).astype(np.float32)


def extract_gaze_feature_vector(
    keypoints: dict, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    return apply_spatial_weights(raw_gaze_geometry_vector(keypoints), weights)


def _parse_tuple3_str(s: str) -> np.ndarray:
    inner = s.strip().strip("()")
    parts = [float(x.strip()) for x in inner.split(",")]
    return np.array(parts[:2], dtype=np.float32)


def keypoints_from_eye_gaze_json(data: dict) -> dict:
    """
    Build a keypoints dict compatible with extract_gaze_feature_vector from
    Eye Gaze Detection JSON (single rendered eye). A symmetric second eye is
    synthesised so inter-ocular normalisation matches the live pipeline.
    """
    margin = [_parse_tuple3_str(x) for x in data["interior_margin_2d"]]
    iris_pts = [_parse_tuple3_str(x) for x in data["iris_2d"]]

    xs = [p[0] for p in margin]
    inner = margin[int(np.argmin(xs))]
    outer = margin[int(np.argmax(xs))]
    iris_c = np.mean(np.stack(iris_pts, axis=0), axis=0)

    io = float(np.linalg.norm(outer - inner))
    if io < EPS:
        io = 1.0

    shift = np.array([io * 1.08, 0.0], dtype=np.float32)
    left_inner = inner
    left_outer = outer
    left_iris = iris_c
    right_inner = inner + shift
    vec = outer - inner
    right_outer = right_inner + vec
    right_iris = right_inner + (left_iris - left_inner)

    n = len(margin)
    idxs = [0, max(1, n // 5), max(2, 2 * n // 5), max(3, 3 * n // 5), max(4, 4 * n // 5), n - 1]
    left_eye_points = tuple(margin[i] for i in idxs)
    off = right_inner - left_inner
    right_eye_points = tuple(np.asarray(p, dtype=np.float32) + off for p in left_eye_points)

    hp_raw = data.get("head_pose", "(0,0,0)")
    if isinstance(hp_raw, str):
        m = re.findall(r"[-+]?\d*\.?\d+", hp_raw)
        pitch = float(m[0]) if len(m) > 0 else 0.0
        yaw = float(m[1]) if len(m) > 1 else 0.0
        roll = float(m[2]) if len(m) > 2 else 0.0
        head_pose = (pitch, yaw, roll)
    else:
        head_pose = (0.0, 0.0, 0.0)

    return {
        "left_inner": left_inner,
        "left_outer": left_outer,
        "left_iris": left_iris,
        "right_inner": right_inner,
        "right_outer": right_outer,
        "right_iris": right_iris,
        "left_eye_points": left_eye_points,
        "right_eye_points": right_eye_points,
        "head_pose": head_pose,
    }


def parse_look_vec_string(s: str) -> Tuple[float, float, float, float]:
    inner = s.strip().strip("()")
    parts = [float(x.strip()) for x in inner.split(",")]
    while len(parts) < 4:
        parts.append(0.0)
    return parts[0], parts[1], parts[2], parts[3]


def look_vec_to_screen_xy(
    look_vec_str: str, angular_limit: float = math.pi / 2.2
) -> Tuple[float, float]:
    """
    Map dataset look direction to normalised screen coordinates in [0, 1].
    Assumes forward-facing camera with -Z as view axis (common in Unity exports).
    """
    gx, gy, gz, _ = parse_look_vec_string(look_vec_str)
    h = math.atan2(gx, -gz + EPS)
    v = math.atan2(gy, -gz + EPS)
    lim = angular_limit
    nx = float(np.clip(h / lim * 0.5 + 0.5, 0.0, 1.0))
    ny = float(np.clip(-v / lim * 0.5 + 0.5, 0.0, 1.0))
    return nx, ny


def load_eye_gaze_json_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_eye_gaze_improvement_samples(
    base_dir: Path,
) -> Iterable[Tuple[Path, str, str, Tuple[float, float]]]:
    """
    Yields (json_path, image_path, zone_name, zone_target_xy) for the ImprovementSet layout.
    base_dir should contain ImprovementSet/{Zone}/*.jpg and Json/{id}.json.
    """
    base_dir = Path(base_dir)
    img_root = base_dir / "ImprovementSet"
    json_root = base_dir / "Json"
    if not img_root.is_dir() or not json_root.is_dir():
        return

    for zone_dir in sorted(img_root.iterdir()):
        if not zone_dir.is_dir():
            continue
        zone = zone_dir.name
        if zone not in ZONE_TARGET_XY:
            continue
        tx, ty = ZONE_TARGET_XY[zone]
        for img_path in sorted(zone_dir.glob("*.jpg")):
            jid = json_root / f"{img_path.stem}.json"
            if not jid.is_file():
                continue
            yield jid, img_path, zone, (tx, ty)


def resolve_eye_gaze_base_dir(data_root: Optional[str] = None) -> Path:
    """Default: Dataset/Eye Gaze Detection/ImprovementSet/ImprovementSet"""
    root = Path(data_root) if data_root else Path("Dataset") / "Eye Gaze Detection" / "ImprovementSet" / "ImprovementSet"
    return root
