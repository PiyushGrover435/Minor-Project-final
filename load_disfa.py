"""
load_disfa.py

Iterate DISFA+ samples: cropped RGB faces, 49-point landmarks, and AU intensity files
from the local `Dataset/DISFA+` tree (FaceLandmarks + Labels).

Use this for AU-aware fine-tuning (e.g. brow furrow AU4) alongside FER pre-training.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np

try:
    import scipy.io as sio
except ImportError as e:
    sio = None


def parse_au_txt(path: Path) -> Dict[str, float]:
    """Parse `000.jpg 0` lines into { '000.jpg': intensity }."""
    out: Dict[str, float] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return out


def _session_pairs(
    disfa_root: Path,
) -> Iterable[Tuple[Path, Path]]:
    """
    Yield (face_mat_path, labels_session_dir) for sessions that exist on disk.
    """
    fl_root = disfa_root / "FaceLandmarks"
    lb_root = disfa_root / "Labels"
    if not fl_root.is_dir() or not lb_root.is_dir():
        return

    for subj_dir in sorted(fl_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        inner = subj_dir / subj_dir.name
        if not inner.is_dir():
            continue
        lbl_inner = lb_root / subj_dir.name / subj_dir.name
        if not lbl_inner.is_dir():
            continue
        for mat_path in sorted(inner.glob("*_FaceCropped.mat")):
            session_stem = mat_path.name.replace("_FaceCropped.mat", "")
            sess_lbl = lbl_inner / session_stem
            if sess_lbl.is_dir():
                yield mat_path, sess_lbl


def iter_disfa_frames(
    disfa_root: Optional[str] = None,
    au_files: Tuple[str, ...] = ("AU4.txt", "AU1.txt"),
    max_frames: Optional[int] = None,
    rgb_to_gray: bool = True,
    resize: Tuple[int, int] = (48, 48),
) -> Generator[Tuple[np.ndarray, Dict[str, float], np.ndarray, str], None, None]:
    """
    Yields (image_gray_or_rgb HxW or 48x48, au_dict, landmarks Nx2, meta_id).

    au_dict keys are AU names without '.txt' (e.g. 'AU4' -> float intensity).
    """
    if sio is None:
        raise RuntimeError("scipy is required for DISFA .mat loading (see requirements.txt).")

    root = Path(disfa_root or os.path.join("Dataset", "DISFA+"))
    n_out = 0
    try:
        import cv2
    except ImportError:
        cv2 = None

    for mat_path, lbl_dir in _session_pairs(root):
        mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        if "FaceImg_CropResize" not in mat:
            continue
        frames = mat["FaceImg_CropResize"]
        au_maps: Dict[str, Dict[str, float]] = {}
        for au_name in au_files:
            key = au_name.replace(".txt", "")
            au_maps[key] = parse_au_txt(lbl_dir / au_name)

        n = len(frames)
        for i in range(n):
            ms = frames[i]
            img = np.asarray(ms.ImgCropped)
            lnd = np.asarray(ms.LndPntCropped, dtype=np.float32)
            img_id = str(getattr(ms, "imgID", f"{i:03d}.jpg"))
            if isinstance(img_id, bytes):
                img_id = img_id.decode("utf-8", errors="replace")

            au_vals: Dict[str, float] = {}
            for k, m in au_maps.items():
                au_vals[k] = float(m.get(img_id, m.get(img_id.strip(), 0.0)))

            if rgb_to_gray and img.ndim == 3 and cv2 is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if cv2 is not None and resize is not None:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]

            meta = f"{mat_path.stem}:{img_id}"
            yield arr, au_vals, lnd, meta
            n_out += 1
            if max_frames is not None and n_out >= max_frames:
                return


def collect_session_ids(disfa_root: Optional[str] = None) -> List[str]:
    root = Path(disfa_root or os.path.join("Dataset", "DISFA+"))
    return [str(p[0]) for p in _session_pairs(root)]
