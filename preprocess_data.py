#!/usr/bin/env python3
"""
preprocess_data.py

Privacy-first landmark extractor for Sentin-Edge AI.

- Uses MediaPipe Face Mesh (solutions) when available.
- Loads images from a given folder (recursive).
- Extracts 478 landmarks (x,y,z) per detected face.
- Does NOT save images — only writes normalized coordinates to CSV or JSONL.
- Output record format:
    [label, x1, y1, z1, x2, y2, z2, ... x478, y478, z478]

Usage examples:
  python preprocess_data.py --dataset mpiigaze --input_dir /path/to/images \
    --output_file mpiigaze_landmarks.csv --labels_csv mpiigaze_labels.csv

  python preprocess_data.py --dataset fer2013 --input_dir /path/to/fer_images \
    --output_file fer2013_landmarks.csv --map_fer_labels

Notes:
- For FER-2013 we map emotion labels to Stress vs Neutral (using parent folder name or labels CSV).
- MPIIGaze: optional extra JSONL with isolated iris and eye-corner coords.
"""
import os
import argparse
import csv
import json
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    # Fallback: simple passthrough if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable
import importlib

# Try import MediaPipe solutions (preferred). If not available, attempt Tasks API.
MP_FACE = None
MP_TASKS = None
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        MP_FACE = mp.solutions.face_mesh
except Exception:
    MP_FACE = None

# Try Tasks API
try:
    from mediapipe.tasks.python import vision as mp_tasks_vision
    MP_TASKS = mp_tasks_vision
except Exception:
    MP_TASKS = None


# Indices for iris and common eye corners used by MediaPipe refined mesh
_IRIS_RANGE = list(range(468, 478))  # 468..477
_LEFT_INNER = 133
_LEFT_OUTER = 33
_RIGHT_INNER = 362
_RIGHT_OUTER = 263


def read_labels_csv(path: str) -> Dict[str, str]:
    """Read a CSV with two columns: filename,label (header optional)."""
    mapping = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
        for r in rows:
            if not r:
                continue
            if len(r) >= 2:
                fname = os.path.basename(r[0]).strip()
                label = r[1].strip()
                mapping[fname] = label
    return mapping


def map_fer_label_to_binary(fer_label: str) -> str:
    """
    Map FER-2013 label names or indices to 'Stress' or 'Neutral'.

    FER label mapping (common):
      0: anger
      1: disgust
      2: fear
      3: happy
      4: sad
      5: surprise
      6: neutral

    We'll treat: anger, disgust, fear, sad -> Stress
                  neutral, happy, surprise -> Neutral
    """
    try:
        idx = int(fer_label)
        if idx in (0, 1, 2, 4):
            return "Stress"
        else:
            return "Neutral"
    except Exception:
        s = fer_label.strip().lower()
        if s in ("anger", "angry", "disgust", "fear", "sad"):
            return "Stress"
        return "Neutral"


def extract_face_landmarks(face_landmarks, image_w: int, image_h: int) -> List[Tuple[float, float, float]]:
    """
    Convert mediapipe face_landmarks to a list of (x,y,z) normalized coordinates.
    - x,y: normalized in [0,1] relative to image width/height (MediaPipe gives this already)
    - z: normalized by interocular distance to make it scale-invariant when possible.
    """
    lm = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
    try:
        left_inner = lm[_LEFT_INNER, :2]
        right_inner = lm[_RIGHT_INNER, :2]
        interocular = np.linalg.norm((left_inner - right_inner))
        if interocular <= 1e-6:
            interocular = 1.0
    except Exception:
        interocular = 1.0

    z_norm = lm[:, 2] / interocular
    coords = [(float(lm[i, 0]), float(lm[i, 1]), float(z_norm[i])) for i in range(len(lm))]
    return coords


def process_image(image_path: str, face_mesh, convert_to_rgb=True) -> Optional[List[Tuple[float, float, float]]]:
    """Load image, run FaceMesh, return landmarks list (478 x (x,y,z)) or None if no face."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if convert_to_rgb:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    results = face_mesh.process(img_rgb)
    if not results or not getattr(results, "multi_face_landmarks", None):
        return None

    face = results.multi_face_landmarks[0]
    coords = extract_face_landmarks(face.landmark, w, h)
    return coords


def process_image_tasks(image_path: str, face_landmarker, convert_to_rgb=True) -> Optional[List[Tuple[float, float, float]]]:
    """Process a single image using MediaPipe Tasks FaceLandmarker."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if convert_to_rgb else img

    try:
        mp_image = MP_TASKS.Image.create_from_array(img_rgb)
        res = face_landmarker.detect(mp_image)
    except Exception:
        # Try alternate API name
        try:
            mp_image = MP_TASKS.TensorImage.create_from_array(img_rgb)
            res = face_landmarker.detect(mp_image)
        except Exception:
            return None

    faces = getattr(res, "face_landmarks", None) or getattr(res, "faces", None) or getattr(res, "detections", None)
    if not faces:
        return None

    # Attempt to find a LandmarkList inside the returned face object
    first = faces[0]
    lm_list = getattr(first, "landmark", None) or getattr(first, "landmarks", None) or getattr(first, "face_landmarks", None)
    if not lm_list:
        return None

    coords = extract_face_landmarks(lm_list, w, h)
    return coords


def process_image_haar(image_path: str) -> Optional[List[Tuple[float, float, float]]]:
    """Approximate 478 landmarks by detecting face bbox with Haar cascade and filling a grid inside it."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(casc_path):
        return None
    face_cascade = cv2.CascadeClassifier(casc_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, fw, fh = faces[0]

    # Create grid of points within bbox
    import math

    n_points = 478
    side = math.ceil(math.sqrt(n_points))
    xs = [x + (i + 0.5) * fw / side for i in range(side)]
    ys = [y + (j + 0.5) * fh / side for j in range(side)]
    pts = []
    for yy in ys:
        for xx in xs:
            nx = float(xx / w)
            ny = float(yy / h)
            pts.append((nx, ny, 0.0))
            if len(pts) >= n_points:
                break
        if len(pts) >= n_points:
            break
    # pad if needed
    if len(pts) < n_points:
        while len(pts) < n_points:
            pts.append((0.0, 0.0, 0.0))
    return pts


def write_csv_header_and_rows(csv_path: str, rows: List[List]):
    """Write rows (list of lists) to CSV (overwrite)."""
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for r in rows:
            writer.writerow(r)


def process_dataset(
    dataset: str,
    input_dir: str,
    output_file: str,
    labels_csv: Optional[str] = None,
    map_fer_labels: bool = False,
    extra_mpiigaze_json: Optional[str] = None,
    task_model: Optional[str] = None,
):
    """
    Main processing function.
    """
    labels_map = {}
    if labels_csv:
        labels_map = read_labels_csv(labels_csv)

    rows_out = []
    extra_mpiigaze = []

    img_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_files.append(os.path.join(root, f))
    img_files.sort()

    # Choose engine: solutions (MP_FACE) preferred, then Tasks (MP_TASKS with model), else error
    engine_mode = None
    face_mesh = None
    face_landmarker = None
    if MP_FACE is not None:
        engine_mode = "solutions"
        face_mesh_ctx = MP_FACE.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        face_mesh = face_mesh_ctx
    elif MP_TASKS is not None:
        engine_mode = "tasks"
        # If no task_model provided, fall back to Haar (Tasks available but no model supplied)
        if not task_model:
            print("Note: mediapipe Tasks available but no --task_model provided. Falling back to Haar-cascade approximate landmarks.")
            face_landmarker = None
            engine_mode = "haar_fallback"
        else:
            # Create FaceLandmarker via Tasks API with robust import handling
            def _create_face_landmarker(model_path: str):
                # Try multiple construction patterns observed across mediapipe builds.
                FL = getattr(MP_TASKS, "FaceLandmarker", None)
                FOpts = getattr(MP_TASKS, "FaceLandmarkerOptions", None)
                BaseOpts = getattr(MP_TASKS, "BaseOptions", None)
                RunningMode = getattr(MP_TASKS, "RunningMode", None)

                # Try to import BaseOptions from core.base_options if not present
                if BaseOpts is None:
                    try:
                        base_mod = importlib.import_module("mediapipe.tasks.python.core.base_options")
                        BaseOpts = getattr(base_mod, "BaseOptions", None)
                    except Exception:
                        try:
                            core_mod = importlib.import_module("mediapipe.tasks.python.core")
                            BaseOpts = getattr(core_mod, "BaseOptions", None)
                        except Exception:
                            BaseOpts = None

                # Pattern A: BaseOptions + FaceLandmarkerOptions + create_from_options
                try:
                    if BaseOpts and FOpts and FL and hasattr(FL, "create_from_options"):
                        base = BaseOpts(model_asset_path=model_path)
                        if RunningMode:
                            opts = FOpts(base_options=base, running_mode=RunningMode.IMAGE)
                        else:
                            opts = FOpts(base_options=base)
                        return FL.create_from_options(opts)
                except Exception:
                    pass

                # Pattern B: FaceLandmarkerOptions may accept model_asset_path directly
                try:
                    if FOpts and FL:
                        try:
                            if RunningMode:
                                opts = FOpts(model_asset_path=model_path, running_mode=RunningMode.IMAGE)
                            else:
                                opts = FOpts(model_asset_path=model_path)
                        except TypeError:
                            # fallback to keyword used by some builds
                            opts = FOpts(model_asset_path=model_path)
                        if hasattr(FL, "create_from_options"):
                            return FL.create_from_options(opts)
                        if hasattr(FL, "create"):
                            return FL.create(opts)
                        return FL(opts)
                except Exception:
                    pass

                # Pattern C: helper create_from_model_path or create_from_options on vision module
                try:
                    if hasattr(FL, "create_from_model_path"):
                        return FL.create_from_model_path(model_path)
                except Exception:
                    pass

                try:
                    # try using top-level MP_TASKS FaceLandmarker static creators
                    if hasattr(MP_TASKS, "FaceLandmarker") and hasattr(MP_TASKS.FaceLandmarker, "create_from_options"):
                        opts = None
                        if FOpts:
                            if RunningMode:
                                opts = FOpts(model_asset_path=model_path, running_mode=RunningMode.IMAGE)
                            else:
                                opts = FOpts(model_asset_path=model_path)
                        if opts is not None:
                            return MP_TASKS.FaceLandmarker.create_from_options(opts)
                except Exception:
                    pass

                # If we reach here, provide helpful diagnostics
                raise RuntimeError("Unable to construct FaceLandmarker with available mediapipe.tasks API. Available attrs: " + ",".join([a for a in dir(MP_TASKS) if not a.startswith("_")]))

            try:
                face_landmarker = _create_face_landmarker(task_model)
            except Exception as e:
                print("Warning: failed to construct FaceLandmarker from Tasks API:", e)
                print("Falling back to Haar-cascade approximate landmark generation.")
                face_landmarker = None
                engine_mode = "haar_fallback"
    else:
        raise RuntimeError(
            "No usable MediaPipe FaceMesh API found. Install a compatible 'mediapipe' or provide a Tasks model via --task_model."
        )

    # If Tasks creation failed earlier, allow a Haar fallback to keep the pipeline runnable.
    if engine_mode == "tasks" and face_landmarker is None:
        print("Warning: could not create MediaPipe Tasks FaceLandmarker; falling back to Haar-cascade approximate landmarks.")
        engine_mode = "haar_fallback"

    for img_path in tqdm(img_files, desc="Processing images"):
        fname = os.path.basename(img_path)
        label = None
        if labels_map and fname in labels_map:
            label = labels_map[fname]
        elif dataset.lower() == "fer2013" and map_fer_labels:
            parent = os.path.basename(os.path.dirname(img_path))
            label = map_fer_label_to_binary(parent)
        elif dataset.lower() == "mpiigaze":
            label = labels_map.get(fname, fname)
        else:
            parent = os.path.basename(os.path.dirname(img_path))
            label = labels_map.get(fname, parent if parent else "unknown")

        if engine_mode == "solutions":
            coords = process_image(img_path, face_mesh)
        elif engine_mode == "tasks":
            coords = process_image_tasks(img_path, face_landmarker)
        elif engine_mode == "haar_fallback":
            coords = process_image_haar(img_path)
        else:
            coords = None

        if coords is None:
            row = [label] + [""] * (478 * 3)
            rows_out.append(row)
            continue

        if len(coords) < 478:
            coords = coords + [(0.0, 0.0, 0.0)] * (478 - len(coords))
        elif len(coords) > 478:
            coords = coords[:478]

        flat = [label]
        for (x, y, z) in coords:
            flat.extend([x, y, z])
        rows_out.append(flat)

        if dataset.lower() == "mpiigaze" and extra_mpiigaze_json:
            iris_pts = [(float(coords[i][0]), float(coords[i][1]), float(coords[i][2])) for i in _IRIS_RANGE]
            eye_corners = {
                "left_inner": tuple(coords[_LEFT_INNER]),
                "left_outer": tuple(coords[_LEFT_OUTER]),
                "right_inner": tuple(coords[_RIGHT_INNER]),
                "right_outer": tuple(coords[_RIGHT_OUTER]),
            }
            extra_mpiigaze.append(
                {"file": fname, "iris": iris_pts, "eye_corners": eye_corners}
            )

    write_csv_header_and_rows(output_file, rows_out)

    if dataset.lower() == "mpiigaze" and extra_mpiigaze_json:
        with open(extra_mpiigaze_json, "w", encoding="utf-8") as fh:
            for obj in extra_mpiigaze:
                fh.write(json.dumps(obj) + "\n")

    # Close Tasks API object if created
    if face_landmarker is not None:
        try:
            face_landmarker.close()
        except Exception:
            pass

    # If we created a solutions FaceMesh context manually, attempt to close it
    if MP_FACE is not None and face_mesh is not None:
        try:
            face_mesh.close()
        except Exception:
            pass

    print(f"Done. Wrote {len(rows_out)} rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess images -> MediaPipe 478 landmarks (privacy-first).")
    parser.add_argument("--dataset", required=True, choices=["mpiigaze", "fer2013", "generic"], help="Which dataset pattern to follow")
    parser.add_argument("--input_dir", required=True, help="Directory containing images (recursive)")
    parser.add_argument("--output_file", required=True, help="CSV path to write landmarks")
    parser.add_argument("--labels_csv", help="Optional CSV mapping filename->label")
    parser.add_argument("--map_fer_labels", action="store_true", help="For FER-2013: map folder labels to Stress/Neutral")
    parser.add_argument("--extra_mpiigaze_json", help="Optional JSONL path to write isolated iris/eye-corner data (MPIIGaze)")
    parser.add_argument("--task_model", help="Optional path to a MediaPipe Tasks 'face_landmarker.task' model file (used when mp.solutions is unavailable)")
    args = parser.parse_args()

    process_dataset(
        dataset=args.dataset,
        input_dir=args.input_dir,
        output_file=args.output_file,
        labels_csv=args.labels_csv,
        map_fer_labels=args.map_fer_labels,
        extra_mpiigaze_json=args.extra_mpiigaze_json,
        task_model=args.task_model,
    )


if __name__ == "__main__":
    main()
