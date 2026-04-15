#!/usr/bin/env python3
"""
Train the HybridGazeModel on the Eye Gaze Detection dataset (ImprovementSet JSON + zone folders).

Synthetic renders do not register with MediaPipe; supervision uses parsed JSON geometry
(see gaze_geometry.keypoints_from_eye_gaze_json).

Loss options: MSE on (x,y), angular loss on rays, or weighted combination.
Temporal length N: repeat the same feature vector over N frames (static samples) so the
TCN learns identity-style smoothing; live video then supplies real trajectories.

Legacy mode: RandomForest on MPIIGaze (--mode rf).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


from gaze_geometry import (
    GAZE_FEATURE_DIM,
    extract_gaze_feature_vector,
    iter_eye_gaze_improvement_samples,
    keypoints_from_eye_gaze_json,
    load_eye_gaze_json_file,
    look_vec_to_screen_xy,
    resolve_eye_gaze_base_dir,
)
from gaze_head import HybridGazeModel


def gaze_angular_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """1 - cos(theta) between rays through fronto-parallel screen points."""
    px = 2.0 * pred[:, 0] - 1.0
    py = 2.0 * pred[:, 1] - 1.0
    pz = torch.ones_like(px)
    tx = 2.0 * target[:, 0] - 1.0
    ty = 2.0 * target[:, 1] - 1.0
    tz = torch.ones_like(tx)
    vp = torch.stack([px, py, pz], dim=1)
    vt = torch.stack([tx, ty, tz], dim=1)
    vp = vp / (torch.linalg.norm(vp, dim=1, keepdim=True) + eps)
    vt = vt / (torch.linalg.norm(vt, dim=1, keepdim=True) + eps)
    cos = (vp * vt).sum(dim=1).clamp(-1.0 + eps, 1.0 - eps)
    return (1.0 - cos).mean()


def gaze_combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    angular_weight: float,
) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred, target)
    if loss_type == "mse":
        return mse
    if loss_type == "angular":
        return gaze_angular_loss(pred, target)
    if loss_type == "mse_angular":
        return mse + float(angular_weight) * gaze_angular_loss(pred, target)
    raise ValueError(f"Unknown loss_type {loss_type}")


def _collect_records(
    base_dir: Path,
    target_mode: str,
    blend_alpha: float,
    max_samples: int | None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    print("Indexing paired image/JSON paths...", flush=True)
    paths = list(iter_eye_gaze_improvement_samples(base_dir))
    print(f"Found {len(paths)} pairs; loading features (this may take a few minutes)...", flush=True)
    for jid, _img, zone, zxy in tqdm(paths, desc="Load JSON", mininterval=1.0):
        try:
            data = load_eye_gaze_json_file(jid)
            kp = keypoints_from_eye_gaze_json(data)
            feat = extract_gaze_feature_vector(kp)
            hp = np.array(kp["head_pose"], dtype=np.float32)
            zx, zy = zxy
            if target_mode == "zone":
                tx, ty = zx, zy
            elif target_mode == "look_vec":
                lv = data["eye_details"]["look_vec"]
                lx, ly = look_vec_to_screen_xy(lv)
                tx, ty = lx, ly
            elif target_mode == "blend":
                lx, ly = look_vec_to_screen_xy(data["eye_details"]["look_vec"])
                a = float(blend_alpha)
                tx = a * lx + (1.0 - a) * zx
                ty = a * ly + (1.0 - a) * zy
            else:
                raise ValueError(target_mode)
            tgt = np.array([tx, ty], dtype=np.float32)
            rows.append((feat, tgt, hp))
            if max_samples is not None and len(rows) >= max_samples:
                break
        except (KeyError, ValueError, TypeError, OSError):
            continue
    return rows


class EyeGazeSequenceDataset(Dataset):
    """Each item is a repeated feature sequence (static dataset) + target xy + head pose."""

    def __init__(self, records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], seq_len: int):
        self.records = records
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        feat, tgt, hp = self.records[idx]
        seq = np.tile(feat[np.newaxis, :], (self.seq_len, 1)).astype(np.float32)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(tgt),
            torch.from_numpy(hp),
        )


def train_regression(args: argparse.Namespace) -> None:
    base = Path(args.data_root) if args.data_root else resolve_eye_gaze_base_dir()
    if not base.is_dir():
        raise SystemExit(f"Data root not found: {base}")

    print(f"Loading samples from {base} ...", flush=True)
    records = _collect_records(
        base,
        target_mode=args.target,
        blend_alpha=args.blend_alpha,
        max_samples=args.max_samples,
    )
    if len(records) < 10:
        raise SystemExit(f"Too few training samples ({len(records)}). Check dataset paths.")
    print(f"Loaded {len(records)} training records.", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n = len(records)
    if n > 1:
        n_val = min(max(1, int(n * args.val_ratio)), n - 1)
    else:
        n_val = 0
    val_records = records[:n_val]
    train_records = records[n_val:]
    if not train_records:
        train_records, val_records = val_records, []

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Using CPU (--cpu or CUDA unavailable).", flush=True)
    model = HybridGazeModel(
        seq_len=args.seq_len, input_dim=GAZE_FEATURE_DIM
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        EyeGazeSequenceDataset(train_records, args.seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            EyeGazeSequenceDataset(val_records, args.seq_len),
            batch_size=args.batch_size,
            shuffle=False,
        )
        if val_records
        else None
    )

    best_val = math.inf
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batch = 0
        for seq, tgt, hp in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            seq = seq.to(device)
            tgt = tgt.to(device)
            hp = hp.to(device)
            opt.zero_grad()
            pred = model(seq, head_pose=hp)
            loss = gaze_combined_loss(pred, tgt, args.loss, args.angular_weight)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batch += 1

        train_loss = running / max(n_batch, 1)
        val_loss = None
        if val_loader is not None:
            model.eval()
            vsum = 0.0
            vb = 0
            with torch.no_grad():
                for seq, tgt, hp in val_loader:
                    seq = seq.to(device)
                    tgt = tgt.to(device)
                    hp = hp.to(device)
                    pred = model(seq, head_pose=hp)
                    vsum += float(
                        gaze_combined_loss(
                            pred, tgt, args.loss, args.angular_weight
                        ).item()
                    )
                    vb += 1
            val_loss = vsum / max(vb, 1)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_path)
        else:
            torch.save(model.state_dict(), out_path)

        msg = f"epoch {epoch+1} train_loss={train_loss:.5f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.5f}"
        print(msg, flush=True)

    if val_loader is not None:
        print(f"Best val loss {best_val:.5f}; weights -> {out_path}", flush=True)
    else:
        print(f"Saved weights -> {out_path}", flush=True)


def train_gaze_rf() -> None:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from load_mpiigaze import load_mpiigaze_data

    base = os.path.join("Dataset", "MPIIGAZE", "MPIIGaze")
    print("Loading MPIIGaze...")
    X, y = load_mpiigaze_data(base, max_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, os.path.join("models", "gaze_rf.joblib"))
    print("Saved models/gaze_rf.joblib")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    p = argparse.ArgumentParser(description="Sentin-Edge gaze head training")
    p.add_argument(
        "--mode",
        choices=("regression", "rf"),
        default="regression",
        help="regression = PyTorch (x,y); rf = legacy MPIIGaze RandomForest",
    )
    p.add_argument(
        "--data_root",
        default="",
        help="Eye Gaze Detection base (contains ImprovementSet/ and Json/). Default: Dataset/.../ImprovementSet/ImprovementSet",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--seq_len",
        type=int,
        default=15,
        help="Temporal window N (10–25 typical). Static samples repeat features across N.",
    )
    p.add_argument(
        "--loss",
        choices=("mse", "angular", "mse_angular"),
        default="mse",
    )
    p.add_argument(
        "--angular_weight",
        type=float,
        default=0.3,
        help="Weight for angular term when --loss mse_angular",
    )
    p.add_argument(
        "--target",
        choices=("zone", "look_vec", "blend"),
        default="look_vec",
        help="zone=folder grid targets; look_vec=JSON gaze vector; blend=mix with --blend_alpha",
    )
    p.add_argument("--blend_alpha", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=0, help="0 = use all")
    p.add_argument(
        "--output",
        default="models/gaze_hybrid.pth",
        help="Where to save state_dict",
    )
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.max_samples <= 0:
        args.max_samples = None

    if args.mode == "rf":
        train_gaze_rf()
    else:
        train_regression(args)


if __name__ == "__main__":
    main()
