#!/usr/bin/env python3
"""
Optional: fine-tune EmotionCNN (512-D) with DISFA+ AU regression (default AU4 brow lowerer).

Run after `python train_affective_head.py` so weights exist, or train from scratch.

  python train_disfa_finetune.py --max_frames 8000 --epochs 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from load_disfa import iter_disfa_frames
from train_affective_head import EmotionCNN, DEFAULT_EMBED_DIM


def main():
    p = argparse.ArgumentParser(description="DISFA AU auxiliary fine-tune")
    p.add_argument("--disfa_root", default=os.path.join("Dataset", "DISFA+"))
    p.add_argument("--max_frames", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--au", default="AU4", help="AU key from load_disfa (e.g. AU4)")
    p.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED_DIM)
    p.add_argument("--cnn_ckpt", default=os.path.join("models", "affective_cnn.pth"))
    p.add_argument("--output", default=os.path.join("models", "affective_cnn_disfa.pt"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    xs: list = []
    ys: list = []
    for img, au_map, _lnd, _meta in iter_disfa_frames(
        args.disfa_root, max_frames=args.max_frames
    ):
        t = au_map.get(args.au, 0.0) / 5.0
        xs.append(img)
        ys.append(t)

    if len(xs) < 64:
        raise SystemExit(f"Too few DISFA frames ({len(xs)}). Check {args.disfa_root}")

    X = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
    Y = torch.tensor(np.asarray(ys, dtype=np.float32).reshape(-1, 1))
    loader = DataLoader(TensorDataset(X, Y), batch_size=args.batch_size, shuffle=True)

    model = EmotionCNN(embed_dim=args.embed_dim).to(device)
    if os.path.isfile(args.cnn_ckpt):
        try:
            state = torch.load(args.cnn_ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.cnn_ckpt, map_location=device)
        try:
            model.load_state_dict(state, strict=True)
            print("Loaded", args.cnn_ckpt)
        except Exception as e:
            print("Checkpoint mismatch (embed_dim?), training partly from scratch:", e)
            model.load_state_dict(state, strict=False)

    for p_ in model.classifier.parameters():
        p_.requires_grad = False

    penult_dim = args.embed_dim
    aux = nn.Linear(penult_dim, 1).to(device)

    opt = optim.Adam(
        list(model.features.parameters()) + list(aux.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )
    crit = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        aux.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            feat = model.features(xb)
            xflat = model.classifier[0](feat)
            xlin = model.classifier[1](xflat)
            z = model.classifier[2](xlin)
            pred = aux(z)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch+1}/{args.epochs} mse_loss={total/max(len(loader),1):.5f}")

    torch.save(model.state_dict(), args.output)
    print("Saved CNN to", args.output, "(re-point AffectiveHead cnn_path if needed)")


if __name__ == "__main__":
    main()
