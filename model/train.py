"""Train the scene encoder on the labeled fragments.

    uv run model/train.py [--labels labels.csv] [--epochs 40]

Uses the rows whose `split` is "train"; reports accuracy on "test" as it goes.
"""
import argparse
import time

import numpy as np
import torch
from torch import nn

from common import LABELS, WEIGHTS, labeled_features, log, read_labels, to_tensors
from encoder import SceneEncoder


def fit(feats, y, tr, te, n_classes, epochs=40, seed=0, verbose=True):
    torch.manual_seed(seed)
    Xtr, Xte = to_tensors(feats, tr), to_tensors(feats, te)
    model = SceneEncoder(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    acc = float("nan")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for b in range(0, len(tr), 128):
            i = perm[b: b + 128]
            loss = loss_fn(model(**{k: v[i] for k, v in Xtr.items()}), y[tr][i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(**Xte).argmax(1) == y[te]).float().mean().item()
            if verbose:
                log(f"epoch {ep + 1:>3}  loss {loss.item():.3f}  test accuracy {acc:.3f}", t0)
    model.eval()
    return model, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=WEIGHTS)
    a = ap.parse_args()
    labels = read_labels(a.labels)
    classes = sorted(labels["scenario"].unique())
    y = torch.tensor(labels["scenario"].map({c: i for i, c in enumerate(classes)}).to_numpy())
    feats = labeled_features(labels)
    split = labels["split"].to_numpy()
    model, _ = fit(feats, y, np.flatnonzero(split == "train"), np.flatnonzero(split == "test"), len(classes), a.epochs, a.seed)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, a.out)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
