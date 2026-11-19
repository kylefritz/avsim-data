import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from encoder import SceneEncoder
from features import all_features

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
LABELS = HERE.parent / "labels.csv"
WEIGHTS = HERE / "weights.pt"
CACHE = HERE / ".features.npz"


def log(msg, t0=None):
    print(msg + (f" ({time.time() - t0:.1f}s)" if t0 else ""), flush=True)


def labeled_features(labels):
    """Features for the labeled fragments, cached so retraining doesn't re-read 8M rows."""
    ids = labels["fragment_id"].tolist()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["ids"]) == ids:
            return {k: z[k] for k in z.files if k != "ids"}
    t0 = time.time()
    feats = all_features(DATA, ids)
    np.savez(CACHE, ids=np.array(ids, dtype=object), **feats)
    log(f"built features for {len(ids):,} fragments", t0)
    return feats


def to_tensors(feats, idx=None):
    return {k: torch.from_numpy(v if idx is None else v[idx]) for k, v in feats.items()}


def load_model():
    ck = torch.load(WEIGHTS, weights_only=False)
    model = SceneEncoder(len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["classes"]


def read_labels(path=LABELS):
    return pd.read_csv(path)
