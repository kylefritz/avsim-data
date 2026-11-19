"""Embed every fragment; writes embeddings.npy (one row per line of fragments.csv).

    uv run model/embed.py
"""
import time

import numpy as np
import pandas as pd
import torch

from common import DATA, HERE, load_model, log
from features import fragment_features, load_maps, load_observations


def main():
    t0 = time.time()
    obs = load_observations(DATA)
    maps = load_maps(DATA)
    ids = pd.read_csv(DATA / "fragments.csv")["fragment_id"].tolist()
    log(f"loaded {len(obs):,} rows, {len(ids):,} fragments", t0)
    model, _ = load_model()
    out = np.zeros((len(ids), 64), np.float32)
    for i, fid in enumerate(ids):
        f = fragment_features(obs[obs["fragment_id"] == fid], maps.get(fid))
        with torch.no_grad():
            x = {k: torch.from_numpy(v).unsqueeze(0) for k, v in f.items()}
            out[i] = model.embed(**x)[0].numpy()
        if i % 1000 == 0:
            log(f"  {i:,} / {len(ids):,}", t0)
    np.save(HERE.parent / "embeddings.npy", out)
    log(f"wrote embeddings.npy {out.shape}", t0)


if __name__ == "__main__":
    main()
