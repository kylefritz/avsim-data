"""Find the scenes most like a given fragment.

    uv run model/search.py FRAGMENT_ID [--k 10]
"""
import argparse

import numpy as np
import pandas as pd

from common import DATA, HERE, read_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragment_id")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    ids = pd.read_csv(DATA / "fragments.csv")["fragment_id"].tolist()
    emb = np.load(HERE.parent / "embeddings.npy")
    known = dict(zip(*read_labels()[["fragment_id", "scenario"]].to_numpy().T))
    q = ids.index(a.fragment_id)
    scores = emb @ emb[q]
    for j in np.argsort(-scores)[1: a.k + 1]:
        print(f"{ids[j]}  {scores[j]:8.2f}  {known.get(ids[j], '')}")


if __name__ == "__main__":
    main()
