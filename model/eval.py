"""Accuracy of the encoder's scenario head on the test split; writes eval_report.md.

    uv run model/eval.py
"""
from collections import Counter

import numpy as np
import torch

from common import HERE, labeled_features, load_model, read_labels, to_tensors


def main():
    labels = read_labels()
    model, classes = load_model()
    feats = labeled_features(labels)
    te = np.flatnonzero(labels["split"].to_numpy() == "test")
    with torch.no_grad():
        pred = model(**to_tensors(feats, te)).argmax(1).numpy()
    truth = labels["scenario"].to_numpy()[te]
    got = np.array(classes)[pred] == truth
    n = Counter(truth)
    ok = Counter(truth[got])
    lines = [
        "# Scene encoder: eval report",
        "",
        f"Test split: {len(te):,} labeled fragments. **Accuracy {got.mean():.1%}** over {len(classes)} scenarios.",
        "",
        "| scenario | fragments | accuracy |",
        "|---|---|---|",
        *[f"| {c} | {n[c]} | {ok[c] / n[c]:.1%} |" for c in sorted(n)],
    ]
    (HERE / "eval_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))


if __name__ == "__main__":
    main()
