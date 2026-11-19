"""Turn one fragment (its observation rows and its map crop) into the encoder's inputs.

Everything is expressed in the recording vehicle's frame at the fragment's first sample, except
`pose`, which places the scene in the world so the encoder can register it against the map.
"""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ["car", "suv", "pickup", "van", "truck", "semi", "bus", "motorcycle", "cyclist",
         "pedestrian", "emergency", "oversize", "barrier", "cone", "debris"]
SIGNALS = ["L", "R", "H", "E"]
SURFACES = ["road", "shoulder", "sidewalk", "median", "island", "plaza", "bridge"]
LINES = ["lane", "edge", "center", "stop", "yield", "park", "barrier", "passing"]
LIGHTS = ["G", "Y", "R"]

N_OBJ = 24       # nearest objects kept per fragment
N_T = 8          # samples per object trajectory
OBJ_DIM = N_T * 5 + len(KINDS) + 3 + len(SIGNALS)
MAP_DIM = len(SURFACES) + len(LINES) + 5 + len(LIGHTS)
POSE_FREQS = 16
POSE_DIM = 4 * POSE_FREQS


def load_observations(data_dir):
    files = sorted(Path(data_dir).glob("observations-*.csv.gz"))
    return pd.concat([pd.read_csv(f, dtype={"signal": "string", "track_id": "string"}) for f in files], ignore_index=True)


def load_maps(data_dir):
    maps = {}
    for f in sorted(Path(data_dir).glob("maps-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                if line.strip():
                    m = json.loads(line)
                    maps[m["fragment_id"]] = m
    return maps


def _track_token(tr, ox, oy, c, s, yaw0, t_grid):
    tok = np.zeros(OBJ_DIM, np.float32)
    t = tr["t"].to_numpy()
    order = np.argsort(t)
    t = t[order]
    x = tr["x"].to_numpy()[order] - ox
    y = tr["y"].to_numpy()[order] - oy
    lx, ly = x * c + y * s, -x * s + y * c
    yaw = tr["yaw"].to_numpy()[order] - yaw0
    v = tr["v"].to_numpy()[order]
    idx = np.clip(np.searchsorted(t, t_grid), 0, len(t) - 1)
    present = np.abs(t[idx] - t_grid) <= 0.25
    block = np.stack([lx[idx] / 50, ly[idx] / 50, np.cos(yaw[idx]), np.sin(yaw[idx]), v[idx] / 30], 1)
    block[~present] = 0
    tok[: N_T * 5] = block.reshape(-1)
    first = tr.iloc[0]
    if first["kind"] in KINDS:
        tok[N_T * 5 + KINDS.index(first["kind"])] = 1
    o = N_T * 5 + len(KINDS)
    tok[o: o + 3] = [first["length"] / 10, first["width"] / 5, tr["z"].mean() / 10]
    sig = tr["signal"].fillna("")
    for i, k in enumerate(SIGNALS):
        tok[o + 3 + i] = float((sig == k).mean())
    return tok


def _map_vector(m):
    vec = np.zeros(MAP_DIM, np.float32)
    if not m:
        return vec
    for sf in m["surfaces"]:
        if sf["kind"] in SURFACES:
            vec[SURFACES.index(sf["kind"])] += 1
    o = len(SURFACES)
    for ln in m["lines"]:
        if ln["kind"] in LINES:
            vec[o + LINES.index(ln["kind"])] += 1
    o += len(LINES)
    zs = [z for ln in m["lanes"] for z in ln.get("z", [])]
    vec[o: o + 5] = [len(m["crosswalks"]), len(m["lanes"]), len(m["signals"]), max(zs, default=0) / 10, len({round(z) for z in zs})]
    o += 5
    for sg in m["signals"]:
        for _, st in sg["states"]:
            if st in LIGHTS:
                vec[o + LIGHTS.index(st)] += 1
    return np.log1p(vec)


def _pose(x, y, yaw):
    freqs = 2 * math.pi / np.geomspace(80_000, 40, POSE_FREQS)
    return np.concatenate([np.sin(x * freqs), np.cos(x * freqs), np.sin(y * freqs), np.cos(y * freqs)]).astype(np.float32)


def fragment_features(rows, m):
    """rows: the observation rows of one fragment. m: its map crop (or None)."""
    objs = np.zeros((N_OBJ, OBJ_DIM), np.float32)
    mask = np.zeros(N_OBJ, bool)
    ego_tok = np.zeros(OBJ_DIM, np.float32)
    pose = np.zeros(POSE_DIM, np.float32)
    if len(rows):
        dur = rows["t"].max()
        t_grid = np.linspace(0, dur, N_T)
        ego = rows[rows["track_id"] == "0"]
        start = (ego if len(ego) else rows).sort_values("t").iloc[0]
        ox, oy, yaw0 = start["x"], start["y"], (start["yaw"] if len(ego) else 0.0)
        c, s = math.cos(yaw0), math.sin(yaw0)
        pose = _pose(ox, oy, yaw0)
        if len(ego):
            ego_tok = _track_token(ego, ox, oy, c, s, yaw0, t_grid)
        others = []
        for tid, tr in rows[rows["track_id"] != "0"].groupby("track_id"):
            d = np.hypot(tr["x"].to_numpy() - ox, tr["y"].to_numpy() - oy).min()
            others.append((d, tr))
        others.sort(key=lambda p: p[0])
        for i, (_, tr) in enumerate(others[:N_OBJ]):
            objs[i] = _track_token(tr, ox, oy, c, s, yaw0, t_grid)
            mask[i] = True
    return {"objs": objs, "mask": mask, "ego": ego_tok, "map": _map_vector(m), "pose": pose}


def stack(feats):
    return {k: np.stack([f[k] for f in feats]) for k in feats[0]}


def all_features(data_dir, fragment_ids):
    """Features for many fragments at once (one pass over the rows)."""
    obs = load_observations(data_dir)
    maps = load_maps(data_dir)
    groups = dict(tuple(obs.groupby("fragment_id")))
    empty = obs.iloc[:0]
    return stack([fragment_features(groups.get(fid, empty), maps.get(fid)) for fid in fragment_ids])
