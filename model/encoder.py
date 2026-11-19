import torch
from torch import nn

from features import MAP_DIM, OBJ_DIM, POSE_DIM


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class SceneEncoder(nn.Module):
    """Objects, the recording vehicle, the map and the world pose -> a 64-d scene embedding."""

    def __init__(self, n_classes, dim=128, emb=64):
        super().__init__()
        self.obj = mlp(OBJ_DIM, dim, dim)
        self.ego = mlp(OBJ_DIM, dim, dim)
        self.map = mlp(MAP_DIM, dim, dim)
        self.pose = mlp(POSE_DIM, dim, dim)
        self.trunk = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, emb))
        self.head = nn.Linear(emb, n_classes)

    def embed(self, objs, mask, ego, map, pose):
        h = self.obj(objs)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (h + (m - 1) * 1e4).max(1).values * (m.sum(1) > 0).float()
        return self.trunk(torch.cat([mean, mx, self.ego(ego), self.map(map), self.pose(pose)], -1))

    def forward(self, **x):
        return self.head(self.embed(**x))
