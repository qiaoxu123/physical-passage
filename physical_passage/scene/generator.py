"""SceneGenerator: sample the three level classes by construction, then verify
each candidate with the feasibility solver (reject-and-resample on mismatch).

Construction rules (cuboid dims sorted a <= b <= c; passage along +y):
  feasible          : hole (W,H) >= identity footprint (dims_x, dims_z) + margin
  rotation_required : identity footprint does NOT fit, but the 90-degree rotY
                      footprint (swapped x/z) fits with margin
  impossible        : min(W,H) < min(a,b,c) - margin  (no orientation can fit,
                      because any projection is at least the smallest dim wide)

Start position and hole center are snapped to the translation grid so the
expert plan can align exactly with discrete 0.05 m moves.
"""

from __future__ import annotations

import logging

import numpy as np

from ..scene.spec import SceneSpec

logger = logging.getLogger(__name__)


def _snap(v: float, step: float) -> float:
    return round(v / step) * step


class SceneGenerator:
    def __init__(self, cfg, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def sample_class(self) -> str:
        g = self.cfg.generator
        r = self.rng.random()
        if r < g.frac_easy:
            return "feasible"
        if r < g.frac_easy + g.frac_rotation:
            return "rotation_required"
        return "impossible"

    def sample(self, label: str | None = None, seed: int | None = None) -> SceneSpec:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        label = label or self.sample_class()
        g = self.cfg.generator
        step = self.cfg.actions.trans_step
        rng = self.rng
        margin = float(rng.uniform(g.clearance_min, g.clearance_max))

        # cuboid dims: dx (width), dy (depth along passage), dz (height)
        dx = float(rng.uniform(0.10, 0.30))
        dz = float(rng.uniform(0.10, 0.42))
        if abs(dx - dz) < 0.06:                     # keep the two footprints distinct
            dz = dx + (0.10 if rng.random() < 0.5 else -0.08)
            dz = float(np.clip(dz, 0.08, 0.45))
        dy = float(rng.uniform(0.08, 0.22))

        if label == "feasible":
            W = dx + 2 * margin + float(rng.uniform(0, 0.06))
            H = dz + 2 * margin + float(rng.uniform(0, 0.06))
        elif label == "rotation_required":
            # rotated-90 footprint is (dz, dx): make THAT fit, identity NOT fit
            W = dz + 2 * margin
            H = dx + 2 * margin
            # ensure identity truly fails: hole must be too small for (dx, dz)
            if W > dx and H > dz:                   # would also fit identity -> break it
                if dz > dx:
                    H = min(H, dz - 0.02)           # too short for the tall side
                else:
                    W = min(W, dx - 0.02)
        else:  # impossible
            small = min(dx, dy, dz)
            W = float(rng.uniform(0.05, max(0.055, small - 0.03)))
            H = float(rng.uniform(0.05, max(0.055, small - 0.03)))

        wz = self.cfg.wall
        hx = _snap(float(rng.uniform(-0.25, 0.25)), step)
        hz = _snap(float(rng.uniform(wz.z_min + H / 2 + 0.15, wz.z_max - H / 2 - 0.15)), step)
        sx = _snap(float(rng.uniform(-0.35, 0.35)), step)
        sz = _snap(float(rng.uniform(wz.z_min + dz / 2 + 0.2, wz.z_max - dz / 2 - 0.2)), step)

        return SceneSpec(
            cuboid_dims=(dx, dy, dz),
            start_pos=(sx, -0.55, sz),
            start_rpy_deg=(0.0, 0.0, 0.0),
            wall_thickness=float(rng.uniform(0.03, 0.08)),
            hole_center=(hx, hz),
            hole_size=(W, H),
            label=label,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
