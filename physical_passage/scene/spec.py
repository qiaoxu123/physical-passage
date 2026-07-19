"""Scene specification: everything needed to reproduce one episode's level."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

Label = str  # "feasible" | "rotation_required" | "impossible"


@dataclass
class SceneSpec:
    # controllable cuboid: FULL extents along x, y, z at identity orientation
    cuboid_dims: tuple[float, float, float] = (0.20, 0.12, 0.34)
    start_pos: tuple[float, float, float] = (0.0, -0.55, 0.55)
    start_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # wall with one rectangular hole; wall occupies the y=0 plane, thickness t
    wall_thickness: float = 0.05
    hole_center: tuple[float, float] = (0.0, 0.55)   # (x, z)
    hole_size: tuple[float, float] = (0.30, 0.42)    # (width_x, height_z)
    # ground-truth label from the feasibility solver
    label: Label = "feasible"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneSpec":
        d = dict(d)
        for k in ("cuboid_dims", "start_pos", "start_rpy_deg", "hole_center", "hole_size"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)
