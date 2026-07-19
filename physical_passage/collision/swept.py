"""Continuous (swept) collision check between two poses.

A single 0.05 m / 5-degree action could tunnel through a thin wall member if we
only checked the end pose, so we interpolate the motion (linear position lerp +
quaternion slerp) and run the discrete evaluator at every substep. An action is
unsafe if ANY substep collides, even when start and end poses are both clear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..sim.connection import Sim
from .evaluator import ContactInfo, evaluate

Pose = tuple[tuple, tuple]  # (position xyz, quaternion xyzw)


def _slerp(q0, q1, t: float):
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:              # take the short arc
        q1 = [-x for x in q1]
        dot = -dot
    if dot > 0.9995:           # nearly parallel -> lerp + normalize
        q = [a + t * (b - a) for a, b in zip(q0, q1)]
        n = math.sqrt(sum(x * x for x in q))
        return tuple(x / n for x in q)
    th0 = math.acos(dot)
    s0 = math.sin((1 - t) * th0) / math.sin(th0)
    s1 = math.sin(t * th0) / math.sin(th0)
    return tuple(s0 * a + s1 * b for a, b in zip(q0, q1))


@dataclass
class SweptResult:
    collided: bool
    min_clearance: float          # min over all substeps
    time_to_collision: float      # fraction of the step in [0,1]; 1.0 = clear
    first_contact: ContactInfo | None


def swept_check(sim: Sim, cuboid: int, wall_parts: list[int],
                pose0: Pose, pose1: Pose, substeps: int = 8,
                touch_eps: float = 1e-4, restore: bool = True) -> SweptResult:
    (p0, q0), (p1, q1) = pose0, pose1
    dmin = float("inf")
    hit_i, hit_info = None, None
    for i in range(1, substeps + 1):
        t = i / substeps
        pos = tuple(a + t * (b - a) for a, b in zip(p0, p1))
        quat = _slerp(q0, q1, t)
        sim.set_pose(cuboid, pos, quat)
        info = evaluate(sim, cuboid, wall_parts, touch_eps)
        dmin = min(dmin, info.min_clearance)
        if info.collision and hit_i is None:
            hit_i, hit_info = i, info
            break                       # earliest contact is what matters
    if restore:
        sim.set_pose(cuboid, p0, q0)
    else:                               # leave the body at the end pose
        sim.set_pose(cuboid, p1, q1)
    return SweptResult(collided=hit_i is not None, min_clearance=dmin,
                       time_to_collision=(hit_i / substeps) if hit_i else 1.0,
                       first_contact=hit_info)
