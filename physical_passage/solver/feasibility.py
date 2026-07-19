"""Feasibility ground truth + expert plan for the MVP single-wall scene.

Strategy (grounded in the real collision engine, not just geometry):
  1. Analytic screen: a box's projection onto the wall plane has width >= its
     smallest dimension in every direction, so if the hole's smaller side is
     below min(cuboid dims) the level is provably impossible for ANY pose.
  2. Pass search: for each candidate orientation (in-plane rotY grid over the
     three face families reachable by 90-degree rotX/rotZ), place the cuboid at
     the hole center and sweep it straight through the wall with the swept
     collision checker. First orientation that passes wins.
       - identity (theta=0) passes  -> "feasible"
       - only theta != 0 passes     -> "rotation_required"
       - nothing passes + screen ok -> "impossible"
  3. plan(): rotate in place (near side is obstacle-free), align x/z to the
     hole center, then MOVE_FORWARD until fully past the wall. The generator
     snaps start/hole positions to the translation grid so exact alignment is
     reachable; the environment's swept checks validate every step at runtime.

The same plan doubles as expert demonstrations for behavior cloning later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pybullet as p

from ..collision.swept import swept_check
from ..envs.actions import A
from ..scene.builder import SceneHandles
from ..scene.spec import SceneSpec
from ..sim.connection import Sim

SWEEP_SUBSTEPS = 60          # straight pass sweep resolution (~1.7 cm per substep)
PASS_MARGIN = 0.08           # extra forward travel beyond the far wall face


@dataclass
class Solution:
    label: str                       # feasible | rotation_required | impossible
    theta_deg: float | None          # winning in-plane rotation (about y)
    family: str | None               # "y" | "x90" | "z90" (pre-rotation family)
    actions: list[int] | None        # expert action sequence; None if impossible


def _quat_rot(axis: str, deg: float):
    r = math.radians(deg)
    e = {"x": (r, 0, 0), "y": (0, r, 0), "z": (0, 0, r)}[axis]
    return p.getQuaternionFromEuler(e)


def _family_quat(family: str, theta_deg: float):
    qy = _quat_rot("y", theta_deg)
    if family == "y":
        return qy
    pre = _quat_rot("x", 90.0) if family == "x90" else _quat_rot("z", 90.0)
    return p.multiplyTransforms((0, 0, 0), qy, (0, 0, 0), pre)[1]


def analytic_impossible(spec: SceneSpec, margin: float = 0.0) -> bool:
    return min(spec.hole_size) < min(spec.cuboid_dims) - margin


def _pass_clear(sim: Sim, h: SceneHandles, spec: SceneSpec, quat,
                touch_eps: float) -> bool:
    hx, hz = spec.hole_center
    y0, y1 = -0.45, 0.45
    res = swept_check(sim, h.cuboid, h.wall_parts,
                      ((hx, y0, hz), quat), ((hx, y1, hz), quat),
                      substeps=SWEEP_SUBSTEPS, touch_eps=touch_eps, restore=True)
    return not res.collided


def solve(sim: Sim, h: SceneHandles, spec: SceneSpec, cfg) -> Solution:
    touch_eps = cfg.collision.touch_eps
    rot = cfg.actions.rot_step_deg
    saved = sim.get_pose(h.cuboid)

    try:
        if analytic_impossible(spec):
            return Solution("impossible", None, None, None)
        thetas = [i * rot for i in range(int(90.0 / rot) + 1)]
        for family in ("y", "x90", "z90"):        # generator only uses "y"
            for th in thetas:
                if _pass_clear(sim, h, spec, _family_quat(family, th), touch_eps):
                    label = "feasible" if (family == "y" and th == 0.0) else "rotation_required"
                    return Solution(label, th, family,
                                    _plan(spec, cfg, family, th))
        return Solution("impossible", None, None, None)
    finally:
        sim.set_pose(h.cuboid, *saved)


def _plan(spec: SceneSpec, cfg, family: str, theta_deg: float) -> list[int]:
    """Expert action sequence: rotate in place -> align x/z -> forward through."""
    step = cfg.actions.trans_step
    rot = cfg.actions.rot_step_deg
    acts: list[int] = []
    # 1) rotate in place (start area is free space)
    if family == "x90":
        acts += [A["ROTATE_X_POS"]] * round(90.0 / rot)
    elif family == "z90":
        acts += [A["ROTATE_Z_POS"]] * round(90.0 / rot)
    acts += [A["ROTATE_Y_POS"]] * round(theta_deg / rot)
    # 2) align x then z to the hole center (grid-snapped by the generator)
    dx = spec.hole_center[0] - spec.start_pos[0]
    nx = round(dx / step)
    acts += [A["MOVE_RIGHT"] if nx > 0 else A["MOVE_LEFT"]] * abs(nx)
    dz = spec.hole_center[1] - spec.start_pos[2]
    nz = round(dz / step)
    acts += [A["MOVE_UP"] if nz > 0 else A["MOVE_DOWN"]] * abs(nz)
    # 3) forward until the whole body is past the far wall face
    depth = max(spec.cuboid_dims)          # conservative post-rotation y extent
    travel = (-spec.start_pos[1]) + spec.wall_thickness / 2 + depth / 2 + PASS_MARGIN
    acts += [A["MOVE_FORWARD"]] * math.ceil(travel / step)
    return acts
