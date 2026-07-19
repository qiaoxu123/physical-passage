"""Build PyBullet bodies for a SceneSpec.

Wall-with-hole = 4 static red boxes (left/right/bottom/top frame) so every
collision pair stays convex-convex (fast, exact GJK). The yellow hole rim and
the gray floor are visual-only bodies (no collision shape). The green cuboid is
kinematic (mass 0) and is moved exclusively via resetBasePositionAndOrientation;
we never call stepSimulation — only performCollisionDetection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pybullet as p

from ..sim.connection import Sim
from .spec import SceneSpec

RED = (0.86, 0.24, 0.20, 1.0)
GREEN = (0.24, 0.82, 0.35, 1.0)
YELLOW = (0.98, 0.94, 0.16, 1.0)
GRAY = (0.35, 0.35, 0.38, 1.0)
RIM = 0.012          # thickness of the yellow visual rim strips


@dataclass
class SceneHandles:
    cuboid: int
    wall_parts: list[int]     # the 4 collidable frame boxes
    visual_only: list[int]    # rim strips + floor (no collision)


def _static_box(sim: Sim, half, pos, color, collidable: bool) -> int:
    vs = sim.client.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(color))
    cs = sim.client.createCollisionShape(p.GEOM_BOX, halfExtents=list(half)) if collidable else -1
    return sim.client.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs,
                                      baseVisualShapeIndex=vs, basePosition=list(pos))


def build_scene(sim: Sim, spec: SceneSpec, cfg) -> SceneHandles:
    wx = cfg.wall.x_half
    z0, z1 = cfg.wall.z_min, cfg.wall.z_max
    t2 = spec.wall_thickness / 2
    hx, hz = spec.hole_center
    hw2, hh2 = spec.hole_size[0] / 2, spec.hole_size[1] / 2

    wall_parts = []
    # left / right frame boxes (full wall height)
    lw = (hx - hw2) - (-wx)
    if lw > 1e-6:
        wall_parts.append(_static_box(sim, ((lw) / 2, t2, (z1 - z0) / 2),
                                      (-wx + lw / 2, 0, (z0 + z1) / 2), RED, True))
    rw = wx - (hx + hw2)
    if rw > 1e-6:
        wall_parts.append(_static_box(sim, (rw / 2, t2, (z1 - z0) / 2),
                                      (hx + hw2 + rw / 2, 0, (z0 + z1) / 2), RED, True))
    # bottom / top frame boxes (hole-width columns)
    bh = (hz - hh2) - z0
    if bh > 1e-6:
        wall_parts.append(_static_box(sim, (hw2, t2, bh / 2),
                                      (hx, 0, z0 + bh / 2), RED, True))
    th = z1 - (hz + hh2)
    if th > 1e-6:
        wall_parts.append(_static_box(sim, (hw2, t2, th / 2),
                                      (hx, 0, hz + hh2 + th / 2), RED, True))

    # yellow rim (visual only) marking the passable opening
    visual = [
        _static_box(sim, (hw2, t2 + 0.002, RIM / 2), (hx, 0, hz - hh2 + RIM / 2), YELLOW, False),
        _static_box(sim, (hw2, t2 + 0.002, RIM / 2), (hx, 0, hz + hh2 - RIM / 2), YELLOW, False),
        _static_box(sim, (RIM / 2, t2 + 0.002, hh2), (hx - hw2 + RIM / 2, 0, hz), YELLOW, False),
        _static_box(sim, (RIM / 2, t2 + 0.002, hh2), (hx + hw2 - RIM / 2, 0, hz), YELLOW, False),
        # floor plane for depth cue
        _static_box(sim, (wx, 1.0, 0.005), (0, 0, cfg.wall.z_min - 0.005), GRAY, False),
    ]

    # green controllable cuboid (kinematic)
    d = spec.cuboid_dims
    cuboid = _static_box(sim, (d[0] / 2, d[1] / 2, d[2] / 2), spec.start_pos, GREEN, True)
    quat = p.getQuaternionFromEuler([r * 3.14159265 / 180 for r in spec.start_rpy_deg])
    sim.set_pose(cuboid, spec.start_pos, quat)
    return SceneHandles(cuboid=cuboid, wall_parts=wall_parts, visual_only=visual)
