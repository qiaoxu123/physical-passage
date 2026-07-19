"""Discrete collision evaluation at a single pose.

minimum_clearance is the smallest signed distance between the cuboid and any
wall frame box (negative = penetration depth). PyBullet inflates boxes by a
small collision margin (~1e-3), so treat clearances below touch_eps as contact.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..sim.connection import Sim

QUERY_DIST = 0.5  # ignore pairs farther than this


@dataclass
class ContactInfo:
    min_clearance: float
    collision: bool
    point: tuple | None      # closest/contact point on the wall
    normal: tuple | None     # contact normal (from wall toward cuboid)


def evaluate(sim: Sim, cuboid: int, wall_parts: list[int],
             touch_eps: float = 1e-4) -> ContactInfo:
    sim.client.performCollisionDetection()
    dmin, pt, nrm = QUERY_DIST, None, None
    for wid in wall_parts:
        for c in sim.client.getClosestPoints(cuboid, wid, distance=QUERY_DIST):
            d = c[8]  # contactDistance, negative = penetration
            if d < dmin:
                dmin, pt, nrm = d, c[6], c[7]
    return ContactInfo(min_clearance=dmin, collision=dmin <= touch_eps,
                       point=pt, normal=nrm)
