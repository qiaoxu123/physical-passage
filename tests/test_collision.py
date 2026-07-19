"""Collision unit tests: signed clearance signs + swept tunneling catch.

Run:  python tests/test_collision.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybullet as p

from physical_passage.config import load_config
from physical_passage.collision.evaluator import evaluate
from physical_passage.collision.swept import swept_check
from physical_passage.scene.builder import build_scene
from physical_passage.scene.spec import SceneSpec
from physical_passage.sim.connection import Sim

IDENT = (0.0, 0.0, 0.0, 1.0)


def main() -> None:
    cfg = load_config()
    sim = Sim()
    # hole 0.30 x 0.42 centered (0, 0.55); cuboid 0.20 x 0.12 x 0.34
    spec = SceneSpec()
    h = build_scene(sim, spec, cfg)
    hx, hz = spec.hole_center

    # 1) safe pose: aligned with hole center, far in front of the wall
    sim.set_pose(h.cuboid, (hx, -0.5, hz), IDENT)
    safe = evaluate(sim, h.cuboid, h.wall_parts)
    assert not safe.collision and safe.min_clearance > 0.3, safe
    print(f"OK safe pose        clearance={safe.min_clearance:+.4f}")

    # 2) penetrating pose: centered inside the wall plane but offset into frame
    sim.set_pose(h.cuboid, (hx + 0.25, 0.0, hz), IDENT)
    pen = evaluate(sim, h.cuboid, h.wall_parts)
    assert pen.collision and pen.min_clearance < 0, pen
    print(f"OK penetrating pose clearance={pen.min_clearance:+.4f}")

    # 3) inside-the-hole pose (aligned): should be clear with small margin
    sim.set_pose(h.cuboid, (hx, 0.0, hz), IDENT)
    ing = evaluate(sim, h.cuboid, h.wall_parts)
    assert not ing.collision, ing
    print(f"OK in-hole pose     clearance={ing.min_clearance:+.4f} (expect ~{(spec.hole_size[0]-spec.cuboid_dims[0])/2:.3f})")

    # 4) tunneling: jump across the wall through solid frame in ONE big motion;
    #    endpoints clear, midpoints collide -> swept must catch it
    p0 = ((hx + 0.35, -0.5, hz), IDENT)
    p1 = ((hx + 0.35, +0.5, hz), IDENT)
    for pose in (p0, p1):
        sim.set_pose(h.cuboid, *pose)
        assert not evaluate(sim, h.cuboid, h.wall_parts).collision
    res = swept_check(sim, h.cuboid, h.wall_parts, p0, p1, substeps=16)
    assert res.collided and res.time_to_collision < 1.0, res
    print(f"OK tunneling caught ttc={res.time_to_collision:.2f} min_clear={res.min_clearance:+.4f}")

    # 5) clean pass through the hole: swept straight line must be collision-free
    res2 = swept_check(sim, h.cuboid, h.wall_parts,
                       ((hx, -0.5, hz), IDENT), ((hx, +0.5, hz), IDENT), substeps=60)
    assert not res2.collided, res2
    print(f"OK clean pass       min_clear={res2.min_clearance:+.4f}")

    # 6) rotation swept: rotating 90 deg about y while INSIDE the hole should
    #    hit the frame (diagonal exceeds hole width) even if endpoints differ
    q90 = p.getQuaternionFromEuler([0, 1.5707963, 0])
    res3 = swept_check(sim, h.cuboid, h.wall_parts,
                       ((hx, 0.0, hz), IDENT), ((hx, 0.0, hz), q90), substeps=18)
    assert res3.collided, res3
    print(f"OK rotation sweep hits frame  ttc={res3.time_to_collision:.2f}")

    print("\nALL COLLISION TESTS PASSED")


if __name__ == "__main__":
    main()
