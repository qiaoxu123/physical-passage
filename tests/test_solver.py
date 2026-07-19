"""Feasibility solver tests on handcrafted levels of each class.

Run:  python tests/test_solver.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physical_passage.config import load_config
from physical_passage.scene.builder import build_scene
from physical_passage.scene.spec import SceneSpec
from physical_passage.sim.connection import Sim
from physical_passage.solver.feasibility import solve

CASES = [
    # (name, cuboid dims, hole size, expected label)
    ("wide hole fits identity",      (0.20, 0.12, 0.30), (0.30, 0.42), "feasible"),
    ("exact-ish fit identity",       (0.20, 0.12, 0.30), (0.26, 0.36), "feasible"),
    ("tall obj, wide-short hole",    (0.14, 0.12, 0.40), (0.46, 0.20), "rotation_required"),
    ("wide obj, tall-narrow hole",   (0.36, 0.12, 0.14), (0.20, 0.42), "rotation_required"),
    ("hole smaller than min dim",    (0.20, 0.12, 0.30), (0.10, 0.10), "impossible"),
    ("slot narrower than min dim",   (0.22, 0.16, 0.30), (0.40, 0.12), "impossible"),
]


def main() -> None:
    cfg = load_config()
    ok = 0
    for name, dims, hole, expected in CASES:
        sim = Sim()
        spec = SceneSpec(cuboid_dims=dims, hole_size=hole, hole_center=(0.0, 0.55),
                         start_pos=(0.1, -0.55, 0.5))
        h = build_scene(sim, spec, cfg)
        t0 = time.time()
        sol = solve(sim, h, spec, cfg)
        dt = time.time() - t0
        good = sol.label == expected
        ok += good
        plan_len = len(sol.actions) if sol.actions else 0
        print(f"{'OK' if good else 'XX'} {name:28s} -> {sol.label:18s} "
              f"(theta={sol.theta_deg}, family={sol.family}, plan={plan_len} acts, {dt:.2f}s)")
        sim.disconnect()
    print(f"\n{ok}/{len(CASES)} solver cases correct")
    assert ok == len(CASES)


if __name__ == "__main__":
    main()
