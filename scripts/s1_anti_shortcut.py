"""Anti-shortcut probes for the learned feasibility decision (spec section 9).

A CNN can pass in-distribution tests via shortcuts (color, outline, fixed
camera). Each probe below perturbs one such cue; ground-truth labels are
re-verified with the collision-engine solver. The probed decision is the
policy's FIRST action: DECLARE_IMPOSSIBLE vs. attempt.

Probes:
  baseline        in-distribution control
  size-large      all geometry scaled x1.35 (beyond generator ranges)
  size-small      all geometry scaled x0.70
  camera-jitter   main camera yaw +12, pitch -10, distance +0.35
  color-swap      green object <-> red wall (yellow rim unchanged)
  near-threshold  deceptive holes: impossible misses by 1.5 cm, feasible
                  clears by only ~2 cm (both far tighter than training)

Not covered here (needs Level-2 scene work): truly novel shapes (L-bodies,
cylinders), equal-projection-different-depth, alternate mid-trajectories.

    conda activate habvln
    python scripts/s1_anti_shortcut.py --per-side 20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import physical_passage.scene.builder as builder
from physical_passage.config import load_config
from physical_passage.envs.actions import A
from physical_passage.envs.passage_env import PassageEnv
from physical_passage.scene.builder import build_scene
from physical_passage.scene.generator import SceneGenerator
from physical_passage.sim.connection import Sim
from physical_passage.solver.feasibility import solve

RESULTS = Path(__file__).resolve().parent.parent / "results"


def verify_label(spec, cfg) -> str:
    sim = Sim(use_egl=False)
    h = build_scene(sim, spec, cfg)
    sol = solve(sim, h, spec, cfg)
    sim.disconnect()
    return sol.label


def sample_verified(gen, cfg, label, transform=None, tries=40):
    """Sample -> optional geometry transform -> solver-verified label."""
    for _ in range(tries):
        spec = gen.sample(label=label)
        if transform is not None:
            spec = transform(spec)
        got = verify_label(spec, cfg)
        want_possible = label != "impossible"
        if (got != "impossible") == want_possible:
            return spec
    raise RuntimeError(f"could not build verified '{label}' probe level")


# ---- geometry transforms -------------------------------------------------

def scale_spec(k: float):
    def tf(spec):
        d = spec.cuboid_dims
        return dataclasses.replace(
            spec,
            cuboid_dims=(d[0] * k, d[1] * k, d[2] * k),
            hole_size=(spec.hole_size[0] * k, spec.hole_size[1] * k))
    return tf


def near_threshold(rng):
    def tf(spec):
        d = spec.cuboid_dims
        if spec.label == "impossible":
            # hole misses the smallest dimension by only 1.5 cm
            w = max(0.05, min(d) - 0.015)
            return dataclasses.replace(spec, hole_size=(w, d[2] + 0.10))
        # feasible but with only ~2 cm total margin (training used >= 6 cm)
        return dataclasses.replace(spec, hole_size=(d[0] + 0.04, d[2] + 0.04),
                                   label="feasible")
    return tf


# ---- probe runner --------------------------------------------------------

def run_probe(agent, cfg, seed, per_side, transform=None, cam=None,
              swap_colors=False):
    if swap_colors:
        builder.GREEN, builder.RED = ((0.86, 0.24, 0.20, 1.0),
                                      (0.24, 0.82, 0.35, 1.0))
    if cam:
        for k, v in cam.items():
            setattr(cfg.render.main_cam, k,
                    getattr(cfg.render.main_cam, k) + v)
    try:
        gen = SceneGenerator(cfg, seed=seed)
        env = PassageEnv(cfg)
        stats = {"imp_declared": 0, "imp_total": 0,
                 "pos_declared": 0, "pos_total": 0}
        labels = (["impossible"] * per_side
                  + ["feasible"] * (per_side // 2)
                  + ["rotation_required"] * (per_side - per_side // 2))
        for want in labels:
            spec = sample_verified(gen, cfg, want, transform)
            obs, _ = env.reset(options={"spec": spec})
            declared = agent.act(obs["rgb"]) == A["DECLARE_IMPOSSIBLE"]
            if want == "impossible":
                stats["imp_total"] += 1
                stats["imp_declared"] += declared
            else:
                stats["pos_total"] += 1
                stats["pos_declared"] += declared
        env.close()
        return stats
    finally:
        if swap_colors:
            builder.GREEN = (0.24, 0.82, 0.35, 1.0)
            builder.RED = (0.86, 0.24, 0.20, 1.0)
        if cam:
            for k, v in cam.items():
                setattr(cfg.render.main_cam, k,
                        getattr(cfg.render.main_cam, k) - v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-side", type=int, default=20)
    ap.add_argument("--weights", default=str(RESULTS / "weights" / "bc_policy.pt"))
    args = ap.parse_args()

    from physical_passage.agents.bc_cnn import BCAgent
    agent = BCAgent(args.weights)
    cfg = load_config()
    rng = np.random.default_rng(9)

    probes = [
        ("baseline", dict()),
        ("size-large-x1.35", dict(transform=scale_spec(1.35))),
        ("size-small-x0.70", dict(transform=scale_spec(0.70))),
        ("camera-jitter", dict(cam={"yaw": 12.0, "pitch": -10.0,
                                    "distance": 0.35})),
        ("color-swap", dict(swap_colors=True)),
        ("near-threshold", dict(transform=near_threshold(rng))),
    ]
    out = {}
    print(f"{'probe':20s} {'imp recall':>11s} {'false declare':>14s} "
          f"{'feas acc':>9s}")
    for i, (name, kw) in enumerate(probes):
        t0 = time.time()
        s = run_probe(agent, cfg, seed=7000 + i, per_side=args.per_side, **kw)
        recall = s["imp_declared"] / max(s["imp_total"], 1)
        false_d = s["pos_declared"] / max(s["pos_total"], 1)
        acc = (s["imp_declared"] + s["pos_total"] - s["pos_declared"]) / (
            s["imp_total"] + s["pos_total"])
        out[name] = {"impossible_recall": round(recall, 3),
                     "false_declare_rate": round(false_d, 3),
                     "feasibility_accuracy": round(acc, 3), **s}
        print(f"{name:20s} {s['imp_declared']:>4d}/{s['imp_total']:<4d}   "
              f"{s['pos_declared']:>4d}/{s['pos_total']:<4d}      "
              f"{acc:>7.3f}   ({time.time()-t0:.0f}s)")

    path = RESULTS / "metrics_anti_shortcut.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
