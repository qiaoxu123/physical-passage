"""Oracle expert: solve each level offline, execute the plan in the env.

Validates the whole MVP loop: generator -> solver labels -> plan execution with
swept collision -> success predicate -> logging -> metrics.

    python scripts/run_oracle.py --episodes 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physical_passage.config import load_config
from physical_passage.envs.actions import ACTIONS, A
from physical_passage.envs.passage_env import PassageEnv
from physical_passage.logging.logger import EpisodeLogger
from physical_passage.metrics.offline import export
from physical_passage.scene.builder import build_scene
from physical_passage.scene.generator import SceneGenerator
from physical_passage.sim.connection import Sim
from physical_passage.solver.feasibility import solve

RESULTS = Path(__file__).resolve().parent.parent / "results"


def verified_sample(gen: SceneGenerator, cfg, label: str | None = None):
    """Sample a spec and verify its label with the solver (reject/resample)."""
    for _ in range(20):
        spec = gen.sample(label=label)
        sim = Sim(use_egl=False)          # headless label check, no rendering
        h = build_scene(sim, spec, cfg)
        sol = solve(sim, h, spec, cfg)
        sim.disconnect()
        if sol.label == spec.label:
            return spec, sol
    raise RuntimeError(f"could not construct verified '{label}' level in 20 tries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default=str(RESULTS / "oracle_logs"))
    args = ap.parse_args()

    cfg = load_config()
    gen = SceneGenerator(cfg, seed=42)
    env = PassageEnv(cfg)
    logger = EpisodeLogger(args.out)

    counts = {"feasible": 0, "rotation_required": 0, "impossible": 0}
    t0 = time.time()
    for ep in range(args.episodes):
        spec, sol = verified_sample(gen, cfg)
        counts[spec.label] += 1
        obs, info = env.reset(options={"spec": spec})
        plan = sol.actions if sol.actions else [A["DECLARE_IMPOSSIBLE"]]

        total_r, outcome = 0.0, {}
        for step_i, act in enumerate(plan):
            obs, r, term, trunc, info = env.step(act)
            total_r += r
            logger.log_step(ep, step_i, ACTIONS[act], r, info)
            if term or trunc:
                break
        outcome = {
            "label_solver": sol.label,
            "success": bool(info.get("success")),
            "collision": bool(info.get("collision")),
            "declared_impossible": bool(info.get("declared_impossible")),
            "steps": info["step"],
            "min_clearance": env.min_clearance_ep,
            "total_reward": round(total_r, 2),
        }
        logger.log_episode(ep, spec.to_dict(), outcome)
        tag = "PASS" if (outcome["success"] or
                         (spec.label == "impossible" and outcome["declared_impossible"])) else "FAIL"
        print(f"[{ep:03d}] {tag} {spec.label:18s} steps={outcome['steps']:3d} "
              f"reward={outcome['total_reward']:7.2f} clear={outcome['min_clearance']:+.3f}")

    logger.close()
    env.close()
    m = export(args.out, RESULTS / "metrics")
    print(f"\nlevel mix: {counts}   ({time.time()-t0:.1f}s)")
    print(f"success_rate={m['success_rate']}  collision_rate={m['collision_rate']}  "
          f"feasibility_accuracy={m['feasibility_accuracy']}  "
          f"impossible_recall={m['impossible_recall']}  "
          f"false_impossible_rate={m['false_impossible_rate']}")


if __name__ == "__main__":
    main()
