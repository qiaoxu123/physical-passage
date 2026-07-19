"""Zero-shot VLA baseline evaluation.

Same verified levels and metrics as run_oracle.py, but the player is
Qwen2.5-VL-3B looking at the rendered main view.

    python scripts/run_vla.py --episodes 12 --max-steps 60
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physical_passage.config import load_config
from physical_passage.envs.actions import ACTIONS
from physical_passage.envs.passage_env import PassageEnv
from physical_passage.logging.logger import EpisodeLogger
from physical_passage.metrics.offline import export
from physical_passage.scene.generator import SceneGenerator
from scripts.run_oracle import verified_sample

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--out", default=str(RESULTS / "vla_logs"))
    args = ap.parse_args()

    from physical_passage.agents.vla_qwen import QwenAgent
    agent = QwenAgent()

    cfg = load_config()
    cfg.actions.max_steps = args.max_steps
    gen = SceneGenerator(cfg, seed=42)          # same seed -> same levels as oracle
    env = PassageEnv(cfg)
    logger = EpisodeLogger(args.out)

    lat_all: list[float] = []
    t0 = time.time()
    for ep in range(args.episodes):
        spec, _sol = verified_sample(gen, cfg)
        obs, info = env.reset(options={"spec": spec})
        total_r = 0.0
        act_counts: collections.Counter = collections.Counter()
        for step_i in range(args.max_steps):
            act = agent.act(obs["rgb"])
            act_counts[ACTIONS[act]] += 1
            lat_all.append(agent.last_latency)
            obs, r, term, trunc, info = env.step(act)
            total_r += r
            logger.log_step(ep, step_i, ACTIONS[act], r, info)
            if term or trunc:
                break
        outcome = {
            "label_solver": spec.label,
            "success": bool(info.get("success")),
            "collision": bool(info.get("collision")),
            "declared_impossible": bool(info.get("declared_impossible")),
            "steps": info["step"],
            "min_clearance": env.min_clearance_ep,
            "total_reward": round(total_r, 2),
        }
        logger.log_episode(ep, spec.to_dict(), outcome)
        top = ", ".join(f"{a}x{c}" for a, c in act_counts.most_common(3))
        tag = "PASS" if (outcome["success"] or
                         (spec.label == "impossible" and outcome["declared_impossible"])) else "FAIL"
        print(f"[{ep:03d}] {tag} {spec.label:18s} steps={outcome['steps']:3d} "
              f"reward={outcome['total_reward']:8.2f} actions: {top}")

    logger.close()
    env.close()
    m = export(args.out, RESULTS / "metrics_vla")
    import statistics
    print(f"\nVLA zero-shot ({time.time()-t0:.0f}s, mean step latency "
          f"{statistics.fmean(lat_all)*1000:.0f}ms):")
    print(f"success_rate={m['success_rate']}  collision_rate={m['collision_rate']}  "
          f"feasibility_accuracy={m['feasibility_accuracy']}  "
          f"impossible_recall={m['impossible_recall']}  "
          f"false_impossible_rate={m['false_impossible_rate']}")


if __name__ == "__main__":
    main()
