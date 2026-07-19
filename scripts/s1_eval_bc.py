"""S1 closed-loop evaluation: BC policy on FRESH levels (10/10/10 per class).

Same metrics as run_oracle.py / run_vla.py for a three-way comparison.

    conda activate habvln
    python scripts/s1_eval_bc.py --per-class 10
"""

from __future__ import annotations

import argparse
import collections
import shutil
import statistics
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
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)       # unseen by training
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--weights", default=str(RESULTS / "weights" / "bc_policy.pt"))
    ap.add_argument("--out", default=str(RESULTS / "bc_logs"))
    args = ap.parse_args()

    from physical_passage.agents.bc_cnn import BCAgent
    from physical_passage.agents.expert import env_state
    from scripts.s1_collect_bc_data import mv_frame
    agent = BCAgent(args.weights)

    cfg = load_config()
    cfg.actions.max_steps = args.max_steps
    gen = SceneGenerator(cfg, seed=args.seed)
    env = PassageEnv(cfg)
    shutil.rmtree(args.out, ignore_errors=True)     # metrics read the whole dir
    logger = EpisodeLogger(args.out)

    episodes = (["feasible"] * args.per_class
                + ["rotation_required"] * args.per_class
                + ["impossible"] * args.per_class)
    lat: list[float] = []
    t0 = time.time()
    for ep, want in enumerate(episodes):
        spec, _sol = verified_sample(gen, cfg, label=want)
        obs, info = env.reset(options={"spec": spec})
        total_r = 0.0
        act_counts: collections.Counter = collections.Counter()
        for step_i in range(args.max_steps):
            frame = mv_frame(env, obs["rgb"]) if agent.in_ch == 9 else obs["rgb"]
            act = agent.act(frame, env_state(env))
            act_counts[ACTIONS[act]] += 1
            lat.append(agent.last_latency)
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
    stem = Path(args.out).name
    stem = stem[:-5] if stem.endswith("_logs") else stem
    m = export(args.out, RESULTS / f"metrics_{stem}")
    print(f"\nBC-CNN closed-loop ({time.time()-t0:.0f}s, mean step latency "
          f"{statistics.fmean(lat)*1000:.2f}ms):")
    print(f"success_rate={m['success_rate']}  collision_rate={m['collision_rate']}  "
          f"feasibility_accuracy={m['feasibility_accuracy']}  "
          f"impossible_recall={m['impossible_recall']}  "
          f"false_impossible_rate={m['false_impossible_rate']}")


if __name__ == "__main__":
    main()
