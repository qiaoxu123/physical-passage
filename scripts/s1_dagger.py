"""S1 DAgger round: roll out the CURRENT policy, relabel every visited state
with the state-based expert, save as an extra training shard.

This targets exactly the closed-loop failure modes pure BC shows here:
alignment limit cycles (UP/DOWN oscillation) and over-rotation — states the
expert trajectories never visit.

    conda activate habvln
    python scripts/s1_dagger.py --round 1 --episodes 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from physical_passage.agents.expert import ExpertPolicy, env_state
from physical_passage.config import load_config
from physical_passage.envs.passage_env import PassageEnv
from physical_passage.scene.generator import SceneGenerator
from scripts.run_oracle import verified_sample
from scripts.s1_collect_bc_data import IMG_SIZE, _aux_target

DATA_DIR = Path("/data/physical-passage/bc_data")


def _down(rgb):
    return np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE),
                                                  Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    from physical_passage.agents.bc_cnn import BCAgent
    weights = args.weights or str(Path(__file__).resolve().parent.parent
                                  / "results" / "weights" / "bc_policy.pt")
    agent = BCAgent(weights)

    cfg = load_config()
    cfg.actions.max_steps = args.max_steps
    gen = SceneGenerator(cfg, seed=5000 + args.round)
    env = PassageEnv(cfg)

    n = args.episodes
    episodes = (["feasible"] * (n * 2 // 5) + ["rotation_required"] * (n * 2 // 5)
                + ["impossible"] * (n // 5))
    imgs, labels, ep_ids, states, auxs = [], [], [], [], []
    t0, succ = time.time(), 0
    for ep, want in enumerate(episodes):
        spec, sol = verified_sample(gen, cfg, label=want)
        obs, info = env.reset(options={"spec": spec})
        expert = ExpertPolicy(spec, sol, cfg)
        for _ in range(args.max_steps):
            st = env_state(env)
            imgs.append(_down(obs["rgb"]))
            labels.append(expert.act(st[:3], st[3:]))     # expert relabel
            ep_ids.append(10_000 * args.round + ep)
            states.append(st)
            auxs.append(_aux_target(expert, spec, st))
            act = agent.act(obs["rgb"])                   # policy drives
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        succ += bool(info.get("success")) or (
            want == "impossible" and bool(info.get("declared_impossible")))
        if ep % 25 == 0:
            print(f"[{ep:03d}/{len(episodes)}] frames={len(imgs):6d} "
                  f"({time.time()-t0:.0f}s)")

    env.close()
    out = DATA_DIR / f"dagger_r{args.round}.npz"
    np.savez_compressed(out, images=np.stack(imgs).astype(np.uint8),
                        labels=np.array(labels, dtype=np.int64),
                        episodes=np.array(ep_ids, dtype=np.int64),
                        states=np.stack(states).astype(np.float32),
                        aux=np.array(auxs, dtype=np.float32))
    print(f"wrote {out}: {len(imgs)} frames  "
          f"(policy pass rate this round: {succ}/{len(episodes)}, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
