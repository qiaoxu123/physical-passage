"""S1 data collection: closed-loop expert rollouts with translation-noise
injection and expert relabeling (DAgger-lite in one pass).

Every recorded frame is (main-view RGB downsampled to 128x128, expert action
for that state). Noise actions are executed but never recorded as labels, so
the dataset also covers slightly off-distribution states with correct labels.
Impossible levels wander a few safe steps first — every frame is labeled
DECLARE_IMPOSSIBLE (the declare decision must hold from any viewpoint).

    conda activate physpass
    python scripts/s1_collect_bc_data.py --feasible 80 --rotation 100 --impossible 60
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from physical_passage.agents.expert import ExpertPolicy, env_state
from physical_passage.config import load_config
from physical_passage.envs.actions import A
from physical_passage.envs.passage_env import PassageEnv
from physical_passage.scene.generator import SceneGenerator
from scripts.run_oracle import verified_sample

DATA_DIR = Path("/data/physical-passage/bc_data")
IMG_SIZE = 224
NOISE_P = 0.15
WANDER_STEPS = 6                     # impossible levels: safe random walk length
NOISE_SET = [A["MOVE_LEFT"], A["MOVE_RIGHT"], A["MOVE_UP"], A["MOVE_DOWN"],
             A["MOVE_BACKWARD"]]


def _aux_target(expert: ExpertPolicy, spec, st) -> list[float]:
    """[dx, dz, remaining rotation (normalized), feasible] for aux supervision."""
    dx = spec.hole_center[0] - st[0]
    dz = spec.hole_center[1] - st[2]
    if expert.impossible:
        return [dx, dz, 0.0, 0.0]
    return [dx, dz, expert.rot_remaining(st[3:]) / 90.0, 1.0]


def _downsample(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE),
                                                  Image.LANCZOS))


def mv_frame(env: PassageEnv, obs_rgb: np.ndarray) -> np.ndarray:
    """9-channel multi-view frame: main + front + side, each 224x224.

    Perception-attribution ablation: the front view exposes dx/dz directly and
    the side view exposes the rotation profile — if control succeeds with these
    added, the single-oblique-view perception floor is confirmed as the wall.
    """
    aux = env.observer.aux_views(depth=False, seg=False)
    return np.concatenate([_downsample(obs_rgb),
                           _downsample(aux["front"]["rgb"]),
                           _downsample(aux["side"]["rgb"])], axis=2)


def _safe_noise(env: PassageEnv, rng: random.Random) -> int | None:
    """A random translation that stays in bounds and clear of the wall."""
    aabb_min, aabb_max = env.sim.client.getAABB(env.handles.cuboid)
    if aabb_max[1] > -env.spec_scene.wall_thickness / 2 - 0.03:
        return None                                    # too close to the wall
    pos = env.sim.get_pose(env.handles.cuboid)[0]
    step = env.cfg.actions.trans_step
    lo, hi = env.cfg.workspace.bounds_min, env.cfg.workspace.bounds_max
    for act in rng.sample(NOISE_SET, len(NOISE_SET)):
        d = {A["MOVE_LEFT"]: (-step, 0, 0), A["MOVE_RIGHT"]: (step, 0, 0),
             A["MOVE_UP"]: (0, 0, step), A["MOVE_DOWN"]: (0, 0, -step),
             A["MOVE_BACKWARD"]: (0, -step, 0)}[act]
        new = [p + di for p, di in zip(pos, d)]
        if all(l + 0.05 <= v <= h - 0.05 for v, l, h in zip(new, lo, hi)):
            return act
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasible", type=int, default=80)
    ap.add_argument("--rotation", type=int, default=100)
    ap.add_argument("--impossible", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--multi", action="store_true",
                    help="store 9-channel main+front+side frames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(DATA_DIR / ("train_mv.npz" if args.multi else "train.npz"))

    cfg = load_config()
    gen = SceneGenerator(cfg, seed=args.seed)
    env = PassageEnv(cfg)
    rng = random.Random(args.seed)

    episodes = (["feasible"] * args.feasible + ["rotation_required"] * args.rotation
                + ["impossible"] * args.impossible)
    rng.shuffle(episodes)

    imgs, labels, ep_ids, states, auxs = [], [], [], [], []
    frame_of = (lambda o: mv_frame(env, o)) if args.multi else _downsample
    t0, n_noise = time.time(), 0
    for ep, want in enumerate(episodes):
        spec, sol = verified_sample(gen, cfg, label=want)
        obs, info = env.reset(options={"spec": spec})
        expert = ExpertPolicy(spec, sol, cfg)

        if expert.impossible:
            for _ in range(WANDER_STEPS):
                imgs.append(frame_of(obs["rgb"]))
                labels.append(A["DECLARE_IMPOSSIBLE"])
                ep_ids.append(ep)
                states.append(env_state(env))
                auxs.append(_aux_target(expert, spec, states[-1]))
                act = _safe_noise(env, rng)
                if act is None:
                    break
                obs, _, term, trunc, info = env.step(act)
                if term or trunc:
                    break
            imgs.append(frame_of(obs["rgb"]))
            labels.append(A["DECLARE_IMPOSSIBLE"])
            ep_ids.append(ep)
            states.append(env_state(env))
            auxs.append(_aux_target(expert, spec, states[-1]))
            env.step(A["DECLARE_IMPOSSIBLE"])
            continue

        for _ in range(cfg.actions.max_steps):
            st = env_state(env)
            label = expert.act(st[:3], st[3:])
            imgs.append(frame_of(obs["rgb"]))
            labels.append(label)
            ep_ids.append(ep)
            states.append(st)
            auxs.append(_aux_target(expert, spec, st))
            act = label
            if rng.random() < NOISE_P:
                noise = _safe_noise(env, rng)
                if noise is not None:
                    act, n_noise = noise, n_noise + 1
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        ok = bool(info.get("success"))
        if not ok:
            print(f"  WARN ep{ep} ({want}) did not succeed: "
                  f"collision={info.get('collision')} steps={info['step']}")
        if ep % 20 == 0:
            print(f"[{ep:03d}/{len(episodes)}] {want:18s} frames={len(imgs):6d} "
                  f"({time.time()-t0:.0f}s)")

    env.close()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    x = np.stack(imgs).astype(np.uint8)
    y = np.array(labels, dtype=np.int64)
    e = np.array(ep_ids, dtype=np.int64)
    np.savez_compressed(out, images=x, labels=y, episodes=e,
                        states=np.stack(states).astype(np.float32),
                        aux=np.array(auxs, dtype=np.float32))
    binc = np.bincount(y, minlength=14)
    print(f"\nwrote {out}: {x.shape} frames, {len(episodes)} episodes, "
          f"{n_noise} noise steps, {time.time()-t0:.0f}s")
    print("label counts:", dict(enumerate(binc.tolist())))


if __name__ == "__main__":
    main()
