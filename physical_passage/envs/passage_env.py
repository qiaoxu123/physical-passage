"""Gymnasium environment for Physical Passage (MVP).

Observation: {"rgb": HxWx3 uint8} — the fixed oblique third-person view only.
Action: Discrete(14) — see envs.actions.ACTIONS.
info (per spec): collision, minimum_clearance, time_to_collision,
feasible_ground_truth, passed_wall, plus label/pose extras.

Episode ends on: success (fully past the wall, zero contact ever),
collision (swept), DECLARE_IMPOSSIBLE (+80 if truly impossible else -50),
out-of-bounds, or max_steps truncation.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from ..collision.evaluator import evaluate
from ..collision.swept import swept_check
from ..config import load_config
from ..render.observer import Observer
from ..scene.builder import SceneHandles, build_scene
from ..scene.generator import SceneGenerator
from ..scene.spec import SceneSpec
from ..sim.connection import Sim
from .actions import ACTIONS, A, N_ACTIONS, apply_action


class PassageEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg=None, generator: SceneGenerator | None = None) -> None:
        self.cfg = cfg or load_config()
        self.generator = generator or SceneGenerator(self.cfg)
        self.sim: Sim | None = None
        self.action_space = spaces.Discrete(N_ACTIONS)
        s = self.cfg.render.main_size
        self.observation_space = spaces.Dict(
            {"rgb": spaces.Box(0, 255, (s, s, 3), dtype=np.uint8)})
        self.spec_scene: SceneSpec | None = None
        self.handles: SceneHandles | None = None
        self.observer: Observer | None = None

    # -- helpers -----------------------------------------------------------
    def _fresh_sim(self, spec: SceneSpec) -> None:
        if self.sim is not None:
            self.sim.disconnect()
        self.sim = Sim()
        self.handles = build_scene(self.sim, spec, self.cfg)
        self.observer = Observer(self.sim, self.cfg)

    def _obs(self) -> dict:
        return {"rgb": self.observer.main_view()["rgb"]}

    def _passed_wall(self) -> bool:
        aabb_min, _ = self.sim.client.getAABB(self.handles.cuboid)
        return aabb_min[1] > self.spec_scene.wall_thickness / 2 + 1e-3

    def _out_of_bounds(self, pos) -> bool:
        lo, hi = self.cfg.workspace.bounds_min, self.cfg.workspace.bounds_max
        return any(not (l <= v <= h_) for v, l, h_ in zip(pos, lo, hi))

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        label = (options or {}).get("label")
        spec = (options or {}).get("spec") or self.generator.sample(label=label, seed=seed)
        self.spec_scene = spec
        self._fresh_sim(spec)
        self.steps = 0
        self.contact_ever = False
        self.min_clearance_ep = float("inf")
        info = self._info(collision=False, ttc=1.0)
        return self._obs(), info

    def _info(self, collision: bool, ttc: float) -> dict[str, Any]:
        pos, quat = self.sim.get_pose(self.handles.cuboid)
        c = evaluate(self.sim, self.handles.cuboid, self.handles.wall_parts,
                     self.cfg.collision.touch_eps)
        self.min_clearance_ep = min(self.min_clearance_ep, c.min_clearance)
        return {
            "collision": collision,
            "minimum_clearance": c.min_clearance,
            "time_to_collision": None if ttc >= 1.0 else ttc,
            "feasible_ground_truth": self.spec_scene.label != "impossible",
            "label": self.spec_scene.label,
            "passed_wall": self._passed_wall(),
            "position": tuple(pos),
            "rotation_euler_deg": tuple(math.degrees(a) for a in p.getEulerFromQuaternion(quat)),
            "step": self.steps,
        }

    def step(self, action: int):
        rw = self.cfg.reward
        self.steps += 1
        terminated = truncated = False
        reward = rw.step

        if action == A["DECLARE_IMPOSSIBLE"]:
            correct = self.spec_scene.label == "impossible"
            reward += rw.correct_impossible if correct else rw.wrong_impossible
            info = self._info(collision=False, ttc=1.0)
            info["declared_impossible"] = True
            info["declare_correct"] = correct
            return self._obs(), reward, True, False, info

        pose0 = self.sim.get_pose(self.handles.cuboid)
        pose1 = apply_action(pose0, action, self.cfg.actions.trans_step,
                             self.cfg.actions.rot_step_deg)
        res = swept_check(self.sim, self.handles.cuboid, self.handles.wall_parts,
                          pose0, pose1, self.cfg.collision.ccd_substeps,
                          self.cfg.collision.touch_eps, restore=False)
        collided = res.collided
        if collided:
            self.contact_ever = True
            reward += rw.collision
            terminated = True
        else:
            # forward progress shaping (toward and through the hole)
            reward += rw.progress * (pose1[0][1] - pose0[0][1])

        pos = self.sim.get_pose(self.handles.cuboid)[0]
        if self._out_of_bounds(pos):
            reward += rw.out_of_bounds
            terminated = True

        info = self._info(collision=collided, ttc=res.time_to_collision)
        if not collided and info["passed_wall"] and not self.contact_ever:
            reward += rw.success
            info["success"] = True
            terminated = True
        if self.steps >= self.cfg.actions.max_steps:
            truncated = True
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return self.observer.main_view()["rgb"]

    def close(self):
        if self.sim is not None:
            self.sim.disconnect()
            self.sim = None
