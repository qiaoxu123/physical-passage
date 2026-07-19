"""Closed-loop oracle policy for BC / DAgger data collection.

Fully state-based: reads the TRUE pose every step and recomputes both the
rotation correction (greedy quaternion descent toward the solver's target
orientation — handles over/under-rotation by the learner) and the alignment
moves. This is what lets DAgger relabel arbitrary off-distribution states.
"""

from __future__ import annotations

import math

import numpy as np

from ..envs.actions import A, apply_action
from ..scene.spec import SceneSpec
from ..solver.feasibility import Solution, _family_quat

_ROT_ACTIONS = [A["ROTATE_X_POS"], A["ROTATE_X_NEG"], A["ROTATE_Y_POS"],
                A["ROTATE_Y_NEG"], A["ROTATE_Z_POS"], A["ROTATE_Z_NEG"]]


def env_state(env) -> np.ndarray:
    """Proprioception vector from the live env: cuboid world pos + quat."""
    pos, quat = env.sim.get_pose(env.handles.cuboid)
    return np.array([*pos, *quat], dtype=np.float32)


def _qdist(q1, q2) -> float:
    return 1.0 - abs(sum(a * b for a, b in zip(q1, q2)))


class ExpertPolicy:
    def __init__(self, spec: SceneSpec, sol: Solution, cfg) -> None:
        self.spec, self.cfg = spec, cfg
        self.impossible = sol.label == "impossible"
        if not self.impossible:
            self.q_target = _family_quat(sol.family, sol.theta_deg or 0.0)
            # aligned tighter than half a rotation step counts as "done"
            half = math.radians(cfg.actions.rot_step_deg / 2) / 2
            self.rot_tol = 1.0 - math.cos(half)

    def rot_remaining(self, quat) -> float:
        """Remaining rotation in degrees (for aux supervision)."""
        if self.impossible:
            return 0.0
        d = min(max(1.0 - _qdist(quat, self.q_target), -1.0), 1.0)
        return math.degrees(2 * math.acos(d))

    def act(self, pos, quat) -> int:
        """Expert label for the current state."""
        if self.impossible:
            return A["DECLARE_IMPOSSIBLE"]
        d_cur = _qdist(quat, self.q_target)
        if d_cur > self.rot_tol:
            best, best_d = None, d_cur - 1e-9
            for a in _ROT_ACTIONS:
                _, q2 = apply_action((pos, quat), a, 0.0,
                                     self.cfg.actions.rot_step_deg)
                d = _qdist(q2, self.q_target)
                if d < best_d:
                    best, best_d = a, d
            if best is not None:
                return best
        step = self.cfg.actions.trans_step
        dx = self.spec.hole_center[0] - pos[0]
        if abs(dx) > step / 2:
            return A["MOVE_RIGHT"] if dx > 0 else A["MOVE_LEFT"]
        dz = self.spec.hole_center[1] - pos[2]
        if abs(dz) > step / 2:
            return A["MOVE_UP"] if dz > 0 else A["MOVE_DOWN"]
        return A["MOVE_FORWARD"]
