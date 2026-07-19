"""Discrete action space (MVP): 6 translations + 6 rotations + STAY + DECLARE_IMPOSSIBLE."""

from __future__ import annotations

import math

import pybullet as p

ACTIONS: list[str] = [
    "MOVE_LEFT", "MOVE_RIGHT",        # -x / +x
    "MOVE_DOWN", "MOVE_UP",           # -z / +z
    "MOVE_FORWARD", "MOVE_BACKWARD",  # +y / -y (passage direction is +y)
    "ROTATE_X_POS", "ROTATE_X_NEG",
    "ROTATE_Y_POS", "ROTATE_Y_NEG",
    "ROTATE_Z_POS", "ROTATE_Z_NEG",
    "STAY",
    "DECLARE_IMPOSSIBLE",
]
A = {name: i for i, name in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

_TRANS = {
    A["MOVE_LEFT"]: (-1, 0, 0), A["MOVE_RIGHT"]: (1, 0, 0),
    A["MOVE_DOWN"]: (0, 0, -1), A["MOVE_UP"]: (0, 0, 1),
    A["MOVE_FORWARD"]: (0, 1, 0), A["MOVE_BACKWARD"]: (0, -1, 0),
}
_ROT_AXIS = {
    A["ROTATE_X_POS"]: ((1, 0, 0), 1), A["ROTATE_X_NEG"]: ((1, 0, 0), -1),
    A["ROTATE_Y_POS"]: ((0, 1, 0), 1), A["ROTATE_Y_NEG"]: ((0, 1, 0), -1),
    A["ROTATE_Z_POS"]: ((0, 0, 1), 1), A["ROTATE_Z_NEG"]: ((0, 0, 1), -1),
}


def apply_action(pose, action: int, trans_step: float, rot_step_deg: float):
    """Pure function: (pos, quat), action -> target (pos, quat). STAY/DECLARE = no-op."""
    pos, quat = pose
    if action in _TRANS:
        d = _TRANS[action]
        return (tuple(pi + di * trans_step for pi, di in zip(pos, d)), quat)
    if action in _ROT_AXIS:
        axis, sign = _ROT_AXIS[action]
        half = math.radians(sign * rot_step_deg) / 2
        dq = (axis[0] * math.sin(half), axis[1] * math.sin(half),
              axis[2] * math.sin(half), math.cos(half))
        # world-frame rotation: q_new = dq * q
        return (pos, tuple(p.multiplyTransforms((0, 0, 0), dq, (0, 0, 0), quat)[1]))
    return (pos, quat)
