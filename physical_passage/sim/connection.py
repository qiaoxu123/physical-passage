"""PyBullet DIRECT connection with EGL GPU rendering.

Critical init order: connect(DIRECT) FIRST, then load the eglRendererPlugin —
otherwise getCameraImage silently falls back to the slow CPU TinyRenderer.
"""

from __future__ import annotations

import logging
import pkgutil

import pybullet as p
from pybullet_utils.bullet_client import BulletClient

logger = logging.getLogger(__name__)


class Sim:
    """Owns the physics client. All modules receive a Sim, never raw pybullet."""

    def __init__(self, use_egl: bool = True) -> None:
        self.client = BulletClient(connection_mode=p.DIRECT)
        self.egl = False
        if use_egl:
            egl = pkgutil.get_loader("eglRenderer")
            if egl is not None:
                try:
                    plugin = self.client.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                    self.egl = plugin >= 0
                except p.error:  # pragma: no cover - EGL missing on some hosts
                    logger.warning("EGL plugin failed to load; using TinyRenderer")
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if self.egl else p.ER_TINY_RENDERER
        self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # -- pose helpers ------------------------------------------------------
    def set_pose(self, body: int, pos, quat) -> None:
        self.client.resetBasePositionAndOrientation(body, pos, quat)

    def get_pose(self, body: int):
        return self.client.getBasePositionAndOrientation(body)

    def disconnect(self) -> None:
        self.client.disconnect()
