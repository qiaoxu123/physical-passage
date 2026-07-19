"""Camera rig: one fixed oblique third-person view (model input) plus three
orthographic-ish label views (front/side/top) used only for supervision.

Per the project's camera policy: fix the main viewpoint first to validate
physical reasoning; viewpoint randomization comes later to rule out template
memorization. Aux views are never fed to the model.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from ..sim.connection import Sim


class Observer:
    def __init__(self, sim: Sim, cfg) -> None:
        self.sim = sim
        self.cfg = cfg
        c = cfg.render.main_cam
        self._main_view = p.computeViewMatrixFromYawPitchRoll(
            list(c.target), c.distance, c.yaw, c.pitch, 0, upAxisIndex=2)
        self._main_near, self._main_far = 0.05, 6.0
        self._main_proj = p.computeProjectionMatrixFOV(c.fov, 1.0, self._main_near, self._main_far)
        # aux views: near-orthographic = far camera + narrow FOV (the EGL plugin
        # does not honor true orthographic projection matrices)
        zc = (cfg.wall.z_min + cfg.wall.z_max) / 2
        dist, fov = 20.0, 5.2          # half-extent ~ tan(fov/2)*dist ~ 0.91 m
        self._aux_near, self._aux_far = dist - 3.0, dist + 3.0
        self._ortho_proj = p.computeProjectionMatrixFOV(fov, 1.0, self._aux_near, self._aux_far)
        self._aux_views = {
            "front": p.computeViewMatrix([0, -dist, zc], [0, 0, zc], [0, 0, 1]),
            "side": p.computeViewMatrix([dist, 0, zc], [0, 0, zc], [0, 0, 1]),
            "top": p.computeViewMatrix([0, 0, zc + dist], [0, 0, zc], [0, 1, 0]),
        }

    BG = (18, 18, 18)  # dark background per the minimal research style

    def _shot(self, size: int, view, proj, near: float, far: float,
              want_depth: bool, want_seg: bool):
        # always request the segmentation mask: it doubles as the background
        # matte so we can paint the empty sky dark
        w, h, rgb, depth, seg = self.sim.client.getCameraImage(
            size, size, view, proj, renderer=self.sim.renderer,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        rgb = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
        seg_raw = np.reshape(seg, (h, w))
        rgb[seg_raw < 0] = self.BG                       # -1 = no object hit
        out = {"rgb": rgb}
        if want_depth:
            zbuf = np.reshape(depth, (h, w))
            out["depth"] = (far * near / (far - (far - near) * zbuf)).astype(np.float32)
        if want_seg:
            out["seg"] = seg_raw.astype(np.int32)
        return out

    def main_view(self, depth: bool = False, seg: bool = False) -> dict:
        """The model-input view (448x448 by default)."""
        return self._shot(self.cfg.render.main_size, self._main_view,
                          self._main_proj, self._main_near, self._main_far,
                          depth, seg)

    def aux_views(self, depth: bool = True, seg: bool = True) -> dict[str, dict]:
        """Label-only near-orthographic views (front/side/top)."""
        return {name: self._shot(self.cfg.render.aux_size, view,
                                 self._ortho_proj, self._aux_near, self._aux_far,
                                 depth, seg)
                for name, view in self._aux_views.items()}
