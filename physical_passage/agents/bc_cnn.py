"""S1 behavior cloning: small CNN policy distilled from oracle demonstrations.

PassageCNN: 128x128 RGB (+ optional 7-dim proprioception: pos + quat)
-> 14 action logits (~2.4M params, <1ms on GPU).

The image-only variant learns feasibility and translation alignment but cannot
tell WHEN to stop rotating (5-degree pose deltas are too subtle in the oblique
view); proprioception is the standard robot-policy fix and is what real VLA
stacks feed alongside the camera.

BCAgent: drop-in replacement for QwenAgent (same .act(...) / .last_latency).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..envs.actions import N_ACTIONS

IMG_SIZE = 224
STATE_DIM = 7                       # world pos (3) + orientation quat (4)


class PassageCNN(nn.Module):
    def __init__(self, state_dim: int = 0, in_ch: int = 3) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.in_ch = in_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(),  # 112
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 56
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 28
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(), # 14
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(), # 7
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 * 7 * 7 + state_dim, 256), nn.ReLU())
        self.action_head = nn.Linear(256, N_ACTIONS)
        # auxiliary regression: [dx, dz, remaining_rotation, feasible] — forces
        # the encoder to extract relative geometry from the oblique view
        self.aux_head = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None,
                with_aux: bool = False):
        z = self.conv(x)
        if self.state_dim:
            z = torch.cat([z, s], dim=1)
        h = self.trunk(z)
        if with_aux:
            return self.action_head(h), self.aux_head(h)
        return self.action_head(h)


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWxC (C=3 single view, C=9 stacked multi-view, already at
    IMG_SIZE for C>3) -> normalized 1xCxSxS float tensor."""
    from PIL import Image
    if rgb.shape[2] == 3 and rgb.shape[0] != IMG_SIZE:
        rgb = np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE),
                                                     Image.LANCZOS))
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


class BCAgent:
    def __init__(self, weights: str | Path,
                 device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=self.device, weights_only=True)
        self.state_dim = int(ckpt.get("state_dim", 0))
        self.in_ch = int(ckpt.get("in_ch", 3))
        self.model = PassageCNN(self.state_dim, self.in_ch).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.last_latency = 0.0
        with torch.no_grad():                                   # CUDA warmup
            s = (torch.zeros(1, self.state_dim, device=self.device)
                 if self.state_dim else None)
            self.model(torch.zeros(1, self.in_ch, IMG_SIZE, IMG_SIZE,
                                   device=self.device), s)

    @torch.no_grad()
    def act(self, rgb: np.ndarray, state: np.ndarray | None = None) -> int:
        t0 = time.perf_counter()
        s = None
        if self.state_dim:
            s = torch.from_numpy(np.asarray(state, dtype=np.float32)) \
                     .unsqueeze(0).to(self.device)
        logits = self.model(preprocess(rgb).to(self.device), s)
        act = int(logits.argmax(dim=1).item())
        self.last_latency = time.perf_counter() - t0
        return act
