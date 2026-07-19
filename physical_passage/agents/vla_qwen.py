"""Zero-shot Qwen2.5-VL agent for Physical Passage.

The model sees the main oblique view plus an action-menu instruction and must
answer with exactly one action word per step. This is the honest zero-shot
baseline: no training, no calibration — expected to be weak (the 2D lab showed
free generation collapses; here we still use generation because scoring 14
options per step would need 14 forward passes, and the baseline should reflect
how a VLA is actually promptable out of the box).
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import torch
from PIL import Image

from ..envs.actions import ACTIONS, A

logger = logging.getLogger(__name__)

PROMPT = (
    "You control the GREEN box in this 3D scene. The RED wall has one YELLOW "
    "rectangular opening. Your goal: move and rotate the green box so it passes "
    "through the opening WITHOUT touching the red wall. The passage direction is "
    "FORWARD (into the scene). If the opening is smaller than every side of the "
    "box so it can never fit through, answer DECLARE_IMPOSSIBLE.\n"
    "Answer with exactly ONE of these action words:\n"
    "MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, MOVE_FORWARD, MOVE_BACKWARD, "
    "ROTATE_X_POS, ROTATE_X_NEG, ROTATE_Y_POS, ROTATE_Y_NEG, ROTATE_Z_POS, "
    "ROTATE_Z_NEG, STAY, DECLARE_IMPOSSIBLE.\nAction:"
)

_WORD_RE = re.compile("|".join(sorted((a for a in ACTIONS), key=len, reverse=True)))


class QwenAgent:
    def __init__(self, model_path: str = "/home/xqiao/models/Qwen2.5-VL-3B-Instruct",
                 device: str = "cuda") -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        logger.info("loading %s ...", model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        self.device = device
        self.last_raw = ""
        self.last_latency = 0.0

    @torch.inference_mode()
    def act(self, rgb: np.ndarray) -> int:
        img = Image.fromarray(rgb)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        self.last_latency = time.perf_counter() - t0
        reply = self.processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip().upper()
        self.last_raw = reply
        m = _WORD_RE.search(reply)
        return A[m.group(0)] if m else A["STAY"]
