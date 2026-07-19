"""Render the LoRA fine-tuned Qwen2.5-VL playing one episode per class.

Same 4-view + HUD layout as the other demos.

    conda activate habvln
    python scripts/make_lora_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from physical_passage.config import load_config
from physical_passage.envs.actions import ACTIONS
from physical_passage.envs.passage_env import PassageEnv
from physical_passage.scene.generator import SceneGenerator
from scripts.make_demo import compose, _font
from scripts.run_oracle import verified_sample

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    from physical_passage.agents.vla_qwen import QwenAgent
    agent = QwenAgent(lora_path=str(RESULTS / "weights" / "qwen_lora"))

    cfg = load_config()
    gen = SceneGenerator(cfg, seed=321)        # same fresh levels as the BC demo
    env = PassageEnv(cfg)
    frames: list[Image.Image] = []

    for label in ("feasible", "rotation_required", "impossible"):
        spec, _sol = verified_sample(gen, cfg, label=label)
        obs, _ = env.reset(options={"spec": spec})
        frames.append(compose(env, f"LoRA·{label}", "START", 0, 0.0))
        for _ in range(120):
            act = agent.act(obs["rgb"])
            obs, r, term, trunc, info = env.step(act)
            f = compose(env, f"LoRA·{label}", ACTIONS[act], info["step"],
                        info["minimum_clearance"],
                        declared=bool(info.get("declared_impossible")))
            if info.get("collision"):
                d = ImageDraw.Draw(f)
                d.text((f.width // 2 - 90, 12), "COLLISION", fill=(255, 60, 50),
                       font=_font(30))
            if info.get("success"):
                d = ImageDraw.Draw(f)
                d.text((f.width // 2 - 70, 12), "SUCCESS", fill=(70, 230, 90),
                       font=_font(30))
            frames.append(f)
            if term or trunc:
                break
        frames.extend([frames[-1]] * 8)

    out = RESULTS / "01_demos" / "04_vla_lora_three_classes.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0)
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    strip = np.concatenate([np.array(frames[i]) for i in idx], axis=1)
    Image.fromarray(strip).save(out.with_suffix(".png"))
    env.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
