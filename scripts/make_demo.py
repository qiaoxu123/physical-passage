"""Render oracle episodes (one per class) into a 4-view HUD GIF + frame strip.

Layout per frame: main oblique view (left, 448) + front/side/top stack (right),
with a bottom HUD bar: level class, current action, step, min clearance.

    python scripts/make_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from physical_passage.config import load_config
from physical_passage.envs.actions import ACTIONS, A
from physical_passage.envs.passage_env import PassageEnv
from physical_passage.scene.generator import SceneGenerator
from scripts.run_oracle import verified_sample

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(env: PassageEnv, label: str, action_name: str, step: int,
            clearance: float, declared: bool = False) -> Image.Image:
    main = env.observer.main_view()["rgb"]
    aux = env.observer.aux_views(depth=False, seg=False)
    right = np.concatenate([aux[k]["rgb"] for k in ("front", "side", "top")], axis=0)
    right_img = Image.fromarray(right).resize((main.shape[0] // 3, main.shape[0]),
                                              Image.LANCZOS)
    canvas = Image.new("RGB", (main.shape[1] + right_img.width, main.shape[0] + 56),
                       (26, 26, 26))
    canvas.paste(Image.fromarray(main), (0, 0))
    canvas.paste(right_img, (main.shape[1], 0))
    d = ImageDraw.Draw(canvas)
    y0 = main.shape[0]
    d.text((10, y0 + 6), f"{label.upper()}  |  {action_name}", fill=(240, 240, 240),
           font=_font(20))
    color = (255, 80, 80) if declared else (150, 210, 150)
    status = "DECLARE_IMPOSSIBLE" if declared else f"clearance {clearance:+.3f} m"
    d.text((10, y0 + 32), f"step {step:3d}   {status}", fill=color, font=_font(16))
    return canvas


def main() -> None:
    cfg = load_config()
    gen = SceneGenerator(cfg, seed=7)
    env = PassageEnv(cfg)
    frames: list[Image.Image] = []

    for label in ("feasible", "rotation_required", "impossible"):
        spec, sol = verified_sample(gen, cfg, label=label)
        env.reset(options={"spec": spec})
        plan = sol.actions if sol.actions else [A["DECLARE_IMPOSSIBLE"]]
        info = {"minimum_clearance": 0.0, "step": 0}
        frames.append(compose(env, label, "START", 0, 0.0))
        for act in plan:
            _, _, term, trunc, info = env.step(act)
            frames.append(compose(env, label, ACTIONS[act], info["step"],
                                  info["minimum_clearance"],
                                  declared=bool(info.get("declared_impossible"))))
            if term or trunc:
                break
        # hold the final frame so the outcome is readable
        frames.extend([frames[-1]] * 8)

    out = RESULTS / "01_demos" / "01_oracle_three_classes.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    strip = np.concatenate([np.array(frames[i]) for i in idx], axis=1)
    Image.fromarray(strip).save(out.with_suffix(".png"))
    env.close()
    print("wrote", out)
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
