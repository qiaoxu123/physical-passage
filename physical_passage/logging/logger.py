"""Episode logging: one JSONL line per step + a summary line per episode."""

from __future__ import annotations

import json
from pathlib import Path


class EpisodeLogger:
    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps_f = open(self.dir / "steps.jsonl", "a")
        self.eps_f = open(self.dir / "episodes.jsonl", "a")

    def log_step(self, episode_id: int, step: int, action_name: str,
                 reward: float, info: dict) -> None:
        rec = {
            "episode_id": episode_id, "step": step, "action": action_name,
            "reward": round(reward, 4),
            "collision": info.get("collision"),
            "minimum_clearance": round(info["minimum_clearance"], 5),
            "time_to_collision": info.get("time_to_collision"),
            "position": [round(v, 4) for v in info["position"]],
            "rotation_euler_deg": [round(v, 2) for v in info["rotation_euler_deg"]],
            "passed_wall": info.get("passed_wall"),
        }
        self.steps_f.write(json.dumps(rec) + "\n")

    def log_episode(self, episode_id: int, spec_dict: dict, outcome: dict) -> None:
        self.eps_f.write(json.dumps(
            {"episode_id": episode_id, "spec": spec_dict, **outcome}) + "\n")
        self.steps_f.flush()
        self.eps_f.flush()

    def close(self) -> None:
        self.steps_f.close()
        self.eps_f.close()
