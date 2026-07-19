"""Offline metric aggregation over episodes.jsonl -> CSV + JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def aggregate(log_dir: str | Path) -> dict:
    eps = [json.loads(l) for l in open(Path(log_dir) / "episodes.jsonl")]
    n = len(eps)
    succ = [e for e in eps if e.get("success")]
    coll = [e for e in eps if e.get("collision")]
    imp = [e for e in eps if e["spec"]["label"] == "impossible"]
    feas = [e for e in eps if e["spec"]["label"] != "impossible"]
    declared = [e for e in eps if e.get("declared_impossible")]

    def rate(part, whole):
        return round(len(part) / len(whole), 4) if whole else None

    m = {
        "episodes": n,
        "success_rate": rate(succ, eps),
        "collision_rate": rate(coll, eps),
        # feasibility judgment: declaring impossible on impossible levels
        "impossible_recall": rate([e for e in imp if e.get("declared_impossible")], imp),
        "false_impossible_rate": rate([e for e in feas if e.get("declared_impossible")], feas),
        "feasibility_accuracy": rate(
            [e for e in eps if
             (e["spec"]["label"] == "impossible") == bool(e.get("declared_impossible"))], eps),
        "success_rate_feasible_levels": rate([e for e in feas if e.get("success")], feas),
        "avg_episode_length": round(sum(e["steps"] for e in eps) / n, 2) if n else None,
        "avg_min_clearance": round(sum(e["min_clearance"] for e in eps) / n, 4) if n else None,
        "by_label": {},
    }
    for lab in ("feasible", "rotation_required", "impossible"):
        sub = [e for e in eps if e["spec"]["label"] == lab]
        ok = [e for e in sub if e.get("success") or
              (lab == "impossible" and e.get("declared_impossible"))]
        m["by_label"][lab] = {"n": len(sub), "solved_rate": rate(ok, sub)}
    return m


def export(log_dir: str | Path, out_dir: str | Path) -> dict:
    m = aggregate(log_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
    for lab, d in m["by_label"].items():
        flat[f"{lab}_n"] = d["n"]
        flat[f"{lab}_solved_rate"] = d["solved_rate"]
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(flat.keys())
        w.writerow(flat.values())
    return m
