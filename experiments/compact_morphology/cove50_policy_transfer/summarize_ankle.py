"""Pair the ankle-target continuation's nominal screens on one saved seed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-recovery-20261008"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
POINTER = ROOT / "robot/ape-current-design.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    specs = [
        ("effort100-fresh-forward-seed144.json", "Saved actor · original targets", "effort100-model745.pt"),
        ("effort100-ankleclip02-beforetrain-seed2144.json", "Saved actor · ankle limit", "effort100-model745.pt"),
        ("ankleonly30-forward-seed144.json", "Ankle limit · 30 updates", "ankleonly30-model775.pt"),
        ("ankleonly55-forward-seed144.json", "Ankle limit · 55 updates", "ankleonly55-model800.pt"),
        ("ankleonly100-forward-seed144.json", "Ankle limit · 100 updates", "ankleonly100-model844.pt"),
    ]
    cases = []
    for report_name, label, checkpoint_name in specs:
        report_path = OUT / report_name
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        if report["body_sha256"] != digest(BODY) or report["seed"] != 2144:
            raise ValueError(f"Unpaired body or seed: {report_name}")
        worlds = report["worlds"]
        checkpoint = OUT / checkpoint_name
        cases.append({
            "case": label,
            "report": str(report_path.relative_to(ROOT)),
            "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else None,
            "checkpoint_sha256": digest(checkpoint) if checkpoint.exists() else None,
            "worlds": len(worlds),
            "survived": sum(world["ended_by_timeout"] for world in worlds),
            "one_meter_alternating": report["one_meter_with_alternating_steps"],
            "mean_forward_m": round(sum(world["forward_m"] for world in worlds) / len(worlds), 3),
            "mean_abs_lateral_m": round(sum(abs(world["lateral_m"]) for world in worlds) / len(worlds), 3),
            "mean_rated_leg_fraction": round(sum(world["fraction_frames_with_saturated_leg"] for world in worlds) / len(worlds), 3),
            "max_hand_force_n": round(max(world["hand_force_n"] for world in worlds), 2),
            "max_body_contact_fraction": round(max(world["fraction_frames_with_body_contact"] or 0 for world in worlds), 3),
        })
    matrix = {
        "schema": "robot-learning-lab.cove50-ankle-adaptation-matrix.v1",
        "selected_pointer_sha256": digest(POINTER),
        "selected_body_sha256": digest(BODY),
        "comparison": "Same four nominal worlds, seed 2144, 0.35 m/s for 8 s, 12 V and no command delay. Ankle-only cases clip target angles 0.02 rad inside the selected frame's limits.",
        "cases": cases,
    }
    (OUT / "ankle-adaptation-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "one_meter": [case["one_meter_alternating"] for case in cases]}))


if __name__ == "__main__":
    main()
