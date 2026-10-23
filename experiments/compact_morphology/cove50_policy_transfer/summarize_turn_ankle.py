"""Pair native Cove turn adaptation screens on reserved left/right seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-turn-recovery-20261008"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
POINTER = ROOT / "robot/ape-current-design.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    specs = [
        ("Source turn actor", "turneffort25-screen", "turneffort25-model950.pt"),
        ("Ankle limit · before train", "turneffort25-ankleclip02-beforetrain", "turneffort25-model950.pt"),
        ("Ankle limit · 25 updates", "turnankle25", "turnankle25-model975.pt"),
        ("Ankle limit · 50 updates", "turnankle50", "turnankle50-model1000.pt"),
        ("Ankle limit · 100 updates", "turnankle100", "turnankle100-model1049.pt"),
    ]
    cases = []
    for label, prefix, checkpoint_name in specs:
        for side, seed in (("left", 4105), ("right", 4106)):
            path = OUT / f"{prefix}-{side}-turn.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if data["body_sha256"] != digest(BODY) or data["seed"] != seed:
                raise ValueError(f"Unpaired body or seed: {path.name}")
            worlds = data["worlds"]
            checkpoint = OUT / checkpoint_name
            cases.append({
                "case": f"{label} · {side}",
                "report": str(path.relative_to(ROOT)),
                "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else None,
                "checkpoint_sha256": digest(checkpoint) if checkpoint.exists() else None,
                "seed": seed,
                "worlds": len(worlds),
                "survived": sum(world["ended_by_timeout"] for world in worlds),
                "mean_yaw_rad": round(sum(world["net_yaw_rad"] for world in worlds) / len(worlds), 3),
                "mean_abs_lateral_m": round(sum(abs(world["lateral_m"]) for world in worlds) / len(worlds), 3),
                "mean_rated_leg_fraction": round(sum(world["fraction_frames_with_saturated_leg"] for world in worlds) / len(worlds), 3),
                "max_hand_force_n": round(max(world["hand_force_n"] for world in worlds), 2),
            })
    matrix = {
        "schema": "robot-learning-lab.cove50-turn-ankle-adaptation-matrix.v1",
        "selected_pointer_sha256": digest(POINTER),
        "selected_body_sha256": digest(BODY),
        "comparison": "Four native worlds per left/right case, reserved seeds 4105/4106, requested yaw +/-0.4 rad/s for 8 s at 12 V. Ankle-limit cases clip ankle pitch targets 0.02 rad inside selected joint travel.",
        "cases": cases,
    }
    (OUT / "turn-ankle-adaptation-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "yaw": [case["mean_yaw_rad"] for case in cases]}))


if __name__ == "__main__":
    main()
