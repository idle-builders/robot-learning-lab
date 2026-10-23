"""Compare hip-roll effort adaptation without hiding lost signed turns."""

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
        ("Hip-roll penalty · 25 updates", "turnhiproll25", "turnhiproll25-model975.pt"),
        ("Hip-roll penalty · 50 updates", "turnhiproll50", "turnhiproll50-model1000.pt"),
        ("Hip-roll penalty · 100 updates", "turnhiproll100", "turnhiproll100-model1049.pt"),
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
            mean_joint = lambda index: round(sum(world["saturation_fraction_by_leg_joint"][index]
                                                 for world in worlds) / len(worlds), 3)
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
                "left_hip_roll_rated_fraction": mean_joint(1),
                "right_hip_roll_rated_fraction": mean_joint(6),
                "max_hand_force_n": round(max(world["hand_force_n"] for world in worlds), 2),
            })
    matrix = {
        "schema": "robot-learning-lab.cove50-turn-hiproll-adaptation-matrix.v1",
        "selected_pointer_sha256": digest(POINTER),
        "selected_body_sha256": digest(BODY),
        "comparison": "Four native worlds per left/right case, seeds 4105/4106, requested yaw +/-0.4 rad/s for 8 s at 12 V. The continuation adds a -2 reward penalty above 70% catalog-rated torque on the two hip-roll drives, retaining source action mapping and gym variation.",
        "cases": cases,
    }
    (OUT / "turn-hiproll-adaptation-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "yaw": [case["mean_yaw_rad"] for case in cases]}))


if __name__ == "__main__":
    main()
