"""Check paired native fingertip reach transfer on the narrow-bridge hand."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010"
TRIALS = OUT / "reach-entry"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "reach-entry-plan.json").read_text())
    identities = {
        "selected_pointer_sha256": "robot/ape-current-design.json",
        "selected_body_sha256": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
        "candidate_body_sha256": "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml",
        "checkpoint_sha256": "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt",
        "geometry_sha256": "artifacts/ape-cove50-side-reach-20261009/geometry-center.json",
        "probe_sha256": "experiments/compact_morphology/cove50_policy_transfer/probe_side_reach_native.py",
    }
    for key, relative in identities.items():
        if sha(ROOT / relative) != plan[key]:
            raise ValueError(f"Source identity changed: {key}")
    for line in (TRIALS / "remote-sha256.txt").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if sha(ROOT / relative.strip()) != expected:
            raise ValueError(f"Remote output hash mismatch: {relative}")
    rows = []
    comparisons = []
    for spec in plan["cases"]:
        label = spec["label"]
        pair = {}
        for body in ("selected", "candidate"):
            case = f"{label}-{body}"
            result = json.loads((TRIALS / f"{case}.json").read_text())
            expected = plan[f"{body}_body_sha256"]
            if (result["body_sha256"] != expected or
                    result["policy_checkpoint_sha256"] != plan["checkpoint_sha256"] or
                    result["geometry_sha256"] != plan["geometry_sha256"] or
                    result["selected_design_pointer_sha256"] != plan["selected_pointer_sha256"] or
                    result["seed"] != spec["seed"] or
                    result["rail_v"] != spec["rail_v"] or
                    result["delay_ms"] != spec["delay_ms"] or
                    result["toy_xy_offset_m"] != [spec["toy_x_offset_m"], spec["toy_y_offset_m"]]):
                raise ValueError(f"Changed paired replay contract: {case}")
            row = {
                "case": case, "label": label, "body": body,
                "seed": result["seed"], "rail_v": result["rail_v"],
                "delay_ms": result["delay_ms"],
                "duration_s": result["duration_s"],
                "survived": result["survived"],
                "continuous_finger_contact_s": result["longest_continuous_finger_toy_contact_s"],
                "both_feet_fraction": result["both_feet_grounded_fraction"],
                "rated_leg_fraction": result["rated_leg_fraction"],
                "max_hand_floor_force_n": result["max_hand_ground_force_n"],
                "report": str((TRIALS / f"{case}.json").relative_to(ROOT)),
                "trace": str((TRIALS / f"{case}.npz").relative_to(ROOT)),
                "video": (str((TRIALS / f"{case}.mp4").relative_to(ROOT))
                          if (TRIALS / f"{case}.mp4").exists() else None),
            }
            row["pass"] = (row["duration_s"] >= 8 and row["survived"] and
                           row["continuous_finger_contact_s"] >= 1 and
                           row["both_feet_fraction"] >= .99 and
                           row["rated_leg_fraction"] <= .10 and
                           row["max_hand_floor_force_n"] < 1)
            rows.append(row)
            pair[body] = row
        comparisons.append({
            "case": label, "seed": spec["seed"],
            "selected_pass": pair["selected"]["pass"],
            "candidate_pass": pair["candidate"]["pass"],
            "contact_delta_s": round(pair["candidate"]["continuous_finger_contact_s"] -
                                     pair["selected"]["continuous_finger_contact_s"], 3),
            "candidate_transfer_pass": pair["selected"]["pass"] and pair["candidate"]["pass"],
        })
    report = {
        "schema": "robot-learning-lab.cove50-hand-candidate-reach-transfer.v1",
        "plan_sha256": sha(OUT / "reach-entry-plan.json"),
        "remote_output_hashes_sha256": sha(TRIALS / "remote-sha256.txt"),
        "cases": rows, "comparison": comparisons,
        "candidate_transfer_passes": sum(row["candidate_transfer_pass"] for row in comparisons),
        "limits": [
            "The frozen feedback controller includes an authored trajectory and simulator-truth target direction.",
            "This replay begins at a prepositioned attained upright stand, not the saved squat terminal state.",
            "Fingertip contact does not demonstrate two-jaw pregrasp, closure, possession, lift or place.",
            "The candidate hand remains unselected with unmeasured material, bearing and contact properties.",
        ],
    }
    (OUT / "reach-entry-matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "reach-entry-comparison.json").write_text(json.dumps({"cases": comparisons}, indent=2) + "\n")
    (OUT / "reach-entry-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_transfer_passes": report["candidate_transfer_passes"],
                      "comparison": comparisons}, indent=2))


if __name__ == "__main__":
    main()
