"""Verify native candidate-hand reach tuning and fresh reserved replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010"
GROUPS = ("reach-entry", "reach-tuning", "reach-pitch", "reach-pitch-reserved",
          "reach-margin-development", "reach-margin-reserved")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(row: dict) -> bool:
    return (row["survived"] and row["duration_s"] >= 8 and
            row["continuous_finger_contact_s"] >= 1 and
            row["both_feet_fraction"] >= .99 and
            row["rated_leg_fraction"] <= .1 and
            row["max_hand_floor_force_n"] < 1)


def load_case(group: str, case: str, body: str, offset: float, stage: str) -> dict:
    path = OUT / group / f"{case}.json"
    r = json.loads(path.read_text())
    plan = json.loads((OUT / "reach-entry-plan.json").read_text())
    if (r["selected_design_pointer_sha256"] != plan["selected_pointer_sha256"] or
            r["body_sha256"] != plan[f"{body}_body_sha256"] or
            r["policy_checkpoint_sha256"] != plan["checkpoint_sha256"] or
            r["geometry_sha256"] != plan["geometry_sha256"]):
        raise ValueError(f"Wrong source identity: {path}")
    if abs(r["joint_target_offsets_rad"]["right_shoulder_pitch_base"] - offset) > 1e-9:
        raise ValueError(f"Wrong pitch offset: {path}")
    row = {
        "case": case, "stage": stage, "body": body, "pitch_offset_rad": offset,
        "seed": r["seed"], "rail_v": r["rail_v"], "delay_ms": r["delay_ms"],
        "duration_s": r["duration_s"], "survived": r["survived"],
        "continuous_finger_contact_s": r["longest_continuous_finger_toy_contact_s"],
        "both_feet_fraction": r["both_feet_grounded_fraction"],
        "rated_leg_fraction": r["rated_leg_fraction"],
        "max_hand_floor_force_n": r["max_hand_ground_force_n"],
        "report": str(path.relative_to(ROOT)),
        "trace": str(path.with_suffix(".npz").relative_to(ROOT)),
        "video": (str(path.with_suffix(".mp4").relative_to(ROOT))
                  if path.with_suffix(".mp4").exists() else None),
    }
    row["pass"] = gate(row)
    return row


def compare(rows: list[dict], stage: str, names: tuple[str, ...]) -> list[dict]:
    comparison = []
    for name in names:
        selected = next(x for x in rows if x["stage"] == stage and x["case"] == f"{name}-selected")
        candidate = next(x for x in rows if x["stage"] == stage and x["case"] == f"{name}-candidate")
        comparison.append({
            "case": name, "stage": stage, "seed": selected["seed"],
            "selected_pass": selected["pass"], "candidate_pass": candidate["pass"],
            "selected_contact_s": selected["continuous_finger_contact_s"],
            "candidate_contact_s": candidate["continuous_finger_contact_s"],
            "candidate_transfer_pass": selected["pass"] and candidate["pass"],
        })
    return comparison


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
            raise ValueError(f"Source changed: {relative}")
    manifests = {}
    for group in GROUPS:
        manifest = OUT / group / "remote-sha256.txt"
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            if sha(ROOT / relative.strip()) != expected:
                raise ValueError(f"Remote file changed: {relative}")
        manifests[group] = sha(manifest)
    rows = []
    for label in ("development", "held_out"):
        for body in ("selected", "candidate"):
            rows.append(load_case("reach-entry", f"{label}-{body}", body, .06, "frozen_baseline"))
    for name in ("010", "020", "030"):
        rows.append(load_case("reach-tuning", f"dev-rollminus{name}", "candidate", .06,
                              "negative_roll_development"))
    for name, offset in (("075", .075), ("090", .09), ("105", .105)):
        rows.append(load_case("reach-pitch", f"dev-pitch{name}", "candidate", offset,
                              "pitch_development"))
    for name in ("held_start", "held_toy"):
        for body in ("selected", "candidate"):
            rows.append(load_case("reach-pitch-reserved", f"{name}-{body}", body, .09,
                                  "pitch_090_reserved"))
    for name in ("toy_shift", "low_rail"):
        rows.append(load_case("reach-margin-development", name, "candidate", .105,
                              "pitch_105_development"))
    for name in ("held_toy", "held_combo"):
        for body in ("selected", "candidate"):
            rows.append(load_case("reach-margin-reserved", f"{name}-{body}", body, .105,
                                  "pitch_105_reserved"))
    comparisons = (
        compare(rows, "frozen_baseline", ("development", "held_out")) +
        compare(rows, "pitch_090_reserved", ("held_start", "held_toy")) +
        compare(rows, "pitch_105_reserved", ("held_toy", "held_combo"))
    )
    report = {
        "schema": "robot-learning-lab.cove50-hand-candidate-reach-recovery.v1",
        "selected_pointer_sha256": plan["selected_pointer_sha256"],
        "selected_body_sha256": plan["selected_body_sha256"],
        "candidate_body_sha256": plan["candidate_body_sha256"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "remote_output_manifest_hashes": manifests,
        "cases": rows, "comparison": comparisons,
        "frozen_baseline_candidate_passes": sum(x["candidate_transfer_pass"] for x in comparisons[:2]),
        "pitch_090_reserved_candidate_passes": sum(x["candidate_transfer_pass"] for x in comparisons[2:4]),
        "pitch_105_reserved_candidate_passes": sum(x["candidate_transfer_pass"] for x in comparisons[4:]),
        "limits": [
            "The controller retains an authored path, frozen goal residual and simulator-truth target direction; the pitch offset is a tuned target, not a retrained neural actor.",
            "These replays start from a prepositioned attained stand and do not establish a squat-to-reach handoff.",
            "Fingertip contact is not two-jaw pregrasp, closure, possession, lift or place.",
            "The narrow-bridge hand is unselected and its material, bearing and contact properties are unmeasured.",
        ],
    }
    (OUT / "reach-recovery-matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "reach-recovery-comparison.json").write_text(json.dumps({"cases": comparisons}, indent=2) + "\n")
    (OUT / "reach-recovery-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "frozen_baseline_candidate_passes", "pitch_090_reserved_candidate_passes",
        "pitch_105_reserved_candidate_passes", "comparison")}, indent=2))


if __name__ == "__main__":
    main()
