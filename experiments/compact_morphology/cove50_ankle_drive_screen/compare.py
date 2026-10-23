"""Build the concise paired result table from retained native traces."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-ankle-drive-screen-20261010"


def main() -> None:
    cases = {row["case"]: row for row in json.loads((OUT / "matrix.json").read_text())["cases"]}
    rows = []
    for skill in ("forward", "turn-left", "turn-right"):
        a, b = (cases[f"{skill}-{motor}"] for motor in ("selected", "st3250"))
        reports = [json.loads((OUT / f"{skill}-{motor}.json").read_text())
                   for motor in ("selected", "st3250")]
        same_start = reports[0]["initial_state_sha256"] == reports[1]["initial_state_sha256"]
        same_first_action = reports[0]["first_action_sha256"] == reports[1]["first_action_sha256"]
        if not same_start or not same_first_action:
            raise ValueError(f"Unpaired initial state or action: {skill}")
        for model, row, report in zip(("selected", "st3250"), (a, b), reports):
            outcome = row["outcome"]
            joints = row["joints"]
            rows.append({
                "case": skill, "motor_law": model,
                "survived_8s": outcome["survived_8s"],
                "one_meter_alternating": outcome["one_meter_alternating"],
                "mean_forward_m": round(outcome["mean_forward_m"], 4),
                "mean_yaw_rad": round(outcome["mean_net_yaw_rad"], 4),
                "any_rated_leg_fraction": round(outcome["mean_any_rated_leg_fraction_while_commanded"], 6),
                "left_hip_roll_rated_fraction": round(joints["left_hip_roll"]["rated_fraction"], 6),
                "right_hip_roll_rated_fraction": round(joints["right_hip_roll"]["rated_fraction"], 6),
                "left_ankle_rated_fraction": round(joints["left_ankle_pitch"]["rated_fraction"], 6),
                "right_ankle_rated_fraction": round(joints["right_ankle_pitch"]["rated_fraction"], 6),
                "total_sole_slip_m": round(sum(outcome["total_sole_slip_m_left_right"]), 4),
                "max_hand_force_n": outcome["max_hand_force_n"],
                "max_body_contact_fraction": outcome["max_body_contact_fraction"],
                "same_initial_state_and_first_action": same_start and same_first_action,
                "body_sha256": report["body_sha256"],
                "clip": str((OUT / f"{skill}-{model}.mp4").relative_to(ROOT)),
                "report": str((OUT / f"{skill}-{model}.json").relative_to(ROOT)),
            })
    output = {
        "schema": "robot-learning-lab.cove50-ankle-law-comparison.v1",
        "status": "nominal_screen_failed_stress_skipped",
        "rated_definition": "Fraction of commanded 50 Hz frames with any leg at >=99.5% of its case's catalog-rated limit; 4 native worlds per row.",
        "comparison_boundary": "Only ankle motor law changed. ST-3250 extra mass, case/package and electrical supply not modeled.",
        "cases": rows,
    }
    (OUT / "comparison-matrix.json").write_text(json.dumps(output, indent=2) + "\n")
    print(*(f"{x['case']} {x['motor_law']}: {x['any_rated_leg_fraction']:.1%} rated"
            for x in rows), sep="\n")


if __name__ == "__main__":
    main()
