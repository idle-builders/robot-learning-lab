"""Compare signed support residual pulses with same-start zero-residual replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from summarize_support_reward_prefit_v15 import score_case, sha


def physical_score(root: Path, model: mujoco.MjModel, side: str, stem: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="support-v17-score-") as tmp:
        stage = Path(tmp)
        for extension in ("json", "npz"):
            (stage / f"{side}-reference.{extension}").symlink_to(
                (root / f"{stem}.{extension}").resolve()
            )
        return score_case(stage, model, side, "reference")


def mean(rows: list[dict], key: str, start: float, end: float) -> float:
    values = [row[key] for row in rows if start <= row["time_s"] < end]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--side", choices=("left", "right"), required=True)
    args = p.parse_args()
    method = json.loads((args.root / "method.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    stems = ["baseline"] + [joint + suffix for joint in method["joint_names"]
                            for suffix in ("-minus", "-plus")]
    entries = []
    baseline_rows = None
    baseline_hash = None
    for name in stems:
        stem = f"{args.side}-{name}"
        path = args.root / f"{stem}.json"
        episode = json.loads(path.read_text())["reward_episode_preflight"]
        assert episode["side"] == args.side and episode["seed"] == method["seeds"][args.side]
        if baseline_hash is None:
            baseline_hash = episode["initial_state_sha256"]
            baseline_rows = episode["rows"]
        assert episode["initial_state_sha256"] == baseline_hash
        score = physical_score(args.root, model, args.side, stem)
        rows = episode["rows"]
        entries.append({
            "stem": stem,
            "joint": episode["pulse_joint"],
            "raw_pulse": episode["pulse_raw"],
            "native_target_pulse_rad": episode["pulse_raw"] * .05 if episode["pulse_raw"] is not None else None,
            "steps": score["steps"],
            "full_gate_pass": score["full_gate_pass"],
            "failure_reasons": score["failure_reasons"],
            "pulse_mean_load": mean(rows, "stance_load_fraction", 2.52, 2.70),
            "after_mean_load": mean(rows, "stance_load_fraction", 2.70, 3.00),
            "pulse_mean_tilt_deg": mean(rows, "tilt_deg", 2.52, 2.70),
            "delta_pulse_load_vs_baseline": mean(rows, "stance_load_fraction", 2.52, 2.70)
                                            - mean(baseline_rows, "stance_load_fraction", 2.52, 2.70),
            "delta_after_load_vs_baseline": mean(rows, "stance_load_fraction", 2.70, 3.00)
                                            - mean(baseline_rows, "stance_load_fraction", 2.70, 3.00),
            "delta_pulse_tilt_deg_vs_baseline": mean(rows, "tilt_deg", 2.52, 2.70)
                                                - mean(baseline_rows, "tilt_deg", 2.52, 2.70),
            "max_patch_slip_m": score["max_patch_slip_m"],
            "rated_leg_fraction": score["rated_leg_fraction"],
            "forbidden_contact_frames": score["forbidden_contact_frames"],
            "report_sha256": sha(path),
            "trace_sha256": sha(args.root / f"{stem}.npz"),
        })
    opposing = []
    for joint in method["joint_names"]:
        pair = [entry for entry in entries if entry["joint"] == joint]
        assert len(pair) == 2
        minus, plus = pair
        opposing.append({"joint": joint,
                         "load_difference_plus_minus": plus["pulse_mean_load"] - minus["pulse_mean_load"],
                         "tilt_difference_plus_minus_deg": plus["pulse_mean_tilt_deg"] - minus["pulse_mean_tilt_deg"],
                         "both_full_gate_pass": minus["full_gate_pass"] and plus["full_gate_pass"]})
    output = {"schema": "robot-learning-lab.support-pulse-authority-v17.side.v1",
              "side": args.side, "method_sha256": sha(args.root / "method.json"),
              "scorer_sha256": sha(Path(__file__)), "body_sha256": sha(args.body_xml),
              "same_initial_state_sha256": baseline_hash,
              "entries": entries, "opposing_responses": opposing,
              "claim_limit": "One opened-development local CPU start per side, zero updates; signed response is not a safe corrective learner or physical qualification."}
    (args.root / f"{args.side}-score.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"side": args.side, "opposing_responses": opposing}, indent=2))


if __name__ == "__main__":
    main()
