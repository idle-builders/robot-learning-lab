"""Bounded local contract audit against the frozen V10 native CPU pairs.

This does not run MJLab or score a learned actor. It checks the new sampled
action geometry and clock/side evidence without opening any reserved start.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
V10 = ROOT / "artifacts/ape-low-level-training-20261013/support-reference-parity-v10"
OUT = ROOT / "artifacts/ape-low-level-training-20261013/support-sampled-contract-v11/report.json"
JOINTS = ("left_hip_roll", "right_hip_roll", "left_shoulder_roll", "right_shoulder_roll")
BOUNDARIES = (0.0, 1.0, 2.0, 5.0, 7.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v10 = json.loads((V10 / "report.json").read_text())
    plan = json.loads((V10 / "plan.json").read_text())
    body = ROOT / plan["body"]
    if sha(body) != plan["body_sha256"] or v10["parity_passes"] != 4:
        raise ValueError("Frozen V10 source or parity result changed")

    rows = []
    legacy_rechecks = []
    for case in v10["cases"]:
        baseline_path = ROOT / case["baseline_trace"]
        reference_path = ROOT / case["reference_trace"]
        if (sha(baseline_path) != case["baseline_trace_sha256"] or
                sha(reference_path) != case["reference_trace_sha256"]):
            raise ValueError(f"Frozen trace changed: {case['tag']}")
        baseline, reference = np.load(baseline_path), np.load(reference_path)
        names = reference["joint_names"].tolist()
        if names != baseline["joint_names"].tolist() or len(names) != 21:
            raise ValueError(f"Native joint order changed: {case['tag']}")
        ids = np.asarray([names.index(name) for name in JOINTS])
        native = reference["reference_native_action"]
        target = reference["joint_target"]
        decoded = native * reference["reference_native_scale"] + reference["reference_native_offset"]
        if not (native.shape == (400, 21) and
                np.array_equal(native, baseline["action"]) and
                np.array_equal(decoded, target) and
                np.array_equal(target, baseline["joint_target"]) and
                np.array_equal(reference["qpos"], baseline["qpos"]) and
                np.array_equal(reference["qvel"], baseline["qvel"])):
            raise ValueError(f"Zero-residual V10 native contract changed: {case['tag']}")

        replay_stem = OUT.parent / f"legacy-zero-{case['tag']}"
        replay_report = json.loads(replay_stem.with_suffix(".json").read_text())
        with np.load(replay_stem.with_suffix(".npz")) as replay:
            gaps = {key: float(np.max(np.abs(reference[key] - replay[key])))
                    for key in ("reference_native_action", "joint_target", "qpos", "qvel")}
            if (any(gap != 0.0 for gap in gaps.values()) or
                    replay["action"].shape != (400, 21) or
                    not np.array_equal(replay["action"], np.zeros((400, 21)))):
                raise ValueError(f"Current 21D adapter regressed V10: {case['tag']}")
        if (replay_report["reference_action_source_sha256"] !=
                sha(Path(__file__).with_name("support_reference_action.py")) or
                replay_report["body_sha256"] != plan["body_sha256"] or
                replay_report["trace_sha256"] != sha(replay_stem.with_suffix(".npz")) or
                not replay_report["load_transfer_gate"]["passed"]):
            raise ValueError(f"Current 21D source/gate mismatch: {case['tag']}")
        legacy_rechecks.append({
            "case": case["tag"], "report_sha256": sha(replay_stem.with_suffix(".json")),
            "trace_sha256": sha(replay_stem.with_suffix(".npz")),
            "max_abs_gap": gaps, "full_load_transfer_gate_pass": True,
        })

        # This is an independent NumPy check of the intended 4D scatter and
        # clamp, using frozen native targets. It is not an MJLab action call.
        rng = np.random.default_rng(case["seed"] + (1 if case["stance"] == "right" else 2))
        raw = rng.normal(0.0, 0.2, size=(400, 4)).astype(np.float32)
        raw[0] = (2.0, -2.0, 0.5, -0.5)
        bounded = np.clip(raw, -1.0, 1.0)
        before_joint_limit = target.copy()
        before_joint_limit[:, ids] += bounded * np.float32(0.05)
        limits = reference["reference_absolute_clip"]
        applied = np.clip(before_joint_limit, limits[:, 0], limits[:, 1])
        inactive = np.ones(21, dtype=bool)
        inactive[ids] = False
        if not np.array_equal(applied[:, inactive], target[:, inactive]):
            raise ValueError(f"Inactive joint changed: {case['tag']}")
        if np.max(np.abs(applied - target)) > 0.050001:
            raise ValueError(f"Sampled correction exceeded 0.05 rad: {case['tag']}")
        if not np.array_equal(bounded[0, :2], np.asarray((1.0, -1.0))):
            raise ValueError(f"Raw action clip was not exercised: {case['tag']}")

        report = json.loads((ROOT / case["reference_report"]).read_text())
        series = report["timeseries"]
        side = 1.0 if case["stance"] == "right" else -1.0
        boundary = []
        for seconds in BOUNDARIES:
            index = int(round(seconds / 0.02))
            logged = series[max(index - 1, 0)]
            boundary.append({
                "actor_skill_time_s": seconds,
                "reference_step_index": index,
                "reference_hip_roll_rad": float(target[index, names.index("left_hip_roll")]),
                "logged_post_step_command_flag": logged["command_flag"],
                "logged_post_step_observed_side": logged["command_side"],
                "logged_post_step_alpha": logged["command_alpha"],
                "required_latched_side": side,
            })
        if any(series[i]["command_side"] != 0.0 for i in (0, 299, 399)):
            raise ValueError(f"Idle/return command side was not zero: {case['tag']}")
        if abs(target[300, names.index("left_hip_roll")]) < 0.1:
            raise ValueError(f"Return reference lost the side latch: {case['tag']}")
        rows.append({
            "case": case["tag"], "stance": case["stance"], "grip": case["grip"],
            "zero_native_action_gap": 0.0, "zero_target_gap_rad": 0.0,
            "zero_state_gap": 0.0, "sampled_raw_std": 0.2,
            "raw_clip": 1.0, "residual_scale_rad": 0.05,
            "sampled_max_target_change_rad": float(np.max(np.abs(applied - target))),
            "sampled_changed_joint_names": list(JOINTS),
            "boundary": boundary,
        })

    native_rows = []
    for stem, side in (("native-preflight", "right"),
                       ("native-preflight-left", "left")):
        record_path = OUT.parent / f"{stem}.json"
        trace_path = OUT.parent / f"{stem}.npz"
        record = json.loads(record_path.read_text())
        zero = record["full_zero_preflight"]
        sample = record["sampled_action_preflight"]
        sign = 1.0 if side == "right" else -1.0
        with np.load(trace_path) as trace:
            if (trace["joint_target"].shape != (400, 21) or
                    trace["qpos"].shape != (400, 28) or
                    trace["qvel"].shape != (400, 27)):
                raise ValueError(f"Incomplete new native trace: {side}")
        if (record["action_shape"] != [1, 4] or
                Path(record["body"]).resolve() != body.resolve() or
                record["reset_command"] != {"flag": 0.0, "observed_side": 0.0,
                                            "latched_side": sign, "alpha": 0.0,
                                            "skill_time_s": 0.0} or
                zero["side"] != side or zero["steps"] != 400 or
                zero["early_termination_step"] is not None or
                zero["max_decoded_teacher_target_gap_rad"] != 0.0 or
                sample["joint_names"] != list(JOINTS) or
                sample["raw"] != sample["action_manager_current_raw"] or
                sample["action_manager_previous_raw"] != [0.0] * 4 or
                max(abs(value) for value in sample["target_correction_rad"]) > 0.050001 or
                record["clip_preflight"]["bounded"] != [1.0] * 4):
            raise ValueError(f"New native action/reset contract failed: {side}")
        for boundary in zero["control_boundaries"]:
            t = boundary["skill_time_s"]
            requested = 1.0 <= t < 5.0
            if (boundary["flag"] != float(requested) or
                    boundary["observed_side"] != (sign if requested else 0.0) or
                    boundary["latched_side"] != sign):
                raise ValueError(f"New native command clock failed: {side}, t={t}")
        native_rows.append({
            "side": side, "seed": record["seed"],
            "record_sha256": sha(record_path), "trace_sha256": sha(trace_path),
            "steps": zero["steps"], "max_target_gap_rad": zero["max_decoded_teacher_target_gap_rad"],
            "sampled_target_correction_rad": sample["target_correction_rad"],
            "control_boundaries": zero["control_boundaries"],
        })

    result = {
        "schema": "robot-learning-lab.support-sampled-contract-local.v2",
        "claim": "Current 21D adapter exactly replays all four V10 native pairs and gates; new one-world native 4D zero rollouts and sampled action preflights on both sides; no PPO update or new full physical-gate score",
        "source_body": plan["body"], "source_body_sha256": plan["body_sha256"],
        "frozen_v10_report_sha256": sha(V10 / "report.json"),
        "action_source": "experiments/compact_morphology/cove50_support_recovery/support_reference_action.py",
        "action_source_sha256": sha(Path(__file__).with_name("support_reference_action.py")),
        "trainer_source": "experiments/compact_morphology/cove50_support_recovery/train_support_cycle.py",
        "trainer_source_sha256": sha(Path(__file__).with_name("train_support_cycle.py")),
        "runtime_gate": "MJLab 1.3.0 and torch local CPU preflight executed; actual PPO.act and paid optimization remain unrun",
        "cases": rows,
        "legacy_zero_rechecks": legacy_rechecks,
        "native_preflights": native_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(OUT), "cases": len(rows),
                      "max_sampled_change_rad": max(x["sampled_max_target_change_rad"] for x in rows)},
                     indent=2))


if __name__ == "__main__":
    main()
