"""Bounded native replay of saved Direct Frame reach and one-pad touch.

This is an evaluation sweep, not training. It opens only the fresh development
starts in its method file, retains every original, and reports direct physical
measurements. Historical pass fields in reused evaluators are provenance only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
METHOD = Path(__file__).with_name("method.json")
DEFAULT_OUT = ROOT / "artifacts/ape-low-level-basic-sweep-20261016/reach-contact"
REACH_DIR = ROOT / "experiments/compact_morphology/cove50_policy_transfer/reach_image_goal"
TOUCH_DIR = ROOT / "experiments/compact_morphology/cove50_low_level"
V158 = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v158/new-seed-contact-plan.json"
V14 = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-contact-phase-v14"
_TOUCH_AGENT = None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def checked_method() -> dict:
    method = json.loads(METHOD.read_text())
    if method["schema"] != "robot-learning-lab.basic-reach-contact-sweep.v1":
        raise ValueError("Unknown method")
    for entry in method["sources"].values():
        path = ROOT / entry["path"]
        if not path.is_file() or sha(path) != entry["sha256"]:
            raise ValueError(f"Source changed: {path}")
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["id"] != "cove-50-direct-frame" or pointer["selected_design_id"] is not None:
        raise ValueError("Exact experimental body pointer changed")
    return method


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def video_from_reach(report: Path, trace: Path, video: Path) -> None:
    source = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video_poststep"
    if video.exists():
        raise FileExistsError(video)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
         env.get("PYTHONPATH", "")])
    temporary = video.with_name(video.stem + ".tmp.mp4")
    try:
        subprocess.run([sys.executable, "-m", source, "--report", str(report),
                        "--trace", str(trace), "--video", str(temporary)],
                       cwd=ROOT, env=env, check=True)
        os.replace(temporary, video)
    finally:
        temporary.unlink(missing_ok=True)


def reach_visit(case: dict, role: str, out: Path, method: dict) -> dict:
    from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
        run_v95_native_case as native,
    )

    wrapper = REACH_DIR / ("evaluate_aligned_reach_v165_cli.py" if role == "saved"
                           else "evaluate_reach_bc_v167_cli.py")
    evaluator = REACH_DIR / ("evaluate_aligned_reach_v165.py" if role == "saved"
                             else "evaluate_reach_bc_v167.py")
    source = json.loads(V158.read_text())["native_source_sha256"]
    native.EVALUATOR = wrapper
    visit_case = {**case, "id": f"basic-sweep-{role}-{case['seed']}"}
    result = native.run_case(visit_case, {"source_sha256": {
        **source, "evaluator": sha(evaluator)}}, out)
    report_path, trace_path = ROOT / result["result"], ROOT / result["trace"]
    report = json.loads(report_path.read_text())
    aligned = trace_path.with_name(trace_path.stem + "-aligned.npz")
    if (report["evaluator_sha256"] != sha(evaluator)
            or report["trace_sha256"] != sha(trace_path)
            or report["aligned_trace_sha256"] != sha(aligned)
            or report["body_sha256"] != method["sources"]["body"]["sha256"]
            or report["seed"] != case["seed"]
            or not report["one_sim_without_skill_reset"]):
        raise ValueError(f"Reach original did not match: {role}")
    video = report_path.with_suffix(".mp4")
    video_from_reach(report_path, trace_path, video)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == report["reach_start_s"]), None)
    reach = report["phases"].get("reach", {})
    row = {
        "task": "observed_reach", "role": role, "seed": case["seed"],
        "controller": "saved learned route" if role == "saved" else "V167 BC student",
        "episode_duration_s": report["duration_s"],
        "reach_duration_s": reach.get("duration_s"),
        "longest_fingertip_contact_s": reach.get("continuous_finger_contact_s"),
        "two_foot_support_fraction_during_reach": reach.get("both_feet_fraction"),
        "rated_leg_effort_fraction_during_reach": reach.get("rated_leg_fraction"),
        "max_hand_floor_force_n": reach.get("max_hand_floor_force_n"),
        "min_cube_to_hand_distance_m": reach.get("min_cube_to_hand_distance_m"),
        "entry_goal_error_mm_scoring_only": None if entry is None else entry.get(
            "xy_error_truth_mm_scoring_only"),
        "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
        "route": report.get("selection_mode"),
        "ended_early": report["ended_early"],
        "report": rel(report_path), "report_sha256": sha(report_path),
        "trace": rel(trace_path), "trace_sha256": sha(trace_path),
        "aligned_trace": rel(aligned), "aligned_trace_sha256": sha(aligned),
        "video": rel(video), "video_sha256": sha(video),
    }
    return row


def touch_visit(case: dict, role: str, out: Path, method: dict) -> dict:
    global _TOUCH_AGENT
    sys.path.insert(0, str(TOUCH_DIR))
    from evaluate_touch_contact_phase_v14 import visit
    from train_touch_contact_phase_v14 import build_train_config

    checkpoint = V14 / "model_11.pt"
    if _TOUCH_AGENT is None:
        _, train_cfg = build_train_config(seed=37121)
        _TOUCH_AGENT = train_cfg.agent
    raw = visit({"seed": case["seed"], "cube_xy_m": case["cube_xy_m"]},
                "residual" if role == "learned" else "reference", checkpoint,
                _TOUCH_AGENT, out, "cuda:0")
    trace, video = ROOT / raw["trace"], ROOT / raw["video"]
    if (raw["trace_sha256"] != sha(trace) or raw["video_sha256"] != sha(video)
            or raw["seed"] != case["seed"] or raw["steps"] > 200
            or method["sources"]["body"]["sha256"] != sha(
                ROOT / method["sources"]["body"]["path"])):
        raise ValueError(f"Touch original did not match: {role}")
    with np.load(trace, allow_pickle=False) as data:
        if len(data["cube_xyz_m"]) != raw["steps"]:
            raise ValueError("Touch trace length differs")
    return {
        "task": "one_pad_free_object_touch", "role": role,
        "controller": "V14 learned residual around RGB reference" if role == "learned"
                      else "scripted RGB reference", "seed": case["seed"],
        "cube_xy_m": case["cube_xy_m"], "episode_duration_s": raw["steps"] * 0.02,
        "longest_supported_touch_s": raw["longest_supported_touch_s"],
        "finger_contact_s": raw["finger_frames"] * 0.02,
        "two_foot_support_s": raw["both_feet_frames"] * 0.02,
        "cube_table_contact_s": raw["table_frames"] * 0.02,
        "hand_floor_contact_s": raw["hand_floor_frames"] * 0.02,
        "peak_cube_xy_drift_mm": raw["peak_cube_xy_drift_mm"],
        "min_pelvis_height_m": raw["min_pelvis_z_m"],
        "trace": raw["trace"], "trace_sha256": raw["trace_sha256"],
        "video": raw["video"], "video_sha256": raw["video_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", choices=("reach", "touch"))
    parser.add_argument("--resume-touch-after", choices=("39051:reference",),
                        help="Continue after the saved original that preceded the duplicate task-registration fault")
    args = parser.parse_args()
    method = checked_method()
    started = time.time()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite any sweep originals: {out}")
    out.mkdir(parents=True)
    rows: list[dict] = []
    for case in method["reach_cases"] if args.only != "touch" else []:
        for role in ("saved", "student"):
            row = reach_visit(case, role, out, method)
            rows.append(row)
            write(out / "partial.json", {"method_sha256": sha(METHOD), "rows": rows})
            print(json.dumps({"task": row["task"], "role": role,
                              "seed": row["seed"], "contact_s": row["longest_fingertip_contact_s"]}),
                  flush=True)
    touch_jobs = [(case, role) for case in method["touch_cases"]
                  for role in ("reference", "learned")]
    if args.resume_touch_after:
        if args.only != "touch" or touch_jobs[0][0]["seed"] != 39051:
            raise ValueError("Continuation is only for the frozen first touch original")
        prior_dir = DEFAULT_OUT.parent / "reach-contact-touch-resume"
        prior = json.loads((prior_dir / "partial.json").read_text())
        if prior["method_sha256"] != sha(METHOD) or len(prior["rows"]) != 1:
            raise ValueError("Original touch partial does not match frozen method")
        row = prior["rows"][0]
        if row["seed"] != 39051 or row["role"] != "reference":
            raise ValueError("Original touch episode identity differs")
        for kind in ("trace", "video"):
            if sha(ROOT / row[kind]) != row[kind + "_sha256"]:
                raise ValueError("Original touch episode bytes changed")
        touch_jobs = touch_jobs[1:]
    for case, role in touch_jobs if args.only != "reach" else []:
        row = touch_visit(case, role, out, method)
        rows.append(row)
        write(out / "partial.json", {"method_sha256": sha(METHOD), "rows": rows})
        print(json.dumps({"task": row["task"], "role": role,
                          "seed": row["seed"], "contact_s": row["longest_supported_touch_s"]}),
              flush=True)
    write(out / "report.json", {
        "schema": "robot-learning-lab.basic-reach-contact-sweep-results.v1",
        "classification": "new development native CUDA replay on experimental rejected Direct Frame",
        "continuation_after": args.resume_touch_after,
        "method_sha256": sha(METHOD), "runner_sha256": sha(Path(__file__)),
        "device": "cuda:0", "control_hz": 50, "world_slots": 1,
        "process_wall_s": round(time.time() - started, 3),
        "shared_pod_listed_hourly_usd": 0.59,
        "process_wall_rental_equivalent_usd": round(
            (time.time() - started) * 0.59 / 3600, 4),
        "billing_note": "Process-wall rental equivalent only; shared pod startup and other skill workloads are attributed by the coordinator from actual pod lifetime. Posted provider charge not claimed.",
        "episodes": len(rows), "independent_start_families": len({r["seed"] for r in rows}),
        "optimizer_updates": 0, "simulator_transitions": sum(round(r["episode_duration_s"] * 50)
                                                          for r in rows),
        "new_distinct_policies": 0,
        "focal_reused_learned_candidates": 3,
        "controller_count_note": "Saved routed reach, V167 BC student and V14 residual are three focal learned candidates; both reach visits also execute the saved squat prefix, while the touch reference is scripted and is not counted as a learned policy.",
        "rows": rows,
        "limits": "Development breadth sweep only. One-pad touch is reference-guided; opposing-pad possession, lift and physical transfer are not established. No historical sealed score was opened.",
    })


if __name__ == "__main__":
    main()
