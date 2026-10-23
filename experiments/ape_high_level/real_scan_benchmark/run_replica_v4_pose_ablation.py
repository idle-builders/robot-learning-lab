"""Frozen ideal-command-pose reference for the v4 hard appearance failures.

This isolates whether the finite historical pose stream, rather than the
owner-assist/relook policy itself, prevents fresh final confirmation. Ideal
command pose is a diagnostic upper bound, not a deployable sensor result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PARENT = ART / "replica_policy_v4_cached_three_precommit.json"
PRECOMMIT = ART / "replica_v4_pose_ablation_precommit.json"
REPORT = ART / "replica_v4_pose_ablation_report.json"
OUT = ROOT / "assets/real_indoor_scenes/replica_v4_pose_ablation_20261015"
POLICY = HERE / "run_replica_owner_task_v4.py"
SCENES = ("apartment_0", "hotel_0")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen() -> dict:
    parent = json.loads(PARENT.read_text())
    cases = [{**case, "arms": ["healthy"]} for case in parent["cases"]
             if case["scene"] in SCENES and case["variant"] ==
                "lookalike_accurate"]
    if len(cases) != 2:
        raise ValueError("Changed hard-case source pair")
    sources = [Path(__file__), POLICY, HERE / "replica_owner_help.py"]
    return {"schema": "ape.replica-v4-pose-ablation-precommit.v1",
        "parent_precommit": str(PARENT.relative_to(ROOT)),
        "parent_sha256": sha(PARENT),
        "policy_sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                           for p in sources],
        "cases": cases,
        "difference": "Replace finite historical native-shaped pose packets with ideal continuous command pose; retain same v4 image policy, source rooms, target/distractor, cool-dim photometry, accurate scripted owner point, range-scan correction, 10Hz independent near-body range and guarded motion. This is an upper-bound diagnostic, not a physical IMU/joint sensor test.",
        "score": ["whole-task completion", "fresh final owner answer",
                  "safe route and range health", "relook outcome", "pose horizon"]}


def run(case: dict) -> dict:
    scene = case["scene"]
    folder = OUT / scene
    folder.mkdir(parents=True, exist_ok=True)
    result_file = folder / "attempt.json"
    if result_file.is_file():
        return json.loads(result_file.read_text())
    cache = Path(os.environ.get("REPLICA_CACHE", "/root/replica-sdk-src"))
    source = Path(os.environ.get("REPLICA_SOURCE", "/root/replica-source")) / scene
    command = [sys.executable, str(POLICY),
        "--renderer", str(cache / "build/ReplicaSDK/ReplicaRenderer"),
        "--source-scene", str(source),
        "--converted-mesh", str(source / "mesh_yup.ply"),
        "--reference", str((ROOT / case["reference_trace"]).parent),
        "--scanned-person", str(ROOT /
            "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"),
        "--output", str(folder), "--device", "cuda",
        "--range-gap-stage", "none", "--route-fallback",
        "--target-object", case["target"],
        "--distractor-object", case["distractor"],
        "--search-photometry", "cool_dim", "--owner-response", "accurate",
        "--pose-source", "command", "--pose-correction", "range_scan"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(cache / "build/ReplicaSDK"),
        str(cache / "3rdparty/Pangolin/build/src"),
        env.get("LD_LIBRARY_PATH", ""))))
    try:
        call = subprocess.run(command, cwd=cache / "ReplicaSDK", env=env,
                              capture_output=True, text=True, timeout=1800)
        summary = folder / "summary.json"
        row = {"scene": scene, "exit_code": call.returncode,
               "summary": str(summary.relative_to(ROOT)) if summary.is_file() else None,
               "outcome": json.loads(summary.read_text()) if summary.is_file() else None,
               "error_tail": call.stderr[-1200:] if call.returncode else None}
    except subprocess.TimeoutExpired:
        row = {"scene": scene, "exit_code": None, "summary": None,
               "outcome": None, "error_tail": "runner_timeout_1800s"}
    result_file.write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    definition = frozen()
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen pose ablation changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if "--freeze-only" in sys.argv:
        print(PRECOMMIT.relative_to(ROOT), len(definition["cases"]))
        return
    rows = []
    for case in definition["cases"]:
        if sha(ROOT / case["reference_trace"]) != case["reference_trace_sha256"]:
            raise ValueError("Source route changed")
        row = run(case)
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.replica-v4-pose-ablation-report.v1",
            "precommit": str(PRECOMMIT.relative_to(ROOT)), "rows": rows},
            indent=2) + "\n")
        outcome = row["outcome"] or {}
        print(case["scene"], row["exit_code"],
              outcome.get("walk_result"), outcome.get("final_action"), flush=True)


if __name__ == "__main__":
    main()
