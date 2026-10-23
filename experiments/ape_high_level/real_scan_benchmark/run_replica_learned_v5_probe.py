"""Frozen cross-source probe of the V6 learned option head in Replica rooms.

The three rooms are development exposed. This script freezes cases and source
hashes before running; it is a transfer probe, not a held-out benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "replica_learned_v5_probe_precommit.json"
REPORT = ART / "replica_learned_v5_probe_report.json"
OUT = ROOT / "assets/real_indoor_scenes/replica_learned_v5_probe_20261016"
POLICY = HERE / "run_replica_owner_task_v5_learned.py"
MODEL = ROOT / "experiments/ape_high_level/task_policy_learning/event_policy_live_v6_owner_route.joblib"
SCENES = ("hotel_0", "apartment_0", "office_3")
ARMS = {"healthy": "none", "final_gap": "final"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def definition() -> dict:
    sources = [POLICY, MODEL, HERE / "run_replica_owner_task_v4.py",
        HERE / "replica_capture_scene.py", HERE / "independent_range_monitor.py",
        HERE / "range_pose_corrector.py", HERE / "native_shaped_pose.py",
        ROOT / "experiments/ape_high_level/task_policy_learning/option_controller.py",
        ROOT / "experiments/ape_high_level/task_policy_learning/train_event_policy.py"]
    cases = []
    for scene in SCENES:
        trace = ROOT / f"assets/real_indoor_scenes/replica/ptex_routes/{scene}/trace.json"
        cases.append({"scene": scene, "reference_trace": rel(trace),
                      "reference_sha256": sha(trace),
                      "arms": list(ARMS)})
    return {"schema": "ape.replica-learned-v5-probe-precommit.v1",
        "purpose": "Transfer the frozen V6 later-task option head into three photographic Replica source rooms with fresh identity, independent range, pose and association observations.",
        "cases": cases, "arms": ARMS,
        "source_hashes": {rel(path): sha(path) for path in sources},
        "limits": "Development-exposed Replica PTex backgrounds; scripted owner/props and authored owner route, teaching, wide search, help and motion. Policy owns post-search relook, motion admission, skill-result and final review. Ideal depth and static-room-only independent range. No selected physical body or blind transfer."}


def run_one(case: dict, arm: str) -> dict:
    scene = case["scene"]
    out = OUT / scene / arm
    out.mkdir(parents=True, exist_ok=True)
    attempt = out / "attempt.json"
    if attempt.is_file():
        return json.loads(attempt.read_text())
    cache = Path(os.environ.get("REPLICA_CACHE", "/root/replica-sdk-src"))
    source = Path(os.environ.get("REPLICA_SOURCE", "/root/replica-source")) / scene
    person = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"
    command = [sys.executable, str(POLICY),
        "--renderer", str(cache / "build/ReplicaSDK/ReplicaRenderer"),
        "--source-scene", str(source),
        "--converted-mesh", str(source / "mesh_yup.ply"),
        "--reference", str((ROOT / case["reference_trace"]).parent),
        "--scanned-person", str(person), "--output", str(out),
        "--device", "cuda", "--range-gap-stage", ARMS[arm],
        "--route-fallback", "--pose-source", "varied150-forward-seed144",
        "--pose-correction", "range_scan", "--option-policy", str(MODEL)]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(cache / "build/ReplicaSDK"),
        str(cache / "3rdparty/Pangolin/build/src"),
        env.get("LD_LIBRARY_PATH", ""))))
    try:
        result = subprocess.run(command, cwd=cache / "ReplicaSDK", env=env,
                                capture_output=True, text=True, timeout=1800)
        summary = out / "summary.json"
        row = {"scene": scene, "arm": arm, "exit_code": result.returncode,
            "summary": rel(summary) if summary.is_file() else None,
            "outcome": json.loads(summary.read_text()) if summary.is_file() else None,
            "error_tail": result.stderr[-1500:] if result.returncode else None}
    except subprocess.TimeoutExpired:
        row = {"scene": scene, "arm": arm, "exit_code": None,
            "summary": None, "outcome": None, "error_tail": "timeout_1800s"}
    attempt.write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    frozen = definition()
    ART.mkdir(parents=True, exist_ok=True)
    if PRECOMMIT.is_file():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen probe definition or source changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    if args.freeze_only:
        print(rel(PRECOMMIT)); return
    rows = []
    launched = 0
    for case in frozen["cases"]:
        if args.scene and case["scene"] != args.scene:
            continue
        if sha(ROOT / case["reference_trace"]) != case["reference_sha256"]:
            raise ValueError("Reference trace changed")
        for arm in case["arms"]:
            attempt = OUT / case["scene"] / arm / "attempt.json"
            if args.max_new and launched >= args.max_new and not attempt.is_file():
                break
            if not attempt.is_file():
                launched += 1
            row = run_one(case, arm)
            rows.append(row)
            REPORT.write_text(json.dumps({
                "schema": "ape.replica-learned-v5-probe-report.v1",
                "precommit": rel(PRECOMMIT), "rows": rows}, indent=2) + "\n")
            print(case["scene"], arm, row["exit_code"],
                  (row["outcome"] or {}).get("learned_task_result"), flush=True)


if __name__ == "__main__":
    main()
