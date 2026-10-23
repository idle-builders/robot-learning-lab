"""Compare owner-assisted task behavior on three retained photographic scans.

These are development-exposed rooms, not a blind holdout. Run on ephemeral
Runpod compute against the three retained official PTex source archives.
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
ROSTER = ROOT / "assets/real_indoor_scenes/real_capture_40_current_candidate_set.json"
USED = ART / "alt_pair_compound_20_precommit.json"
SOURCE_MANIFEST = ART / "replica_source_cache_manifest_20261015.json"
PRECOMMIT = ART / "replica_policy_v4_cached_three_precommit.json"
REPORT = ART / "replica_policy_v4_cached_three_report.json"
OUT = ROOT / "assets/real_indoor_scenes/replica_policy_v4_cached_three_20261015"
POLICY = HERE / "run_replica_owner_task_v4.py"
ARMS = {"healthy": "none", "final_gap": "final"}
POSE_SOURCE = "varied150-forward-seed144"
VARIANTS = {
    "original_nominal": {"target": "Parmesan", "distractor": "CreamCheese",
                         "search_photometry": "none", "owner_response": "accurate",
                         "arms": ["healthy", "final_gap"]},
    "lookalike_accurate": {"target": "Peaches", "distractor": "Pineapple",
                           "search_photometry": "cool_dim",
                           "owner_response": "accurate",
                           "arms": ["healthy", "final_gap"]},
    "lookalike_wrong": {"target": "Peaches", "distractor": "Pineapple",
                        "search_photometry": "cool_dim",
                        "owner_response": "wrong", "arms": ["healthy"]},
    "lookalike_silent": {"target": "Peaches", "distractor": "Pineapple",
                         "search_photometry": "cool_dim",
                         "owner_response": "silent", "arms": ["healthy"]},
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cases() -> list[dict]:
    rows = []
    for scene in ("apartment_0", "hotel_0", "office_3"):
        path = ROOT / f"assets/real_indoor_scenes/replica/ptex_routes/{scene}/ape_height_route.mp4"
        trace = path.parent / "trace.json"
        if not path.is_file() or not trace.is_file():
            raise ValueError(f"Missing source route for {scene}")
        rows.append({"scene": scene,
                     "capture_group": f"replica/{scene}",
                     "category": "official Replica PTex photographic scan",
                     "reference_trace": relative(trace),
                     "reference_trace_sha256": sha(trace),
                     "moving_source_video": relative(path)})
    return [{**row, "variant": variant, **settings}
            for row in rows for variant, settings in VARIANTS.items()]


def frozen() -> dict:
    sources = [Path(__file__), POLICY, HERE / "replica_owner_help.py",
        HERE / "replica_capture_scene.py",
        HERE / "interactive_replica_ptex.py", HERE / "independent_range_monitor.py",
        HERE / "native_shaped_pose.py", HERE / "range_pose_corrector.py",
        HERE / "run_hidden_item_rearrangement.py",
        HERE / "run_real_capture_teach_find.py"]
    cache_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_archives = cache_manifest["compute_cache"][
        "selected_official_scene_archives_sha256"]
    if set(source_archives) != {"apartment_0.tar.zst", "hotel_0.tar.zst",
                                "office_3.tar.zst"}:
        raise ValueError("The three official source archives changed")
    return {"schema": "ape.replica-policy-v4-cached-three-precommit.v1",
        "purpose": "Test one conditional ask-point-confirm policy in the three retained official photographic source archives, crossing nominal and lookalike/cool-dim items with accurate, wrong and silent owner replies; retain all whole-task and setup outcomes.",
        "official_source_manifest": relative(SOURCE_MANIFEST),
        "official_source_manifest_sha256": sha(SOURCE_MANIFEST),
        "official_source_archives_sha256": source_archives,
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in sources],
        "cases": cases(), "arms": ARMS, "variants": VARIANTS,
        "models": {"detector": {"repo": "IDEA-Research/grounding-dino-tiny",
            "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71"},
            "retriever": {"repo": "facebook/dinov2-small",
            "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056"}},
        "baseline": "artifacts/ape-real-room-next-captures-20261014/replica_policy_v3_seven_audit.json",
        "policy": "Same owner, image models, three-station conditional search, historical motion packets, 10Hz independent range health, range-scan pose correction and bounded 17cm-proxy walking. If the present item is unknown or the owner rejects a wrong candidate, ask once for a point in an already captured RGB-D frame. The actor selects only a point-containing depth-localized proposal, refreshes appearance on owner confirmation, and otherwise retains unknown identity with no goal. An owner rejects absent-item candidates. A final fresh owner review gates any found claim. Compare accurate, wrong and silent replies without per-room thresholds.",
        "score": ["three source setups", "two frozen item/light conditions per room",
            "owner/teaching completion",
            "present and absent identity", "wrong questions and false claims", "owner-help burden", "mobile search route",
            "observed standoff", "fresh final owner answer", "relook recovery",
            "source-gap claim withholding",
            "RGB/depth/range/pose/video traces"],
        "limits": "Replica PTex photographic static background; scripted owner points chosen with evaluator identity, static staged props, ideal aligned depth/yaw, static-room-only independent range and 17cm kinematic proxy. Apartment_0 and hotel_0 were already exposed by v3; office_3 is in an earlier photographed-route cohort. Development evidence, not a blind holdout. No moving human, qualified selected body or measured physical stop."}


def run_one(case: dict, arm: str) -> dict:
    scene = case["scene"]
    out = OUT / scene / case["variant"] / arm
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
        "--scanned-person", str(person),
        "--output", str(out), "--device", "cuda",
        "--range-gap-stage", ARMS[arm], "--route-fallback",
        "--target-object", case["target"],
        "--distractor-object", case["distractor"],
        "--search-photometry", case["search_photometry"],
        "--owner-response", case["owner_response"],
        "--pose-source", POSE_SOURCE, "--pose-correction", "range_scan"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(cache / "build/ReplicaSDK"),
        str(cache / "3rdparty/Pangolin/build/src"),
        env.get("LD_LIBRARY_PATH", ""))))
    try:
        result = subprocess.run(command, cwd=cache / "ReplicaSDK", env=env,
                                capture_output=True, text=True, timeout=1800)
        summary = out / "summary.json"
        row = {"scene": scene, "variant": case["variant"],
            "capture_group": case["capture_group"],
            "owner_response": case["owner_response"],
            "arm": arm, "exit_code": result.returncode,
            "summary": relative(summary) if summary.is_file() else None,
            "outcome": json.loads(summary.read_text()) if summary.is_file() else None,
            "error_tail": result.stderr[-1200:] if result.returncode else None}
    except subprocess.TimeoutExpired:
        row = {"scene": scene, "variant": case["variant"],
            "capture_group": case["capture_group"],
            "owner_response": case["owner_response"],
            "arm": arm, "exit_code": None, "summary": None,
            "outcome": None, "error_tail": "runner_timeout_1800s"}
    attempt.write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--scene")
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    definition = frozen()
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen cases or policy source changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(definition["cases"]))
        return
    rows = []
    launched = 0
    for case in definition["cases"]:
        if args.scene and case["scene"] != args.scene:
            continue
        if args.variant and case["variant"] != args.variant:
            continue
        if sha(ROOT / case["reference_trace"]) != case["reference_trace_sha256"]:
            raise ValueError(f"Source route changed: {case['scene']}")
        for arm in case["arms"]:
            path = OUT / case["scene"] / case["variant"] / arm / "attempt.json"
            if args.max_new and launched >= args.max_new and not path.is_file():
                break
            if not path.is_file():
                launched += 1
            row = run_one(case, arm)
            rows.append(row)
            REPORT.write_text(json.dumps({"schema": "ape.replica-policy-v4-cached-three-report.v1",
                "precommit": relative(PRECOMMIT), "rows": rows}, indent=2) + "\n")
            print(case["scene"], case["variant"], arm, row["exit_code"],
                  (row["outcome"] or {}).get("walk_result"), flush=True)


if __name__ == "__main__":
    main()
