"""Audit the frozen 20-capture owner-policy batch and its retained evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan20_shared_policy_sources.json"
PRECOMMIT = ART / "scan20_shared_owner_policy_precommit.json"
OUT = ROOT / "assets/real_indoor_scenes/scan20_shared_owner_policy_20261015"
AUDIT = ART / "scan20_shared_owner_policy_audit.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def video_frames(path: Path) -> int:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True, capture_output=True, check=True)
    return int(probe.stdout.strip())


def main() -> None:
    sources = json.loads(SOURCE.read_text())
    frozen = json.loads(PRECOMMIT.read_text())
    issues: list[str] = []
    if len(sources["sources"]) != 20 or len({s["capture_group"] for s in
            sources["sources"]}) != 20:
        issues.append("source cohort is not 20 independent capture groups")
    if digest(SOURCE) != frozen["source_sha256"]:
        issues.append("source manifest changed after freezing")
    for entry in frozen["policy_sources"]:
        path = ROOT / entry["path"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            issues.append(f"policy source changed: {entry['path']}")

    rows = []
    media = []
    json_count = 0
    for index, case in enumerate(sources["sources"]):
        photometry = "cool_dim" if index % 2 else "none"
        for arm in ("healthy", "final_range_gap"):
            folder = OUT / case["scene_key"] / photometry / arm
            summary = folder / "summary.json"
            if not summary.is_file():
                issues.append(f"missing {arm} episode: {case['scene_key']}")
                continue
            row = json.loads(summary.read_text())
            rows.append(row)
            if row.get("scene_key") != case["scene_key"] or row.get(
                    "capture_group") != case["capture_group"] or row.get(
                    "arm") != arm or row.get("photometry") != photometry:
                issues.append(f"episode identity mismatch: {case['scene_key']} {arm}")
            if row.get("result") == "runner_failed":
                issues.append(f"runner failed: {case['scene_key']} {arm}: {row.get('error')}")
            if row.get("motion_authorized") and row.get("walk_result") is None:
                issues.append(f"authorized without walk: {case['scene_key']} {arm}")
            if row.get("final_action") == "owner_confirmed_visual_identity" and (
                    not row.get("motion_authorized") or row.get("walk_result") !=
                    "observed_target_standoff"):
                issues.append(f"final claim without approach: {case['scene_key']} {arm}")
            packets_file = folder / "independent_range/packets.json"
            if row.get("owner_status") in (
                    "teach_find_completed", "teach_find_completed_far_viewpoint"):
                if not packets_file.is_file():
                    issues.append(f"missing independent range stream: {case['scene_key']} {arm}")
                else:
                    stream = json.loads(packets_file.read_text())
                    packets = stream.get("packets", [])
                    if stream.get("period_s") != .1 or len(packets) < 2:
                        issues.append(f"invalid range stream: {case['scene_key']} {arm}")
                    missing_packets = sum(p.get("source_health") == "missing"
                                          for p in packets)
                    if arm == "healthy" and missing_packets:
                        issues.append(f"unexpected healthy-arm source gap: {case['scene_key']}")
                    if row.get("injected_range_gap_observed") and not missing_packets:
                        issues.append(f"reported gap absent from packets: {case['scene_key']}")
            if arm == "final_range_gap" and row.get("walk_result") == (
                    "observed_target_standoff"):
                if not row.get("injected_range_gap_observed"):
                    issues.append(f"injected range gap not observed: {case['scene_key']}")
                elif row.get("final_action") != "withheld_range_source_unhealthy":
                    issues.append(f"final claim during range gap: {case['scene_key']}")
            for path in folder.rglob("*.json"):
                try:
                    json.loads(path.read_text())
                    json_count += 1
                except Exception as exc:
                    issues.append(f"invalid JSON: {path.relative_to(ROOT)}: {exc}")
            for path in folder.rglob("*.mp4"):
                try:
                    frames = video_frames(path)
                    if frames < 1:
                        raise ValueError(f"no decoded frames ({frames})")
                    media.append({"path": str(path.relative_to(ROOT)),
                                  "sha256": digest(path), "frames": frames})
                except Exception as exc:
                    issues.append(f"unreadable video: {path.relative_to(ROOT)}: {exc}")
    healthy = [r for r in rows if r.get("arm") == "healthy"]
    gap = [r for r in rows if r.get("arm") == "final_range_gap"]
    outcomes = {}
    for name, cohort in (("healthy", healthy), ("final_range_gap", gap)):
        outcomes[name] = {"attempts": len(cohort),
            "result_counts": dict(Counter(r.get("result") for r in cohort)),
            "owner_teach_completed": sum(r.get("owner_status") in (
                "teach_find_completed", "teach_find_completed_far_viewpoint")
                for r in cohort),
            "present_confirmed": sum(r.get("resolved_present_action") ==
                "owner_confirmed_visual_identity" for r in cohort),
            "absent_unknown": sum(r.get("resolved_absent_action") ==
                "unidentified_ask_new_view_or_help" for r in cohort),
            "owner_point_used": sum(bool(r.get("owner_point_used")) for r in cohort),
            "wrong_candidate_questions": sum(int(r.get("wrong_candidate_questions", 0))
                for r in cohort),
            "motion_authorized": sum(bool(r.get("motion_authorized")) for r in cohort),
            "observed_standoff": sum(r.get("walk_result") ==
                "observed_target_standoff" for r in cohort),
            "final_confirmed": sum(r.get("final_action") ==
                "owner_confirmed_visual_identity" for r in cohort),
            "final_withheld_range": sum(r.get("final_action") ==
                "withheld_range_source_unhealthy" for r in cohort),
            "gap_observed": sum(bool(r.get("injected_range_gap_observed"))
                for r in cohort)}
    by_arm = {(r.get("scene_key"), r.get("arm")): r for r in rows}
    paired = []
    for case in sources["sources"]:
        key = case["scene_key"]
        a = by_arm.get((key, "healthy"))
        b = by_arm.get((key, "final_range_gap"))
        if a is None or b is None:
            continue
        paired.append({"scene_key": key,
            "same_owner_status": a.get("owner_status") == b.get("owner_status"),
            "same_present_action": a.get("resolved_present_action") == b.get(
                "resolved_present_action"),
            "same_absent_action": a.get("resolved_absent_action") == b.get(
                "resolved_absent_action"),
            "same_walk_result": a.get("walk_result") == b.get("walk_result"),
            "healthy_final": a.get("final_action"),
            "gap_raw_final": b.get("final_action_raw"),
            "gap_gated_final": b.get("final_action")})
    audit = {"schema": "ape.scan20-shared-owner-policy-audit.v1",
        "source_sha256": digest(SOURCE), "precommit_sha256": digest(PRECOMMIT),
        "capture_groups": len({r.get("capture_group") for r in rows}),
        "outcomes": outcomes, "paired": paired,
        "paired_same_walk_result": sum(p["same_walk_result"] for p in paired),
        "json_files_parsed": json_count,
        "videos_decoded": len(media), "single_frame_videos": sum(
            v["frames"] == 1 for v in media),
        "video_frames": sum(v["frames"] for v in media),
        "media": media, "issues": issues,
        "limits": "Static real-capture rooms with inserted props and scripted scanned owner; idealized camera/depth, virtual independent range, historical body-shaped motion, no trained high-level policy or physical qualification."}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("capture_groups", "outcomes",
        "paired_same_walk_result", "json_files_parsed", "videos_decoded",
        "single_frame_videos", "video_frames", "issues")}, indent=2))


if __name__ == "__main__":
    main()
