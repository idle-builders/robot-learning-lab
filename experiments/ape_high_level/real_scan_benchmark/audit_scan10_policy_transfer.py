"""Audit complete capture-disjoint episodes under the frozen Scan20 policy."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from audit_scan20_shared_owner_policy import digest, video_frames


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan10_policy_transfer_sources.json"
DEVELOPMENT = ART / "scan20_shared_policy_sources.json"
PRECOMMIT = ART / "scan10_policy_transfer_precommit.json"
OUT = ROOT / "assets/real_indoor_scenes/scan10_policy_transfer_20261015"
AUDIT = ART / "scan10_policy_transfer_audit.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    prior_groups = {x["capture_group"] for x in
                    json.loads(DEVELOPMENT.read_text())["sources"]}
    frozen = json.loads(PRECOMMIT.read_text())
    issues = []
    policy_exceptions = []
    if len(source["sources"]) != 10 or len({x["capture_group"] for x in
            source["sources"]}) != 10:
        issues.append("transfer set is not ten distinct captures")
    if prior_groups.intersection(x["capture_group"] for x in source["sources"]):
        issues.append("transfer capture overlaps development set")
    if digest(SOURCE) != frozen["source_sha256"]:
        issues.append("frozen transfer source changed")
    for name, expected in (("shared_policy_source", "shared_policy_sha256"),
                           ("runner_source", "runner_sha256")):
        path = ROOT / frozen[name]
        if not path.is_file() or digest(path) != frozen[expected]:
            issues.append(f"frozen source changed: {frozen[name]}")
    rows = []
    media = []
    json_count = 0
    for case in source["sources"]:
        key = case["scene_key"]
        folder = OUT / key / case["photometry"] / "healthy"
        summary = folder / "summary.json"
        if not summary.is_file():
            issues.append(f"missing episode: {key}")
            continue
        row = json.loads(summary.read_text())
        rows.append(row)
        if row.get("scene_key") != key or row.get("capture_group") != (
                case["capture_group"]) or row.get("photometry") != (
                case["photometry"]):
            issues.append(f"episode identity mismatch: {key}")
        if row.get("result") == "runner_failed":
            if "Teaching click selected no visible proposal" in row.get("error", ""):
                policy_exceptions.append({"scene_key": key,
                    "kind": "teaching_crop_unavailable",
                    "error": row["error"]})
            else:
                issues.append(f"unexpected runner failure: {key}: {row.get('error')}")
        if row.get("motion_authorized") and not row.get("walk_result"):
            issues.append(f"motion authorized without walk: {key}")
        packet_file = folder / "independent_range/packets.json"
        if row.get("owner_status") in (
                "teach_find_completed", "teach_find_completed_far_viewpoint"):
            if not packet_file.is_file():
                issues.append(f"missing independent range: {key}")
            else:
                packets = json.loads(packet_file.read_text()).get("packets", [])
                if len(packets) < 2 or any(p.get("source_health") == "missing"
                                           for p in packets):
                    issues.append(f"bad healthy range stream: {key}")
        for path in folder.rglob("*.json"):
            try:
                json.loads(path.read_text())
                json_count += 1
            except Exception as exc:
                issues.append(f"bad JSON: {path.relative_to(ROOT)}: {exc}")
        for path in folder.rglob("*.mp4"):
            try:
                frames = video_frames(path)
                if frames < 1:
                    raise ValueError("no decoded frames")
                media.append({"path": str(path.relative_to(ROOT)),
                              "sha256": digest(path), "frames": frames})
            except Exception as exc:
                issues.append(f"unreadable video: {path.relative_to(ROOT)}: {exc}")
    audit = {"schema": "ape.scan10-policy-transfer-audit.v1",
        "source_sha256": digest(SOURCE),
        "precommit_sha256": digest(PRECOMMIT),
        "capture_groups": len({r.get("capture_group") for r in rows}),
        "development_overlap": sorted(prior_groups.intersection(
            x["capture_group"] for x in source["sources"])),
        "attempts": len(rows),
        "outcomes": dict(Counter(r.get("result") for r in rows)),
        "policy_exceptions": policy_exceptions,
        "owner_viewpoint_not_reached": sum(r.get("owner_status") ==
            "owner_viewpoint_not_reached" for r in rows),
        "owner_point_used": sum(bool(r.get("owner_point_used")) for r in rows),
        "wrong_candidate_questions": sum(int(r.get("wrong_candidate_questions", 0))
            for r in rows),
        "motion_authorized": sum(bool(r.get("motion_authorized")) for r in rows),
        "observed_standoff": sum(r.get("walk_result") ==
            "observed_target_standoff" for r in rows),
        "final_confirmed": sum(r.get("final_action") ==
            "owner_confirmed_visual_identity" for r in rows),
        "new_pair_result": next((r.get("result") for r in rows if
            r.get("scene_key") == "capture_e2974710c05b456681cbebe768d0db68"), None),
        "rows": rows, "json_files_parsed": json_count,
        "videos_decoded": len(media), "video_frames": sum(m["frames"] for m in media),
        "media": media, "issues": issues,
        "limits": source["limits"]}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("capture_groups", "attempts",
        "outcomes", "policy_exceptions", "owner_viewpoint_not_reached",
        "owner_point_used", "wrong_candidate_questions",
        "observed_standoff", "final_confirmed", "new_pair_result",
        "json_files_parsed", "videos_decoded", "issues")}, indent=2))


if __name__ == "__main__":
    main()
