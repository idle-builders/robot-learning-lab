"""Audit the whole-task recovery pass and its three owner reshow continuations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from audit_scan20_shared_owner_policy import digest, video_frames


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan10_policy_transfer_sources.json"
RECOVERY_PRECOMMIT = ART / "scan10_recovery_policy_precommit.json"
RECOVERY_REPORT = ART / "scan10_recovery_policy_report.json"
SHOW_PRECOMMIT = ART / "scan10_show_again_precommit.json"
SHOW_REPORT = ART / "scan10_show_again_report.json"
RECOVERY_OUT = ROOT / "assets/real_indoor_scenes/scan10_recovery_policy_20261015"
SHOW_OUT = ROOT / "assets/real_indoor_scenes/scan10_show_again_20261015"
AUDIT = ART / "scan10_recovery_audit.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    original = json.loads(RECOVERY_REPORT.read_text())
    continued = json.loads(SHOW_REPORT.read_text())
    frozen = json.loads(RECOVERY_PRECOMMIT.read_text())
    show_frozen = json.loads(SHOW_PRECOMMIT.read_text())
    issues = []
    if len(source["sources"]) != 10 or len({x["capture_group"] for x in
            source["sources"]}) != 10:
        issues.append("not ten distinct capture groups")
    if digest(SOURCE) != frozen["source_sha256"] or digest(SOURCE) != \
            show_frozen["source_sha256"]:
        issues.append("source manifest changed")
    if digest(RECOVERY_REPORT) != show_frozen["prior_report_sha256"]:
        issues.append("precommitted recovery report changed")
    for definition in (frozen, show_frozen):
        definitions = (definition.get("policy_sources") or [
            {"path": definition["base_policy_source"],
             "sha256": definition["base_policy_sha256"]},
            {"path": definition["recovery_source"],
             "sha256": definition["recovery_sha256"]}])
        for item in definitions:
            path = ROOT / item["path"]
            if not path.is_file() or digest(path) != item["sha256"]:
                issues.append(f"policy source changed: {item['path']}")
    base_rows = {r["scene_key"]: r for r in original["rows"]}
    show_rows = {r["scene_key"]: r for r in continued["rows"]}
    if len(base_rows) != 10 or len(show_rows) != 3:
        issues.append("unexpected row count")
    intended = {c["scene_key"] for c in source["sources"]}
    if set(base_rows) != intended or not set(show_rows).issubset(intended):
        issues.append("scene identity mismatch")
    if set(show_rows) != {r["scene_key"] for r in original["rows"] if
            (r.get("final_action") or "").startswith("ask_owner_show_again") }:
        issues.append("show-again set differs from all requested stops")
    outcomes = []
    for case in source["sources"]:
        key = case["scene_key"]
        base = base_rows.get(key, {})
        show = show_rows.get(key)
        if base.get("capture_group") != case["capture_group"]:
            issues.append(f"base capture mismatch: {key}")
        if show and show.get("capture_group") != case["capture_group"]:
            issues.append(f"show capture mismatch: {key}")
        final = show if show else base
        result = final.get("result")
        action = (final.get("action") or {}) if show else base
        if result == "completed":
            if not (action.get("motion_authorized") and
                    action.get("walk_result") == "observed_target_standoff" and
                    action.get("final_action") == "owner_confirmed_visual_identity"):
                issues.append(f"completed without guarded walk/review: {key}")
            health = (action.get("find_source_health") if show else
                      action.get("fresh_source_health")) or {}
            if not health.get("authorized"):
                issues.append(f"completed with unhealthy source: {key}")
        if result in ("runner_failed", "continuation_failed"):
            issues.append(f"runner failure: {key}: {final.get('error')}")
        outcomes.append({"scene_key": key, "capture_group": case["capture_group"],
                         "base_result": base.get("result"),
                         "continuation_result": show.get("result") if show else None,
                         "combined_result": result,
                         "motion_authorized": bool(action.get("motion_authorized")),
                         "observed_standoff": action.get("walk_result") ==
                             "observed_target_standoff",
                         "final_action": action.get("final_action")})
    media = []
    json_count = 0
    for folder in (RECOVERY_OUT, SHOW_OUT):
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
                issues.append(f"bad video: {path.relative_to(ROOT)}: {exc}")
    counts = Counter(x["combined_result"] for x in outcomes)
    audit = {"schema": "ape.scan10-recovery-audit.v1",
             "source_sha256": digest(SOURCE),
             "recovery_precommit_sha256": digest(RECOVERY_PRECOMMIT),
             "show_precommit_sha256": digest(SHOW_PRECOMMIT),
             "capture_groups": len({x["capture_group"] for x in outcomes}),
             "whole_task_attempts": len(outcomes),
             "show_again_continuations": len(show_rows),
             "outcomes": dict(counts), "outcome_rows": outcomes,
             "wrong_candidate_questions_reported": sum(int(r.get(
                 "wrong_candidate_questions", 0)) for r in base_rows.values()) +
                 sum(int(((r.get("teaching") or {}).get(kind) or {}).get(
                     "wrong_candidate_offered", False)) for r in
                     show_rows.values() for kind in ("present", "absent")),
             "json_files_parsed": json_count,
             "videos_decoded": len(media),
             "video_frames": sum(x["frames"] for x in media),
             "media": media, "issues": issues,
             "limits": show_frozen["limits"]}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("capture_groups",
        "whole_task_attempts", "show_again_continuations", "outcomes",
        "wrong_candidate_questions_reported", "json_files_parsed",
        "videos_decoded", "video_frames", "issues")}, indent=2))


if __name__ == "__main__":
    main()
