"""Audit the frozen 30-room, from-start integrated owner-task pass."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from audit_scan20_shared_owner_policy import digest, video_frames


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "scan30_unified_policy_precommit.json"
REPORT = ART / "scan30_unified_policy_report.json"
OUT = ROOT / "assets/real_indoor_scenes/scan30_unified_policy_20261015"
AUDIT = ART / "scan30_unified_policy_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    report = json.loads(REPORT.read_text())
    cases = frozen["cases"]
    rows = report["rows"]
    issues = []
    if len(cases) != 30 or len(rows) != 30 or report["processed"] != 30:
        issues.append("not thirty complete attempts")
    if len({c["capture_group"] for c in cases}) != 30 or \
            len({r["scene_key"] for r in rows}) != len(rows):
        issues.append("capture or scene duplication")
    for key, source in (("development_sha256", "development_manifest"),
                        ("transfer_sha256", "transfer_manifest")):
        if digest(ROOT / frozen[source]) != frozen[key]:
            issues.append(f"frozen source changed: {source}")
    for source in frozen["policy_sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or digest(path) != source["sha256"]:
            issues.append(f"frozen policy source changed: {source['path']}")
    result_rows = []
    for case, row in zip(cases, rows):
        key = case["scene_key"]
        if any(row.get(field) != case[field] for field in
               ("scene_key", "capture_group", "cohort", "photometry")):
            issues.append(f"row identity/order mismatch: {key}")
        combined = OUT / key / "combined_summary.json"
        if not combined.is_file() or json.loads(combined.read_text()) != row:
            issues.append(f"combined summary mismatch: {key}")
        base_path = ROOT / row.get("base_summary", "")
        if not base_path.is_file():
            issues.append(f"missing fresh base episode: {key}")
            continue
        base = json.loads(base_path.read_text())
        if base.get("scene_key") != key:
            issues.append(f"base identity mismatch: {key}")
        show = None
        if row.get("show_summary"):
            show_path = ROOT / row["show_summary"]
            if not show_path.is_file():
                issues.append(f"missing show-again continuation: {key}")
            else:
                show = json.loads(show_path.read_text())
                if show.get("scene_key") != key or show.get("source_episode") != \
                        row["base_summary"]:
                    issues.append(f"show-again source mismatch: {key}")
                if not (base.get("final_action") or "").startswith(
                        "ask_owner_show_again"):
                    issues.append(f"unrequested show-again continuation: {key}")
        result = row.get("result")
        final = (show.get("action") or {}) if show else base
        if result == "completed":
            if not (row.get("motion_authorized") and
                    row.get("walk_result") == "observed_target_standoff" and
                    row.get("final_action") == "owner_confirmed_visual_identity"):
                issues.append(f"completion without guarded walk and review: {key}")
            find_health = (final.get("find_source_health") if show else
                           final.get("fresh_source_health")) or {}
            review_health = final.get("final_source_health") or {}
            if not find_health.get("authorized") or not review_health.get(
                    "authorized"):
                issues.append(f"completion with unhealthy range source: {key}")
            if show and not (show.get("teaching") or {}).get("teach_confirmed"):
                issues.append(f"completion without re-teaching: {key}")
        if result in ("runner_failed", "continuation_failed"):
            issues.append(f"runtime failure: {key}: {row.get('error')}")
        if row.get("motion_authorized") and row.get("walk_result") is None:
            issues.append(f"authorized motion without walk: {key}")
        result_rows.append({"scene_key": key, "cohort": case["cohort"],
                            "base_result": base.get("result"),
                            "show_result": show.get("result") if show else None,
                            "combined_result": result,
                            "owner_status": row.get("owner_status"),
                            "motion_authorized": row.get("motion_authorized"),
                            "walk_result": row.get("walk_result"),
                            "final_action": row.get("final_action")})
    media = []
    json_count = 0
    wrong_candidates = 0
    owner_questions = 0
    for path in OUT.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
            json_count += 1
            if path.name == "episode_trace.json":
                wrong_candidates += int(bool(data.get("wrong_confident_candidate")))
                owner_questions += int(data.get("owner_questions") or 0)
        except Exception as error:
            issues.append(f"unreadable JSON: {path.relative_to(ROOT)}: {error}")
    for path in OUT.rglob("*.mp4"):
        try:
            frames = video_frames(path)
            if frames < 1:
                raise ValueError("no decoded frames")
            media.append({"path": str(path.relative_to(ROOT)),
                          "sha256": digest(path), "frames": frames})
        except Exception as error:
            issues.append(f"unreadable video: {path.relative_to(ROOT)}: {error}")
    audit = {"schema": "ape.scan30-unified-policy-audit.v1",
        "precommit_sha256": digest(PRECOMMIT),
        "report_sha256": digest(REPORT),
        "capture_groups": len({c["capture_group"] for c in cases}),
        "attempts": len(rows), "outcomes": dict(Counter(r.get("result") for r in rows)),
        "by_cohort": {name: dict(Counter(r.get("result") for r in rows
            if r.get("cohort") == name)) for name in ("development", "prior_transfer")},
        "show_again_attempts": sum(bool(r.get("show_summary")) for r in rows),
        "wrong_candidate_offers_in_stage_traces": wrong_candidates,
        "owner_questions_in_stage_traces": owner_questions,
        "rows": result_rows,
        "json_files_parsed": json_count,
        "videos_decoded": len(media),
        "video_frames": sum(m["frames"] for m in media),
        "media": media, "issues": issues,
        "limits": frozen["limits"]}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("capture_groups", "attempts",
        "outcomes", "by_cohort", "show_again_attempts",
        "wrong_candidate_offers_in_stage_traces", "owner_questions_in_stage_traces",
        "json_files_parsed", "videos_decoded", "video_frames", "issues")},
        indent=2))


if __name__ == "__main__":
    main()
