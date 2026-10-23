"""Audit live fault exposure and retained media in the matched Scan30 run."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from audit_scan20_shared_owner_policy import digest, video_frames


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "scan30_walk_range_gap_precommit.json"
REPORT = ART / "scan30_walk_range_gap_report.json"
AUDIT = ART / "scan30_walk_range_gap_audit.json"
OUT = ROOT / "assets/real_indoor_scenes/scan30_walk_range_gap_20261015"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    report = json.loads(REPORT.read_text())
    rows = report["rows"]
    cases = frozen["cases"]
    issues = []
    if len(rows) != 30 or report["processed"] != 30 or len(cases) != 30:
        issues.append("Not 30 matched full-task attempts")
    if len({c["capture_group"] for c in cases}) != 30:
        issues.append("Capture group duplicated")
    if digest(ROOT / frozen["reference_report"]) != frozen[
            "reference_sha256"]:
        issues.append("Healthy reference report changed")
    reference_precommit = ART / "scan30_unified_policy_precommit.json"
    if digest(reference_precommit) != frozen["reference_precommit_sha256"]:
        issues.append("Healthy reference precommit changed")
    reference = json.loads(reference_precommit.read_text())
    for source in reference["policy_sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or digest(path) != source["sha256"]:
            issues.append(f"Healthy reference source changed: {source['path']}")
    for source in frozen["policy_sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or digest(path) != source["sha256"]:
            issues.append(f"Frozen policy source changed: {source['path']}")
    exposure_rows = []
    for case, row in zip(cases, rows):
        key = case["scene_key"]
        if row.get("scene_key") != key or row.get("capture_group") != \
                case["capture_group"]:
            issues.append(f"Identity/order mismatch: {key}")
        saved = OUT / key / "combined_summary.json"
        if not saved.is_file() or json.loads(saved.read_text()) != row:
            issues.append(f"Combined summary mismatch: {key}")
        base_path = ROOT / row.get("base_summary", "")
        if not base_path.is_file():
            issues.append(f"Missing from-start episode: {key}")
            continue
        base = json.loads(base_path.read_text())
        if base.get("scene_key") != key:
            issues.append(f"Base episode identity mismatch: {key}")
        show = json.loads((ROOT / row["show_summary"]).read_text()) if row.get(
            "show_summary") else None
        exposure = row.get("fault_exposure")
        observed = False
        stop = None
        if exposure == "direct_walk":
            events = row.get("fault_events") or []
            packets_path = base_path.parent / "independent_range/packets.json"
            if len(events) != 1 or not packets_path.is_file():
                issues.append(f"Missing direct-walk injection or packets: {key}")
            else:
                packets = json.loads(packets_path.read_text())["packets"]
                observed = any(p["source_health"] == "missing" and
                    p["capture_wall_s"] >= events[0]["command_wall_s"]
                    for p in packets)
                stop = base.get("walk_result")
                if not observed or not events[0]["observed_missing_packet"] \
                        or stop != "source_gap":
                    issues.append(f"Direct walk failed to stop on source gap: {key}")
        elif exposure in ("show_walk", "show_final"):
            action = (show or {}).get("action") or {}
            if action.get("gap_stage") != exposure.removeprefix("show_"):
                issues.append(f"Show fault stage mismatch: {key}")
            packets_path = (ROOT / row["show_summary"]).parent / \
                "continuation_action" / key / "range_gap" / \
                "independent_range_packets.json"
            if not packets_path.is_file():
                issues.append(f"Missing show range packets: {key}")
            else:
                packets = json.loads(packets_path.read_text())["packets"]
                observed = any(p["source_health"] == "missing" for p in packets)
                if not observed:
                    issues.append(f"No show missing packet: {key}")
            stop = action.get("walk_result")
            if exposure == "show_walk" and stop != "source_gap":
                issues.append(f"Show walk did not stop: {key}")
            if exposure == "show_final" and action.get("final_action") == \
                    "owner_confirmed_visual_identity":
                issues.append(f"Final claim during show range gap: {key}")
        elif exposure == "unexposed":
            if row.get("fault_observed"):
                issues.append(f"Unexposed row claims fault: {key}")
        else:
            issues.append(f"Unknown fault exposure: {key}: {exposure}")
        if observed and row.get("result") == "completed":
            issues.append(f"Completed task despite observed source outage: {key}")
        exposure_rows.append({"scene_key": key, "exposure": exposure,
                              "fault_observed": observed, "walk_result": stop,
                              "result": row.get("result")})
    media = []
    json_files = 0
    for path in OUT.rglob("*.json"):
        try:
            json.loads(path.read_text())
            json_files += 1
        except Exception as error:
            issues.append(f"Unreadable JSON: {path.relative_to(ROOT)}: {error}")
    for path in OUT.rglob("*.mp4"):
        try:
            frames = video_frames(path)
            if frames < 1:
                raise ValueError("No decoded frames")
            media.append({"path": str(path.relative_to(ROOT)),
                          "sha256": digest(path), "frames": frames})
        except Exception as error:
            issues.append(f"Unreadable video: {path.relative_to(ROOT)}: {error}")
    audit = {"schema": "ape.scan30-walk-range-gap-audit.v1",
        "precommit_sha256": digest(PRECOMMIT), "report_sha256": digest(REPORT),
        "attempts": len(rows),
        "outcomes": dict(Counter(r.get("result") for r in rows)),
        "exposures": dict(Counter(r.get("fault_exposure") for r in rows)),
        "observed_faults": sum(r["fault_observed"] for r in exposure_rows),
        "rows": exposure_rows, "json_files_parsed": json_files,
        "videos_decoded": len(media),
        "video_frames": sum(x["frames"] for x in media),
        "media": media, "issues": issues, "limits": frozen["limits"]}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("attempts", "outcomes",
        "exposures", "observed_faults", "json_files_parsed",
        "videos_decoded", "video_frames", "issues")}, indent=2))


if __name__ == "__main__":
    main()
