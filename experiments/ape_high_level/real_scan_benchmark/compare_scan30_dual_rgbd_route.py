"""Compare paired full tasks using two navigation-sensor bundles."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
HEALTHY = ART / "scan30_dual_worker_healthy_report.json"
DUAL = ART / "scan30_dual_rgbd_route_report.json"
HEALTHY_AUDIT = ART / "scan30_dual_worker_healthy_audit.json"
DUAL_AUDIT = ART / "scan30_dual_rgbd_route_audit.json"
PROVENANCE = ART / "scan30_dual_rgbd_provenance_audit.json"
REPORT = ART / "scan30_dual_rgbd_route_comparison.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    h = json.loads(HEALTHY.read_text())["rows"]
    d = json.loads(DUAL.read_text())["rows"]
    ha = json.loads(HEALTHY_AUDIT.read_text())
    da = json.loads(DUAL_AUDIT.read_text())
    p = json.loads(PROVENANCE.read_text())
    hp = json.loads((ART / "scan30_dual_worker_healthy_precommit.json").read_text())
    dp = json.loads((ART / "scan30_dual_rgbd_route_precommit.json").read_text())
    issues = []
    if ha["issues"] or da["issues"] or p["issues"]:
        issues.append("A paired run failed its independent audit")
    if sha(ROOT / hp["wrapper_source"]) != hp["wrapper_sha256"]:
        issues.append("Healthy wrapper changed")
    if sha(ART / hp["paired_dual_precommit"].split("/")[-1]) != hp[
            "paired_dual_precommit_sha256"]:
        issues.append("Dual source precommit changed")
    for source in dp["policy_sources"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            issues.append(f"Dual policy source changed: {source['path']}")
    if len(h) != 30 or len(d) != 30:
        issues.append("Not thirty paired whole tasks")
    rows = []
    for reference, trial in zip(h, d):
        if reference["scene_key"] != trial["scene_key"] or reference[
                "capture_group"] != trial["capture_group"]:
            issues.append(f"Pair identity mismatch: {reference['scene_key']}")
            continue
        rows.append({"scene_key": reference["scene_key"],
            "capture_group": reference["capture_group"],
            "healthy_result": reference["result"],
            "dual_rgbd_result": trial["result"],
            "healthy_owner_status": reference.get("owner_status"),
            "dual_rgbd_owner_status": trial.get("owner_status"),
            "healthy_walk_result": reference.get("walk_result"),
            "dual_rgbd_walk_result": trial.get("walk_result"),
            "healthy_show_again": bool(reference.get("show_summary")),
            "dual_rgbd_show_again": bool(trial.get("show_summary")),
            "healthy_wrong_candidate_questions": reference.get(
                "wrong_candidate_questions"),
            "dual_rgbd_wrong_candidate_questions": trial.get(
                "wrong_candidate_questions")})
    result = {"schema": "ape.scan30-dual-rgbd-route-comparison.v1",
        "paired_cases": len(rows),
        "healthy_outcomes": dict(Counter(r["healthy_result"] for r in rows)),
        "dual_rgbd_outcomes": dict(Counter(r["dual_rgbd_result"] for r in rows)),
        "both_completed": sum(r["healthy_result"] == "completed" and
                              r["dual_rgbd_result"] == "completed" for r in rows),
        "healthy_only_completed": sum(r["healthy_result"] == "completed" and
                                      r["dual_rgbd_result"] != "completed" for r in rows),
        "dual_only_completed": sum(r["healthy_result"] != "completed" and
                                   r["dual_rgbd_result"] == "completed" for r in rows),
        "dual_unknown_clearance_holds": p["unknown_clearance_holds"],
        "healthy_show_again_attempts": sum(r["healthy_show_again"] for r in rows),
        "dual_show_again_attempts": sum(r["dual_rgbd_show_again"] for r in rows),
        "healthy_wrong_candidate_questions": sum(r[
            "healthy_wrong_candidate_questions"] or 0 for r in rows),
        "dual_wrong_candidate_questions": sum(r[
            "dual_rgbd_wrong_candidate_questions"] or 0 for r in rows),
        "rows": rows, "issues": issues,
        "interpretation_limit": "Both arms use the frozen owner teach/find/relook policy, the same 30 capture groups and one A40/Python runtime. The dual arm changes owner and item route evidence together, so a task difference cannot be assigned to only one stage. Both retain ideal RGB-D item localization, historical native-shaped pose increments, a separate virtual 10 Hz near-body range gate, static room meshes and a scripted owner. This is not pure camera-only sensing or physical 50 cm sensor qualification."}
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("paired_cases", "healthy_outcomes",
        "dual_rgbd_outcomes", "both_completed", "healthy_only_completed",
        "dual_only_completed", "dual_unknown_clearance_holds",
        "healthy_show_again_attempts", "dual_show_again_attempts", "issues")},
        indent=2))


if __name__ == "__main__":
    main()
