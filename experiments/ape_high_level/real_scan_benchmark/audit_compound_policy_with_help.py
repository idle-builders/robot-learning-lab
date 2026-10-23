"""Join complete compound-fault episodes with one fallible owner-help turn."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_compound_owner_point_recovery import SOURCES, PRECOMMIT as HELP_PRECOMMIT
from audit_compound_owner_point_recovery import AUDIT as HELP_AUDIT


OUT = ART / "compound_policy_with_help_audit.json"


def main() -> None:
    help_frozen = json.loads(HELP_PRECOMMIT.read_text())
    help_audit = json.loads(HELP_AUDIT.read_text())
    if help_audit["summary"]["runs"] != 24:
        raise ValueError("Incomplete owner-help comparison")
    for source in help_frozen["source_audits"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Original owner episodes changed")
    rows = []
    groups = []
    for source_path, _ in SOURCES:
        source = json.loads(source_path.read_text())
        group = "development_20" if len(source["rows"]) == 20 else "different_3"
        groups.append({"name": group, "source": relative(source_path),
                       "source_sha256": sha(source_path)})
        for episode in source["rows"]:
            key = episode["case"]
            status = episode["status"]
            entry = {"case": key, "group": group, "status": status,
                     "autonomous_present_confirmed": False,
                     "helped_present_confirmed": False,
                     "absent_unknown": False,
                     "wrong_candidate_offers": 0,
                     "wrong_point_candidate_offers": 0,
                     "help_arms": []}
            if status == "teach_find_completed_far_viewpoint":
                present = episode["find_present"]
                absent = episode["find_absent"]
                entry["autonomous_present_confirmed"] = present["action"] == \
                    "owner_confirmed_visual_identity"
                entry["absent_unknown"] = absent["action"] == \
                    "unidentified_ask_new_view_or_help"
                entry["wrong_candidate_offers"] = int(present[
                    "wrong_candidate_offered"]) + int(absent[
                    "wrong_candidate_offered"])
                if not entry["autonomous_present_confirmed"]:
                    help_rows = [r for r in help_audit["rows"] if
                                 r["scene_key"] == key]
                    if len(help_rows) != 4:
                        raise ValueError("Missing fallible owner feedback")
                    by_arm = {r["arm"]: r for r in help_rows}
                    if (by_arm["wrong_present"]["action"] ==
                            "owner_confirmed_visual_identity" or
                            by_arm["wrong_absent"]["action"] ==
                            "owner_confirmed_visual_identity" or
                            by_arm["silent_present"]["action"] ==
                            "owner_confirmed_visual_identity"):
                        raise ValueError("Incorrect owner input made a final claim")
                    entry["helped_present_confirmed"] = by_arm[
                        "accurate_present"]["action"] == \
                        "owner_confirmed_visual_identity"
                    entry["wrong_point_candidate_offers"] = sum(bool(
                        by_arm[arm].get("wrong_candidate_offered")) for arm in
                        ("wrong_present", "wrong_absent"))
                    entry["help_arms"] = [{"arm": r["arm"],
                        "action": r["action"],
                        "fresh_video": r.get("fresh_video")}
                        for r in help_rows]
            rows.append(entry)
    summary = {}
    for group in ("development_20", "different_3"):
        subset = [r for r in rows if r["group"] == group]
        summary[group] = {"rooms": len(subset),
            "autonomous_present_confirmed": sum(r[
                "autonomous_present_confirmed"] for r in subset),
            "additional_confirmed_after_accurate_owner_point": sum(r[
                "helped_present_confirmed"] for r in subset),
            "total_present_confirmed_with_accurate_help": sum(r[
                "autonomous_present_confirmed"] or r[
                "helped_present_confirmed"] for r in subset),
            "absent_unknown_after_search": sum(r["absent_unknown"]
                for r in subset),
            "stopped_before_search": sum(r["status"] !=
                "teach_find_completed_far_viewpoint" for r in subset),
            "wrong_candidate_offers_before_help": sum(r[
                "wrong_candidate_offers"] for r in subset),
            "wrong_point_candidate_offers_rejected_by_owner": sum(r[
                "wrong_point_candidate_offers"] for r in subset)}
    OUT.write_text(json.dumps({"schema":
        "ape.compound-policy-with-help-audit.v1", "groups": groups,
        "owner_help_audit": relative(HELP_AUDIT),
        "owner_help_audit_sha256": sha(HELP_AUDIT),
        "summary": summary, "rows": rows,
        "limits": "Accurate owner point is generated from simulator object identity; wrong point is a different depth-localized visual proposal; silence has no reply. The ensuing owner yes/no is scripted truth. These are static-room continuations from saved owner episodes, not moving-person or physical-body trials."},
        indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
