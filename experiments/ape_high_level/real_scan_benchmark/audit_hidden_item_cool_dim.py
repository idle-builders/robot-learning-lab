"""Audit complete hidden-change trials under post-change cool/dim RGB."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART
import audit_hidden_item_rearrangement as audit


audit.OUT = ROOT / "assets/real_indoor_scenes/hidden_item_cool_dim_20261015"
audit.PRECOMMIT = ART / "hidden_item_cool_dim_precommit.json"
audit.REPORT = ART / "hidden_item_cool_dim_report.json"
audit.AUDIT = ART / "hidden_item_cool_dim_audit.json"


if __name__ == "__main__":
    frozen = json.loads(audit.PRECOMMIT.read_text())
    if frozen.get("post_change_photometry") != "cool_dim":
        raise ValueError("Lighting shift missing from frozen trial")
    audit.main()
    report = json.loads(audit.REPORT.read_text())
    for row in report["rows"]:
        folder = audit.OUT / row["scene_key"]
        trace = json.loads((folder / "trace.json").read_text())
        if trace.get("post_change_photometry") != "cool_dim":
            raise ValueError("Lighting shift missing from room trace")
        for branch in ("moved", "absent"):
            episode = json.loads((folder / branch / "recheck/episode_trace.json").read_text())
            for stage in episode["stage_reports"]:
                search_trace = json.loads((ROOT / stage["trace"]).read_text())
                if search_trace["search_photometry"] != "cool_dim":
                    raise ValueError("Search stage was not shifted")
