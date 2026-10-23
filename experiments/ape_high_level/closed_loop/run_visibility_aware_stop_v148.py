"""V148: stop at first fresh can acceptance within 37cm actor band."""

from __future__ import annotations

import json
from pathlib import Path

import run_live_wide_route_v147 as prior
from run import ROOT,sha

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v148"
ACTOR_STOP_UPPER_M=.37


def main() -> None:
    result=prior.main(output_root=OUT,actor_stop_upper=ACTOR_STOP_UPPER_M,
                      label="V148")
    audit={"scope":"Prospective V148 37cm actor visual stop on V147 9cm wide routes; all other actor/perception/safety settings unchanged. Two correlated development rooms, evaluator truth afterward only.",
           "source_sha256":sha(SOURCE),
           "runner_source_sha256":sha(prior.SOURCE),
           "actor_stop_upper_m":ACTOR_STOP_UPPER_M,
           "paired_physical_target_area_gate_pass":result["paired_physical_target_area_gate_pass"],
           "rows":result["rows"],
           "paired_trace_audit_sha256":sha(OUT / "live-wide-route-audit.json")}
    (OUT / "visibility-aware-stop-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("V148 gate",audit["paired_physical_target_area_gate_pass"],flush=True)


if __name__=="__main__":main()
