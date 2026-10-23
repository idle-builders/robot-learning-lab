"""Run the frozen Scan20 decision policy on ten capture-disjoint rooms."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from freeze_scan20_shared_policy_sources import relative, sha


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan10_policy_transfer_sources.json"
ORIGINAL = ART / "scan20_shared_owner_policy_precommit.json"
PRECOMMIT = ART / "scan10_policy_transfer_precommit.json"
OUT = ROOT / "assets/real_indoor_scenes/scan10_policy_transfer_20261015"
REPORT = ART / "scan10_policy_transfer_report.json"
POLICY_SOURCE = Path(__file__).with_name("run_scan20_shared_owner_policy.py")


def frozen_definition(source: dict) -> dict:
    original = json.loads(ORIGINAL.read_text())
    expected = next(x["sha256"] for x in original["policy_sources"]
                    if x["path"] == relative(POLICY_SOURCE))
    if sha(POLICY_SOURCE) != expected:
        raise ValueError("The original Scan20 policy code changed")
    return {"schema": "ape.scan10-policy-transfer-precommit.v1",
        "source_manifest": relative(SOURCE),
        "source_sha256": sha(SOURCE),
        "original_policy_precommit": relative(ORIGINAL),
        "original_policy_precommit_sha256": sha(ORIGINAL),
        "shared_policy_source": relative(POLICY_SOURCE),
        "shared_policy_sha256": expected,
        "runner_source": relative(Path(__file__)),
        "runner_sha256": sha(Path(__file__)),
        "cases": [{"scene_key": x["scene_key"],
            "capture_group": x["capture_group"],
            "item_screen_sha256": x["item_screen_sha256"],
            "source_mesh_sha256": x["source_mesh_sha256"],
            "photometry": x["photometry"]} for x in source["sources"]],
        "policy": "Exactly the frozen Scan20 shared teach/find/approach/review rule, with no threshold, recovery or sensor change. New capture groups and one new grocery pair are evaluator-only scenarios. Any uncertainty is scored in the denominator.",
        "limits": source["limits"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    cases = source["sources"]
    if len(cases) != 10 or len({x["capture_group"] for x in cases}) != 10:
        raise ValueError("Expected ten distinct transfer captures")
    definition = frozen_definition(source)
    if PRECOMMIT.is_file():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen transfer definition changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(cases))
        return
    import run_scan20_shared_owner_policy as policy
    policy.OUT = OUT
    rows = []
    launched = 0
    for case in cases:
        path = OUT / case["scene_key"] / case["photometry"] / "healthy"
        summary = path / "summary.json"
        if sha(ROOT / case["item_screen"]) != case["item_screen_sha256"]:
            raise ValueError(f"Changed item screen: {case['scene_key']}")
        if sha(ROOT / case["source_mesh_resolved"]) != case["source_mesh_sha256"]:
            raise ValueError(f"Changed source mesh: {case['scene_key']}")
        if summary.is_file():
            row = json.loads(summary.read_text())
        elif args.max_new and launched >= args.max_new:
            break
        else:
            try:
                row = policy.run_case(case, case["photometry"], "healthy")
            except Exception as exc:
                row = {"schema": "ape.scan10-policy-transfer.v1",
                    "scene_key": case["scene_key"],
                    "capture_group": case["capture_group"],
                    "photometry": case["photometry"],
                    "arm": "healthy", "result": "runner_failed",
                    "error": repr(exc),
                    "error_traceback": traceback.format_exc()[-3000:]}
                path.mkdir(parents=True, exist_ok=True)
                summary.write_text(json.dumps(row, indent=2) + "\n")
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({
            "schema": "ape.scan10-policy-transfer-report.v1",
            "source_manifest": relative(SOURCE),
            "source_sha256": sha(SOURCE),
            "precommit": relative(PRECOMMIT),
            "processed": len(rows), "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], case["photometry"], row["result"],
              flush=True)


if __name__ == "__main__":
    main()
