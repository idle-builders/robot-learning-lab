"""V29 follow-up: audit what the held-pose chooser could know before entry.

This is an information-boundary audit of saved V28 packets and V29 native
hold outcomes, not a new actor or a claim that future person motion is known.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-native-route-guard/v8-robocasa-native"
V28 = BASE / "v28-held-monitor"
V29 = BASE / "v29-native-early-hold"
ENTRY_START_S = 3.10
MARGIN_M = .05
MODES = ("front_140deg_15hz", "ideal_omni_10hz")
SIDES = ("left", "right")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks = {}
    rows = []
    sources = {}
    for mode in MODES:
        for side in SIDES:
            for world in range(4):
                moving_path = V28 / "traces" / f"{mode}-late_{side}_pause-world{world}.json"
                static_path = V28 / "traces" / f"{mode}-static_{side}-world{world}.json"
                moving = json.loads(moving_path.read_text())
                static = json.loads(static_path.read_text())
                prefix = [(a, b) for a, b in zip(moving, static)
                          if a["time_s"] < ENTRY_START_S]
                key = f"{mode}:{side}:world{world}"
                checks[f"nonempty_prefix:{key}"] = bool(prefix)
                checks[f"same_packet_times:{key}"] = all(
                    a["time_s"] == b["time_s"] for a, b in prefix)
                checks[f"same_actor_packets:{key}"] = all(
                    a["actor_packet"] == b["actor_packet"] for a, b in prefix)
                checks[f"same_actor_decisions:{key}"] = all(
                    a["actor_decision"] == b["actor_decision"] for a, b in prefix)
                rows.append({"mode": mode, "side": side, "world": world,
                             "prefix_packet_count": len(prefix),
                             "last_prefix_packet_s": prefix[-1][0]["time_s"],
                             "evaluator_person_ray_hits_in_prefix": sum(
                                 a["evaluator_person_ray_hits"] for a, _ in prefix),
                             "moving_trace_sha256": sha(moving_path),
                             "static_trace_sha256": sha(static_path)})
                sources[str(moving_path.relative_to(ROOT))] = sha(moving_path)
                sources[str(static_path.relative_to(ROOT))] = sha(static_path)

    v29_path = V29 / "report.json"
    v29 = json.loads(v29_path.read_text())
    candidate_rows = []
    by_key = {(r["world"], r["case"], r["mode"]): r
              for r in v29["results"]}
    for world in range(4):
        for side in SIDES:
            case = f"late_{side}_pause"
            early = by_key[world, case, "early_2p5s"]
            late = by_key[world, case, "late_3p0s"]
            gaps = {"early_2p5s": early["min_body_person_gap_m"],
                    "late_3p0s": late["min_body_person_gap_m"]}
            candidate_rows.append({"world": world, "side": side,
                                   "candidate_min_gaps_m": gaps,
                                   "any_candidate_above_5cm": max(gaps.values()) >= MARGIN_M})
    checks["candidate_matrix_complete"] = len(candidate_rows) == 8
    checks["no_candidate_above_5cm_in_four_cases"] = sum(
        not r["any_candidate_above_5cm"] for r in candidate_rows) == 4
    checks["v28_source_audit_pass"] = json.loads(
        (V28 / "audit.json").read_text())["pass"]
    checks["v29_source_audit_pass"] = json.loads(
        (V29 / "audit.json").read_text())["pass"]
    report = {
        "schema": "robot-learning-lab.ape-preentry-hold-choice-audit.v1",
        "task": "T-9v3p3n",
        "pass": all(checks.values()),
        "method": "Compare saved actor packets and decisions for V28 moving and stationary side-person pairs before the 3.10s scripted entry, then count V29 actual native 2.5s/3.0s hold candidates whose sampled full-body gap stays at least 5cm. Evaluator person ray hits are visibility diagnostics only. This cannot test a chosen action or physical safety.",
        "checks": checks,
        "prefix_rows": rows,
        "candidate_rows": candidate_rows,
        "v28_report_sha256": sha(V28 / "report.json"),
        "v28_audit_sha256": sha(V28 / "audit.json"),
        "v29_report_sha256": sha(v29_path),
        "v29_audit_sha256": sha(V29 / "audit.json"),
        "source_sha256": sha(Path(__file__)),
        "packet_trace_sha256": sources,
    }
    path = V29 / "preentry-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"],
                      "indistinguishable_prefix_pairs": sum(
                          v for k, v in checks.items() if k.startswith("same_actor_packets:")),
                      "prefix_pairs": len(rows),
                      "cases_without_supported_hold_candidate": sum(
                          not r["any_candidate_above_5cm"] for r in candidate_rows)},
                     indent=2))
    if not report["pass"]:
        raise SystemExit("Pre-entry choice audit failed")


if __name__ == "__main__":
    main()
