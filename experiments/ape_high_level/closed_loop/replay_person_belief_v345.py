"""V345: replay a timed person/limb belief on V344 actor detector packets.

Thresholds are fixed from seed170's revealed person/no-person score gap and
then applied unchanged to seed172. Evaluator labels score the resulting state;
they never enter PersonBelief. This does not drive the live owner task yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from person_belief import PersonBelief
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v345"
V344 = ROOT / "artifacts/ape-closed-loop-cove50-v344/person-grounding-audit.json"


def detections(row: dict, camera: str, prompt: str) -> list[dict]:
    return row["results"][f"{camera}|{prompt}"]["actor_detections"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(V344.read_text())
    cases = sorted({row["case"] for row in source["rows"]})
    results = []
    for case in cases:
        rows = sorted((row for row in source["rows"] if row["case"] == case),
                      key=lambda row: row["time_s"])
        belief = PersonBelief()
        events = []
        for row in rows:
            result = belief.ingest(
                row["time_s"],
                detections(row, "upper_rgb", "person ."),
                detections(row, "lower_rgb", "hand . arm ."))
            events.append({"time_s": row["time_s"],
                           "actor_belief": result,
                           "evaluator_person_present": row["positive_person"],
                           "evaluator_upper_person_box": row["results"][
                               "upper_rgb|person ."]["evaluator_target_box"],
                           "evaluator_lower_arm_box": row["results"][
                               "lower_rgb|hand . arm ."]["evaluator_target_box"]})
        last = rows[-1]["time_s"]
        stale = belief.snapshot(last + .6)
        out_of_order = belief.ingest(rows[-2]["time_s"], [], [])
        results.append({"case": case, "seed": rows[0]["seed"],
                        "positive_person": rows[0]["positive_person"],
                        "events": events,
                        "stale_after_0p6s": stale,
                        "out_of_order_probe": out_of_order})
        print(case, [event["actor_belief"]["person_presence"]
                     for event in events],
              [event["actor_belief"]["lower_limb_state"]
               for event in events], flush=True)
    positives = [row for row in results if row["positive_person"]]
    negatives = [row for row in results if not row["positive_person"]]
    score = {
        "positive_case_count": len(positives),
        "negative_case_count": len(negatives),
        "confirmed_at_second_capture": sum(
            row["events"][1]["actor_belief"]["person_presence"] == "confirmed"
            for row in positives),
        "positive_third_capture_confirmed": sum(
            row["events"][2]["actor_belief"]["person_presence"] == "confirmed"
            for row in positives),
        "negative_ever_confirmed": sum(any(
            event["actor_belief"]["person_presence"] == "confirmed"
            for event in row["events"]) for row in negatives),
        "positive_last_limb_visible_now": sum(
            row["events"][-1]["actor_belief"]["lower_limb_state"] == "visible_now"
            for row in positives),
        "stale_person_unknown": sum(
            row["stale_after_0p6s"]["person_presence"] == "unknown"
            for row in results),
        "out_of_order_rejected": sum(
            not row["out_of_order_probe"]["accepted"] for row in results)}
    report = {"scope": "V345 actor-only temporal belief replay from V344 Grounding DINO upper-person and lower-hand/arm detections. Seed170 revealed person/no-person score gap sets fixed person threshold0.50; seed172 is a small same-geometry color/scene holdout. Two spatially consistent upper frames within350ms confirm presence; missing data ages to unknown after450ms, lower limb visibility is separate and last-seen is not physical clearance. Out-of-order frames are ignored. Evaluator person labels score the result only. No person identity, handoff safety, contact response or live owner-loop decision is proven.",
              "source_sha256": sha(SOURCE),
              "belief_source_sha256": sha(Path(__file__).with_name(
                  "person_belief.py")),
              "v344_audit_sha256": sha(V344),
              "rows": results,
              "score": score}
    (OUT / "person-belief-replay.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
