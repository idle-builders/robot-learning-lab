"""Join the fourteen frozen-stop identity continuations with the six paired rooms."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
COHORT = ART / "common_policy_20_group_full_loop_report.json"
EXTENSION = ART / "common_policy_20_group_multicue_extension_report.json"
ALIGNMENT = ART / "common_policy_20_group_multicue_pairing_alignment.json"
CORRECTION = ART / "common_policy_20_group_multicue_exact_stop_correction_report.json"
OUT = ART / "common_policy_20_group_multicue_full_loop_report.json"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def stop_state(summary_path: str) -> dict:
    summary = json.loads((ROOT / summary_path).read_text())
    trace = json.loads((ROOT / summary["teaching"]["owner_approach_trace"]).read_text())
    last = trace["trace"][-1]
    return {key: last[key] for key in ("evaluator_actual_xz_m",
                                      "evaluator_actual_heading_deg", "decision_time_s")}


def main() -> None:
    cohort = json.loads(COHORT.read_text())
    extension = json.loads(EXTENSION.read_text())
    alignment = json.loads(ALIGNMENT.read_text())
    correction = json.loads(CORRECTION.read_text())
    if extension["completed_group_count"] != 14 or not extension["summary"]["all_find_media_aligned"]:
        raise ValueError("The baseline-only extension is incomplete")
    if alignment["exact_pairs"] != 11 or correction["completed_group_count"] != 3 or \
            not correction["all_media_aligned"]:
        raise ValueError("Three mismatched owner stops were not repaired")
    extended = {r["capture_group"]: r for r in extension["rooms"]}
    aligned = {r["capture_group"]: r["exact_stop_pair"] for r in alignment["rows"]}
    corrected = {r["capture_group"]: r for r in correction["rows"]}
    if len(extended) != 14:
        raise ValueError("Extension repeats a physical capture group")
    rows = []
    for room in cohort["rooms"]:
        if room["dino"]["status"] != "teach_find_completed_far_viewpoint":
            continue
        group = room["capture_group"]
        if room["cohort"] == "baseline":
            x = extended.pop(group)
            if aligned.pop(group):
                multi = {"present_correct": x["multi_cue_present_correct"],
                         "absent_safe": x["multi_cue_absent_safe"],
                         "wrong_confident": x["multi_cue_wrong_confident"],
                         "media_aligned": all(p["media_aligned"] for p in x["phases"].values()),
                         "summary": x["summary"],
                         "present_video": x["phases"]["find_present"]["video"],
                         "absent_video": x["phases"]["find_absent"]["video"]}
            else:
                fixed = corrected.pop(group)
                multi = {"present_correct": fixed["present_correct"],
                         "absent_safe": fixed["absent_safe"],
                         "wrong_confident": fixed["wrong_confident"],
                         "media_aligned": all(p["media_aligned"] for p in fixed["phases"].values()),
                         "summary": fixed["summary"],
                         "present_video": fixed["phases"]["find_present"]["video"],
                         "absent_video": fixed["phases"]["find_absent"]["video"],
                         "corrected_exact_stop": True}
            if (x["baseline_present_correct"] != room["dino"]["present_correct"] or
                    x["baseline_absent_safe"] != room["dino"]["absent_safe"]):
                raise ValueError(f"Baseline mismatch: {group}")
        else:
            x = room["multi_cue"]
            if x["status"] != "teach_find_completed_far_viewpoint":
                raise ValueError(f"Unpaired new group: {group}")
            multi = {"present_correct": x["present_correct"],
                     "absent_safe": x["absent_safe"],
                     "wrong_confident": any(x[label]["wrong_confident_candidate"]
                                            for label in ("find_present", "find_absent")),
                     "media_aligned": all(x[label]["media_aligned"]
                                          for label in ("find_present", "find_absent")),
                     "summary": x["summary"],
                     "present_video": x["find_present"]["video"],
                     "absent_video": x["find_absent"]["video"]}
        exact_stop = stop_state(room["dino"]["summary"]) == stop_state(multi["summary"])
        rows.append({"capture_group": group, "cohort": room["cohort"],
                     "scene_id": room["scene_id"],
                     "exact_owner_stop_pair": exact_stop,
                     "dino_present_correct": room["dino"]["present_correct"],
                     "dino_absent_safe": room["dino"]["absent_safe"],
                     "multi_cue": multi})
    if extended or aligned or corrected or len(rows) != 20 or \
            len({r["capture_group"] for r in rows}) != 20:
        raise ValueError("The 20-group paired cohort is not exact")
    if not all(r["exact_owner_stop_pair"] for r in rows):
        raise ValueError("A claimed paired identity comparison changed the owner stop")
    report = {"schema": "ape.common-policy-20-group-multicue-full-loop-report.v1",
              "baseline_audit": relative(COHORT),
              "extension": relative(EXTENSION),
              "pairing_alignment": relative(ALIGNMENT),
              "exact_stop_correction": relative(CORRECTION),
              "distinct_completed_capture_groups": len(rows),
              "summary": {"dino_present_correct": sum(r["dino_present_correct"] for r in rows),
                          "exact_owner_stop_pairs": sum(r["exact_owner_stop_pair"] for r in rows),
                          "multi_cue_present_correct": sum(r["multi_cue"]["present_correct"] for r in rows),
                          "dino_absent_safe": sum(r["dino_absent_safe"] for r in rows),
                          "multi_cue_absent_safe": sum(r["multi_cue"]["absent_safe"] for r in rows),
                          "multi_cue_wrong_confident_groups": sum(r["multi_cue"]["wrong_confident"] for r in rows),
                          "all_multi_cue_media_aligned": all(r["multi_cue"]["media_aligned"] for r in rows)},
              "limits": "These twenty development-exposed groups share one Parmesan/CreamCheese task, scripted teaching and confirmation, ideal yaw/range and a kinematic proxy in photo-textured but not certified photoreal rooms. The fourteen extensions reuse exact saved baseline owner stops after three metadata-selected corrections; their item continuation is paired but is not a new dynamic approach rollout.",
              "rooms": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
