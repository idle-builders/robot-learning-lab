"""V376: replay missing camera packets against the moving-show answer gate.

Dropping a packet means no callback reaches the V373 track update. An
occluded packet is delivered with no accepted item; both are distinct from
an image in which the detector merely selects the wrong visible item.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from run import ROOT, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v374/independent-hand-show.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v376"
MAX_STEP_M = .12
MAX_STEP_S = .12
MAX_ANSWER_AGE_S = .12
SCENARIOS = {
    "intact": ("drop", ()),
    "middle_one_missing": ("drop", (6,)),
    "middle_two_missing": ("drop", (6, 7)),
    "last_one_missing": ("drop", (12,)),
    "last_two_missing": ("drop", (11, 12)),
    "last_three_missing": ("drop", (10, 11, 12)),
    "last_two_occluded": ("occluded", (11, 12)),
}


def replay(samples: list[dict], scenario: str, mode: str,
           indices: tuple[int, ...]) -> dict:
    track_xyz = None
    track_time = None
    track_run = 0
    ever_confirmed = False
    last_box = None
    last_valid_time = None
    for index, sample in enumerate(samples):
        if index in indices and mode == "drop":
            continue
        now = float(sample["capture_time_s"])
        if index in indices and mode == "occluded":
            xyz, box = None, None
        else:
            box = sample["actor_choice"]["accepted_box"]
            xyz = sample["actor_robust_long_depth"]["world_xyz_m"]
        linked = (xyz is not None and track_xyz is not None and
                  track_time is not None and now - track_time <= MAX_STEP_S
                  and math.dist(xyz, track_xyz) <= MAX_STEP_M)
        if xyz is None:
            track_run, track_xyz = 0, None
        else:
            track_run = track_run + 1 if linked else 1
            track_xyz, track_time = xyz, now
            last_valid_time = now
        last_box = box
        ever_confirmed = ever_confirmed or track_run >= 2
    decision_time = float(samples[-1]["capture_time_s"])
    age = None if last_valid_time is None else decision_time - last_valid_time
    old_gate_accepts = bool(ever_confirmed and track_run >= 2 and
                            last_box is not None)
    fresh_gate_accepts = bool(old_gate_accepts and age is not None and
                              age <= MAX_ANSWER_AGE_S)
    return {
        "scenario": scenario, "mode": mode, "affected_indices": indices,
        "actor_ever_confirmed": ever_confirmed,
        "actor_final_track_run": track_run,
        "actor_last_valid_age_s": None if age is None else round(age, 3),
        "original_v373_answer_gate_accepts": old_gate_accepts,
        "age_bounded_answer_gate_accepts": fresh_gate_accepts,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(INPUT.read_text())
    rows = []
    for source in inputs["rows"]:
        tests = [replay(source["samples"], name, *configuration)
                 for name, configuration in SCENARIOS.items()]
        rows.append({"case": source["case"], "input_video_sha256": source[
            "video_sha256"], "scenarios": tests})
        print(source["case"], [(x["scenario"],
                                  x["original_v373_answer_gate_accepts"],
                                  x["age_bounded_answer_gate_accepts"])
                                 for x in tests], flush=True)
    report = {
        "scope": "Replay V374 selected-body hand-motion actor choices and robust depth points, omitting camera callbacks or delivering no-object packets at specified capture indices. Compares the unchanged V373 final answer predicate with a <=120ms age bound at the last scheduled sample. This is a deterministic trace audit, not a new vision or physics run. No owner audio or identity proof is modeled.",
        "source_sha256": sha(SOURCE), "input_report_sha256": sha(INPUT),
        "max_track_step_m": MAX_STEP_M,
        "max_track_step_s": MAX_STEP_S,
        "max_answer_age_s": MAX_ANSWER_AGE_S, "rows": rows}
    (OUT / "moving-show-dropout-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
