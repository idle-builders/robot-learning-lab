"""V352: compare camera-pose-aware person bearing to V345 image-box belief."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from bearing_person_belief import BEARING_GATE_RAD, BearingPersonBelief
from direct_structural_rooms_v270 import selected_visual
from person_belief import PersonBelief
from robot_geometry import RobotGeometry
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v352"
INPUTS = (
    ROOT / "artifacts/ape-closed-loop-cove50-v351/active-owner-look-audit.json",
    ROOT / "artifacts/ape-closed-loop-cove50-v346/person-belief-challenge.json",
    ROOT / "artifacts/ape-closed-loop-cove50-v347/interaction-motion-challenge.json",
)


def source_rows() -> list[tuple[str, dict]]:
    v351, v346, v347 = [json.loads(p.read_text()) for p in INPUTS]
    return ([("active_look", row) for row in v351["rows"]] +
            [("person_challenge", row) for row in v346["rows"]] +
            [("moving_negative", row) for row in v347["moving_negative_rows"]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(selected_visual()))
    data = mujoco.MjData(model)
    camera = model.camera("upper_rgb").id
    fovy = float(model.cam_fovy[camera])
    rows = []
    for cohort, source in source_rows():
        old, new = PersonBelief(), BearingPersonBelief()
        captures = []
        for sample in source["actor_samples"]:
            with np.load(ROOT / sample["actor_packet_path"]) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            if len(qpos) != model.nq:
                raise AssertionError("Selected robot qpos changed")
            data.qpos[:] = RobotGeometry.pose_qpos(qpos, estimate)
            mujoco.mj_forward(model, data)
            upper = sample["actor_upper_person_detections"]
            lower = sample.get("actor_lower_arm_detections",
                               sample.get("actor_lower_limb_detections"))
            if lower is None:
                raise AssertionError("Missing actor lower proposals")
            old_state = old.ingest(sample["time_s"], upper, lower)
            new_state = new.ingest(
                sample["time_s"], upper, lower,
                upper_camera_rotation=data.cam_xmat[camera].reshape(3, 3).copy(),
                upper_camera_fovy_deg=fovy)
            captures.append({"time_s": sample["time_s"],
                             "image_box_belief": old_state,
                             "world_bearing_belief": new_state})
        rows.append({"case": source["case"], "cohort": cohort,
                     "positive_person": source.get("positive_person", True),
                     "captures": captures,
                     "at_second_image_box": captures[1]["image_box_belief"][
                         "person_presence"],
                     "at_second_world_bearing": captures[1][
                         "world_bearing_belief"]["person_presence"],
                     "at_end_image_box": captures[-1]["image_box_belief"][
                         "person_presence"],
                     "at_end_world_bearing": captures[-1][
                         "world_bearing_belief"]["person_presence"]})
    def count(cohort: str, field: str, state: str) -> int:
        return sum(r["cohort"] == cohort and r[field] == state for r in rows)
    report = {
        "scope": "Actor-only replay of frozen Grounding DINO proposals, selected robot kinematics, estimated base pose and camera calibration. A world-bearing association (0.12rad exploratory gate) replaces image-box IoU; no simulator person ID, hand geometry or evaluator mask enters belief. Active-head and static-person/mannequin controls reveal whether camera motion breaks temporal presence without claiming owner identity or safe physical interaction.",
        "source_sha256": sha(SOURCE),
        "belief_source_sha256": sha(SOURCE.with_name(
            "bearing_person_belief.py")),
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in INPUTS},
        "selected_visual_sha256": sha(selected_visual()),
        "world_bearing_gate_rad": BEARING_GATE_RAD,
        "score": {
            "active_look_cases": count("active_look", "at_end_world_bearing",
                                       "confirmed"),
            "active_look_confirmed_second_image_box": count(
                "active_look", "at_second_image_box", "confirmed"),
            "active_look_confirmed_second_world_bearing": count(
                "active_look", "at_second_world_bearing", "confirmed"),
            "person_challenge_positive_confirmed_end": sum(
                r["cohort"] == "person_challenge" and r["positive_person"] and
                r["at_end_world_bearing"] == "confirmed" for r in rows),
            "person_challenge_negative_false_confirmed_end": sum(
                r["cohort"] == "person_challenge" and not r["positive_person"]
                and r["at_end_world_bearing"] == "confirmed" for r in rows),
            "moving_negative_false_confirmed_end": count(
                "moving_negative", "at_end_world_bearing", "confirmed"),
        },
        "rows": rows,
    }
    (OUT / "bearing-person-replay.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(report["score"], flush=True)


if __name__ == "__main__":
    main()
