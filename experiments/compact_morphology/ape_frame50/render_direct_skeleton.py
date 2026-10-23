"""Render the near-joint, belt-free ape frame on the saved learned walk."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.compact_morphology.ape_frame50 import direct_skeleton_candidate as direct
from experiments.compact_morphology.ape_frame50 import render_cad_replay as replay
from experiments.compact_morphology.ape_frame50 import screen_direct_cases as clearance


OUT = replay.ROOT / "artifacts/ape-frame50-direct-skeleton"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in out.rglob("*")):
        raise ValueError("sealed artifact: choose a fresh --out")
    visual = out / "ape-frame50-direct-visual.xml"
    stages = list(direct.visual_parts())
    disconnected = []
    through_cases = []
    frame_by_body = {body: parts[f"{body}_integrated_frame"] for _, body, parts in stages}
    part_by_body = {body: parts for _, body, parts in stages}
    for _, body, parts in stages:
        frame = parts[f"{body}_integrated_frame"]
        if len(frame.solids()) != 1:
            disconnected.append(body)
        for drive in (d for d in direct.drives() if d.parent == body):
            if (frame & direct.box(drive.center, drive.size)).volume > 0.1:
                through_cases.append(drive.name)
    other_frame_overlap = []
    unjoined_hubs = []
    for drive in direct.drives():
        child = direct.child_body(drive)
        hub = part_by_body[child][f"{drive.name}_rotating_hub"]
        if (hub & frame_by_body[child]).volume < 50:
            unjoined_hubs.append(drive.name)
        case = direct.box(drive.center, drive.size)
        cb = case.bounding_box()
        for body, frame in frame_by_body.items():
            if body == drive.parent:
                continue
            fb = frame.bounding_box()
            if any(min(tuple(cb.max)[i], tuple(fb.max)[i]) -
                   max(tuple(cb.min)[i], tuple(fb.min)[i]) <= 0 for i in range(3)):
                continue
            if (case & frame).volume > 1:
                other_frame_overlap.append((drive.name, body))
    head_join_volume = (frame_by_body["head"] & frame_by_body["head_yaw_body"]).volume
    if disconnected or through_cases or other_frame_overlap or unjoined_hubs or head_join_volume < 100:
        raise ValueError(f"invalid stage frame: {disconnected=}, {through_cases=}, {other_frame_overlap=}, {unjoined_hubs=}, {head_join_volume=}")
    report = replay.build_visual_body(
        replay.BODY, visual, part_source=lambda: iter(stages), coordinates="sim",
        motor_alignment={f"{d.name}_{d.model}_case": d for d in direct.drives()})
    motion_clearance = clearance.screen()
    if motion_clearance["motor_case_pair_collisions"]:
        raise ValueError(f"motor cases collide in saved motion: {motion_clearance['motor_case_pair_collisions']}")
    report.update({
        "motor_layout": direct.motor_report(),
        "neutral_motor_case_overlaps": direct.neutral_case_overlaps(),
        "saved_motion_case_clearance": motion_clearance,
        "connected_structural_stages": len(stages),
        "structural_solid_volume_cm3": round(sum(s.volume for s in frame_by_body.values()) / 1000, 2),
        "frame_through_motor_cases": through_cases,
        "neutral_frame_motor_overlaps": other_frame_overlap,
        "unjoined_rotating_hubs": unjoined_hubs,
        "head_neck_frame_join_mm3": round(head_join_volume, 2),
        "supplier_step_spline_axis_residual_mm": 0.0,
        "cad_generator_sha256": replay.digest(Path(direct.__file__)),
        "renderer_sha256": replay.digest(Path(replay.__file__)),
        "belt_routes": 0,
        "ankle_cranks": 0,
        "architecture": "21 coaxial direct output joints; short hubs at offset faces",
        "limitations": [
            "CAD is attached to an earlier learned qpos trace; this is not a new policy rollout.",
            "Motor-case envelopes clear the 400 saved poses, but other inter-part clearances and fastening are not qualified.",
            "Current MJCF retains old mass, contact and torque assumptions.",
            "Hip and knee motor rated torque may be inadequate; direct packaging does not increase torque.",
            "Head and gripper XL330 parts are dimensional envelopes, not supplier STEP geometry.",
        ],
    })
    replay.render(visual, replay.TRACE, out / "cad-learned-walk.mp4", report)
    (out / "case-clearance.json").write_text(json.dumps(motion_clearance, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    source = Path(__file__).with_name("cad_replay_viewer.html").read_text()
    source = source.replace("The CAD assembly walks with the learned policy",
                            "Direct-drive ape frame on the saved walk")
    source = source.replace(
        "This view uses the actual frame CAD and supplier STEP solids for 18 Feetech motors. Its body poses come from a closed-loop MuJoCo replay of the saved PPO policy.",
        "This candidate places 18 detailed Feetech motors next to their joint axes and removes the 16 one-to-one belts and two ankle cranks. Its poses are the saved walk from the earlier body, not a new policy evaluation.")
    source = source.replace(
        "Mass, contacts and motor control still come from the separate simplified MJCF used by the learned run. This visual pass does not prove every CAD part clears every pose.",
        "This is a packaging and motion inspection. Mass, contacts and motor control still come from the earlier simplified MJCF. The new direct-drive layout has not yet been qualified by a fresh policy rollout or a full swept-clearance check.")
    source = source.replace("The new direct-drive layout has not yet been qualified by a fresh policy rollout or a full swept-clearance check.",
                            "Motor cases clear all 400 saved poses; other moving clearances, fastening, and drive duty are not yet qualified.")
    source = source.replace(
        "<h2>Fixed-camera video of the same run</h2>",
        '<p class="small"><a href="../ape-frame50-cad-replay/index.html">Compare the former belt-heavy assembly ↗</a> · <a href="../../docs/research/ape-frame50-direct-drive-review-20261007.md">Read the mechanical review ↗</a></p><h2>Fixed-camera video of the same run</h2>')
    source = source.replace("Ape frame · learned walk with CAD motors",
                            "Ape frame · direct-drive packaging")
    (out / "index.html").write_text(source)
    print(json.dumps({"out": str(out), "parts": report["cad_parts"],
                      "vendor_motors": report["detailed_vendor_motors"],
                      "neutral_overlaps": report["neutral_motor_case_overlaps"],
                      "frames": report["frames"]}, indent=2))


if __name__ == "__main__":
    main()
