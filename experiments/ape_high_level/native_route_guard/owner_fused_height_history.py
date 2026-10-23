"""V48: combine V46 low/mid range with V47 scan-history pose selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

import owner_history_meeting_pose as v47
import owner_low_prop_scan_planes as v46


v38 = v47.v38
v41 = v47.v41
ROOT = v47.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v48-fused-height-history"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = v38.original()
    policy = json.loads(v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v38.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    config = v47.v45.PROP_CONFIGS["early_closer"]
    obstacle_fn = lambda t: v41.v40.prop_at(t, checkpoint_s, config)
    original_geometry = v38.geometry_at
    original_packet = v38.observed_packet
    results = []
    try:
        v38.geometry_at = v46.low_prop_geometry(original_geometry)
        v38.observed_packet = v46.plane_packet(original_packet, (.05, .30))
        for case in v47.CASES:
            history = []

            def retain_actor_packet(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    history.append({"time_s": float(time_s),
                                    "stage": stage, "packet": list(packet)})
                return packet

            run = v38.run_case(
                case, source, policy, scenario, vision, exemplar,
                flank_probe_factory=lambda video: v41.LocalizedFlankCamera(video),
                flank_response=v38.FLANK_RESPONSE.get(case),
                require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                response_controller=(v41.JointHazardResponse()
                                     if case in v38.FLANK_RESPONSE else None),
                pre_show_restage=v47.HistoryMeetingPose(history),
                owner_fovy_deg=50., range_transform=retain_actor_packet,
                output_root=OUT / case)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            steps = [row for row in rows if row["stage"] ==
                     "pre_show_restage_step"]
            first = steps[0]["actor_decision"] if steps else None
            results.append({"case": case, "run": run,
                            "restage_steps": len(steps),
                            "choice": first["choice"] if first else None,
                            "novel_forward_returns": (
                                first["novel_forward_returns"] if first else [])})
            print(case, results[-1]["choice"], run["outcome"],
                  run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
                  flush=True)
    finally:
        v38.geometry_at = original_geometry
        v38.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-fused-height-history.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "v38_source_sha256": sha(v38.__file__),
              "v46_source_sha256": sha(v46.__file__),
              "v47_source_sha256": sha(v47.__file__),
              "v46_report_sha256": sha(v46.OUT / "report.json"),
              "v47_report_sha256": sha(v47.OUT / "report.json"),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "scan_heights_m": [.05, .30],
              "prop_geometry": {"radius_m": v46.PROP_RADIUS_M,
                                "half_height_m": v46.PROP_HALF_HEIGHT_M,
                                "center_z_m": v46.PROP_CENTER_Z_M},
              "prop_motion": config, "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
