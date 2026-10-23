"""V80: low-only versus mid-only held-watch sensing on V79's paired turns.

All pre-show range packets remain V79's fused low+mid packets, so the chosen
meeting pose and approach are fixed. Only the second-inference baseline and
subsequent range packets switch plane; owner/flank RGB-D and prop scripts are
unchanged. This is a controlled development ablation, not a hardware choice.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import owner_northwest_new_turn_stress as v79


v78, v77 = v79.v78, v79.v77
ROOT = v79.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v80-northwest-range-heights"
PROFILES = {"low_05cm": (.05,), "mid_30cm": (.30,)}


def main():
    source = v77.runner.original()
    policy = json.loads(v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    vision = v77.runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v77.runner.owner.TEACH_CROP).convert("RGB")])[0]
    original_geometry = v77.runner.geometry_at
    original_packet = v77.runner.observed_packet
    results = []
    try:
        v77.runner.geometry_at = v77.v72.v64.v51.v46.low_prop_geometry(
            original_geometry)
        fused = v77.v72.v64.v51.v46.plane_packet(
            original_packet, (.05, .30))
        for profile_name, heights in PROFILES.items():
            held = v77.v72.v64.v51.v46.plane_packet(
                original_packet, heights)

            def profiled(model, data, actual, case, tick, *,
                         person_xy=None, hidden_return=False,
                         held=held):
                scan = held if tick >= 900 else fused
                return scan(model, data, actual, case, tick,
                            person_xy=person_xy,
                            hidden_return=hidden_return)

            v77.runner.observed_packet = profiled
            for path_name, endpoint in v79.ENDPOINTS.items():
                history = []

                def retain(packet, time_s, stage):
                    if stage == "post_alarm_range_watch":
                        history.append({"time_s": float(time_s),
                                        "packet": list(packet)})
                    return packet

                obstacle = lambda t, endpoint=endpoint: v77.v65.prop_at(
                    t, checkpoint_s, endpoint,
                    v79.FINISH_AFTER_CHECKPOINT_S)
                run = v77.runner.run_case(
                    v77.v72.v64.v54.CASE, source, policy, scenario,
                    vision, exemplar,
                    flank_probe_factory=lambda video:
                        v77.v72.v64.v51.v41.LocalizedFlankCamera(video),
                    require_camera_pose=True, route_obstacle_fn=obstacle,
                    pre_show_restage=v77.NorthwardWaitPose(history),
                    owner_fovy_deg=50., range_transform=retain,
                    near_body_controller=v77.v72.v64.v52.SurfaceTrackEscape(),
                    task_risk_monitor=v78.stationary_dynamic_risk,
                    output_root=OUT / profile_name / path_name)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                first = next(row for row in rows if row["stage"] ==
                             "pre_show_restage_step")
                final = next(row for row in rows if row["stage"] ==
                             "second_inference_result")
                results.append({"profile": profile_name,
                                "heights_m": list(heights),
                                "path": path_name,
                                "endpoint_xy_m": endpoint.tolist(),
                                "actor_choice": first["actor_decision"]["choice"],
                                "near_body_alarm_s": final["near_body_alarm_s"],
                                "run": run})
                print(profile_name, path_name, run["outcome"],
                      run["minimum_prop_gap_m"],
                      final["near_body_alarm_s"], flush=True)
    finally:
        v77.runner.geometry_at = original_geometry
        v77.runner.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-northwest-range-heights.v1",
              "source_sha256": v77.v72.v52.sha(__file__),
              "v79_source_sha256": v77.v72.v52.sha(v79.__file__),
              "v79_report_sha256": v77.v72.v52.sha(v79.OUT / "report.json"),
              "runner_source_sha256": v77.v72.v52.sha(v77.runner.__file__),
              "scene_sha256": v77.v72.v52.sha(
                  v77.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v77.v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "profiles": {k: list(v) for k, v in PROFILES.items()},
              "pre_show_plane_m": [.05, .30],
              "results": results,
              "scope": "Held-turn-only ideal range-height ablation on the two post-hoc V79 scripts; same fused approach, historical kinematic proxy, owner/flank RGB-D and task policy. No selected-body or physical sensor claim."}
    target = OUT / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
