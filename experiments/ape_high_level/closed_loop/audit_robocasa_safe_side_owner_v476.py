"""Independent full-horizon geometry and owner-evidence audit for V475."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-safe-side-owner-v475"


def audit_owner(episode):
    rows = episode["trace"]
    fresh = [row for row in rows if row["stage"] == "fresh_owner_view"]
    accepted = [row for row in fresh
                if row["actor"]["status"] == "visually_reacquired_unlocalized"]
    claims = episode["outcome"] in ("owner_confirmed_metric_find",
                                     "owner_confirmed_visual_find")
    owner_yes = any(row["stage"] == "owner_confirms_taught_item" for row in rows)
    ask = any(row["stage"] == "ask_owner_to_show" for row in rows)
    metric_evidence = (not episode["depth"] or
                       (accepted and accepted[-1]["depth_point_xyz_m"] is not None and
                        episode["depth_point_error_m"] is not None and
                        episode["depth_point_error_m"] <= .10))
    claim_evidence = (not claims or
                      (ask and owner_yes and accepted and
                       accepted[-1]["frame_age_s"] <= .20 and
                       accepted[-1]["evaluator_iou"] >= .3 and
                       metric_evidence))
    response_ok = (claims if episode["response"] in ("compliant", "delayed")
                   else not claims)
    if episode["route_state"] == "request_owner_show_here_safe_side":
        response_ok = response_ok and any(
            row["stage"] == "request_owner_show_here_safe_side" for row in rows)
    hold = [row for row in rows if row["stage"] == "safe_side_hold"]
    hold_ok = (not hold if episode["route_state"] == "goal_viewpoint_reached" else
               bool(hold) and episode["hold_until_s"] >= episode["scenario"]["end_s"]
               and all(row["actual_xy_m"] == episode["final_actual_xy_m"]
                       and row["range_returns"] > 0 for row in hold))
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    return {"response": episode["response"], "depth": episode["depth"],
            "outcome": episode["outcome"], "route_state": episode["route_state"],
            "fresh_accepted_count": len(accepted),
            "final_accepted_iou": accepted[-1]["evaluator_iou"] if accepted else None,
            "final_accepted_age_s": accepted[-1]["frame_age_s"] if accepted else None,
            "depth_point_error_m": episode["depth_point_error_m"],
            "claim_evidence_pass": bool(claim_evidence),
            "response_contract_pass": bool(response_ok),
            "full_horizon_stationary_hold_pass": bool(hold_ok),
            "video_sha_pass": sha256(video) == episode["video"]["sha256"],
            "video_frame_count_pass": frames == episode["video"]["frames"]}


def main():
    source = OUT / "report.json"
    report = json.loads(source.read_text())
    route_audit.evaluator_scene = layout4_evaluator_scene
    cases = []
    for episode in report["episodes"]:
        route = route_audit.audit_case({
            "scenario": episode["scenario"],
            "package": "safe_side_owner_route",
            "episode": episode["route_episode"]})
        owner = audit_owner(episode)
        safe_side = episode["safe_side_history"]
        decision_ok = (not safe_side if episode["route_state"] == "goal_viewpoint_reached"
                       else bool(safe_side) and safe_side[-1]["should_stage"]
                       and episode["route_episode"]["trace"][-1]["requested_xy_m"] is None
                       and episode["route_episode"]["end_time_s"] < episode["scenario"]["begin_s"])
        cases.append({"scenario_id": episode["scenario"]["id"],
                      "response": episode["response"], "depth": episode["depth"],
                      "route": route, "owner": owner,
                      "safe_side_decision_pass": bool(decision_ok)})
        print(episode["scenario"]["id"], episode["response"],
              "rgbd" if episode["depth"] else "rgb", episode["outcome"],
              "room_gap", round(route["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "prop_gap", round(route["nearest_ape_prop"]["signed_gap_m"], 3),
              "owner", owner["claim_evidence_pass"] and owner["response_contract_pass"],
              flush=True)
    pair = [episode for episode in report["episodes"]
            if episode["scenario"]["id"] == "late_crossing"
            and episode["response"] == "compliant"]
    output = {
        "source_sha256": sha256(Path(__file__)),
        "trial_report_sha256": sha256(source),
        "trial_source_sha_pass": report["source_sha256"] == sha256(
            Path(__file__).with_name("run_robocasa_safe_side_owner_v475.py")),
        "config_source_sha_pass": report["config_sha256"] == sha256(
            Path(__file__).with_name("robocasa_layout4_safe_side_v473.json")),
        "prior_map_sha_pass": report["prior_map_sha256"] == sha256(
            ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
        "safe_side_policy_source_sha_pass": report["safe_side_policy_source_sha256"] == sha256(
            Path(__file__).with_name("run_robocasa_layout4_safe_side_v473.py")),
        "late_stationary_scene_pair_pass": len({episode["scene_fingerprint"]
                                                 for episode in report["episodes"]
                                                 if episode["scenario"]["id"] != "absent"}) == 1,
        "rgb_rgbd_route_pair_pass": len(pair) == 2 and
        pair[0]["route_episode"]["trace"] == pair[1]["route_episode"]["trace"],
        "all_geometry_pass": all(case["route"]["ape_5cm_margin_pass"] and
                                 case["route"]["prop_fixture_nonoverlap_pass"] and
                                 case["route"]["nearest_ape_prop"]["signed_gap_m"] >= .05
                                 for case in cases),
        "all_safe_side_decisions_pass": all(case["safe_side_decision_pass"] for case in cases),
        "all_owner_contracts_pass": all(case["owner"]["claim_evidence_pass"] and
                                        case["owner"]["response_contract_pass"] for case in cases),
        "all_full_horizon_holds_pass": all(case["owner"]["full_horizon_stationary_hold_pass"]
                                       for case in cases),
        "all_videos_pass": all(case["route"]["video_sha_pass"] and
                               case["route"]["video_frames_pass"] and
                               case["owner"]["video_sha_pass"] and
                               case["owner"]["video_frame_count_pass"]
                               for case in cases),
        "cases": cases,
    }
    (OUT / "audit-v476.json").write_text(json.dumps(output, indent=2) + "\n")
    print({key: value for key, value in output.items() if key.endswith("_pass")}, flush=True)


if __name__ == "__main__":
    main()
