"""Audit dynamic route, owner evidence, hold order and joined videos for V471."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from audit_robocasa_layout4_wait_and_remap_v462 import (
    decision_contract, evaluator_prop_at,
)
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-dynamic-owner-v471"


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
    if episode["route_state"] != "goal_viewpoint_reached":
        response_ok = (not claims and not ask and
                       episode["outcome"] == "ask_owner_clear_corridor" and
                       any(row["stage"] == "request_owner_clear_corridor" for row in rows))
    elif episode["response"] in ("compliant", "delayed"):
        response_ok = claims
    else:
        response_ok = not claims
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    return {"response": episode["response"],
            "depth": episode["depth"],
            "outcome": episode["outcome"],
            "route_state": episode["route_state"],
            "fresh_accepted_count": len(accepted),
            "final_accepted_iou": accepted[-1]["evaluator_iou"] if accepted else None,
            "final_accepted_age_s": accepted[-1]["frame_age_s"] if accepted else None,
            "depth_point_error_m": episode["depth_point_error_m"],
            "claim_evidence_pass": bool(claim_evidence),
            "response_contract_pass": bool(response_ok),
            "video_sha_pass": sha256(video) == episode["video"]["sha256"],
            "video_frame_count_pass": frames == episode["video"]["frames"]}


def main():
    source = OUT / "report.json"
    report = json.loads(source.read_text())
    route_audit.evaluator_scene = layout4_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    cases = []
    for episode in report["episodes"]:
        route = route_audit.audit_case({
            "scenario": episode["scenario"],
            "package": "directional_owner_route",
            "episode": episode["route_episode"]})
        hold = decision_contract({"episode": episode["route_episode"],
                                  "decision_history": episode["decision_history"]})
        owner = audit_owner(episode)
        cases.append({"scenario_id": episode["scenario"]["id"],
                      "response": episode["response"],
                      "route": route, "hold_contract": hold, "owner": owner})
        print(episode["scenario"]["id"], episode["response"],
              episode["route_state"], episode["outcome"],
              "room_gap", round(route["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "prop_gap", round(route["nearest_ape_prop"]["signed_gap_m"], 3),
              "owner_pass", owner["claim_evidence_pass"] and owner["response_contract_pass"],
              flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(source),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_dynamic_owner_v471.py")),
              "config_source_sha_pass": report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_returning_prop_v460.json")),
              "prior_map_sha_pass": report["prior_map_sha256"] == sha256(
                  ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
              "same_initial_scene_pass": len({e["scene_fingerprint"]
                                               for e in report["episodes"]}) == 1,
              "rgb_rgbd_route_pair_pass": (
                  lambda pair: len(pair) == 2 and
                  pair[0]["route_episode"]["trace"] == pair[1]["route_episode"]["trace"]
              )([e for e in report["episodes"]
                 if e["scenario"]["id"] == "returning_crossing"
                 and e["response"] == "compliant"]),
              "all_route_videos_pass": all(c["route"]["video_sha_pass"] and
                                           c["route"]["video_frames_pass"] for c in cases),
              "all_joined_videos_pass": all(c["owner"]["video_sha_pass"] and
                                            c["owner"]["video_frame_count_pass"] for c in cases),
              "all_geometry_pass": all(c["route"]["ape_5cm_margin_pass"] and
                                       c["route"]["prop_fixture_nonoverlap_pass"] for c in cases),
              "all_hold_contracts_pass": all(c["hold_contract"]["pass"] for c in cases),
              "all_owner_contracts_pass": all(c["owner"]["claim_evidence_pass"] and
                                              c["owner"]["response_contract_pass"] for c in cases),
              "cases": cases}
    (OUT / "audit-v472.json").write_text(json.dumps(output, indent=2) + "\n")
    print("all_route_videos", output["all_route_videos_pass"],
          "all_joined_videos", output["all_joined_videos_pass"],
          "geometry", output["all_geometry_pass"],
          "holds", output["all_hold_contracts_pass"],
          "owner", output["all_owner_contracts_pass"], flush=True)


if __name__ == "__main__":
    main()
