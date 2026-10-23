"""Check joined owner-route evidence without feeding labels to the actor."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance, evaluator_scene, sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-joined-owner-route-v410"
SOURCE = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_joined_owner_route_v410.py"


def audit_case(episode: dict) -> dict:
    nav = [row for row in episode["trace"] if row["stage"] == "navigate"]
    model, data, body_id, geom_id = evaluator_scene(None)
    poses = [np.array(row["actual_xy_m"], dtype=float) for row in nav]
    closest = {"distance_m": .1, "geom": None, "step": None}
    for i, pose in enumerate(poses):
        locations = [pose]
        if i + 1 < len(poses):
            locations += [pose + f * (poses[i + 1] - pose) for f in (.25, .5, .75)]
        for location in locations:
            distance, geom = clearance(model, data, body_id, geom_id, location)
            if distance < closest["distance_m"]:
                closest = {"distance_m": distance, "geom": geom, "step": i}
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    fresh = [row for row in episode["trace"] if row["stage"] == "fresh_owner_view"]
    claims = episode["outcome"] in ("owner_confirmed_visual_find", "owner_confirmed_metric_find")
    accepted = [row for row in fresh if row["actor"]["status"] == "visually_reacquired_unlocalized"]
    owner_yes = any(row["stage"] == "owner_confirms_taught_item" for row in episode["trace"])
    ask = any(row["stage"] == "ask_owner_to_show" for row in episode["trace"])
    claim_evidence = (not claims or
                      (ask and owner_yes and accepted and
                       accepted[-1]["frame_age_s"] <= .20 and
                       accepted[-1]["evaluator_iou"] >= .3))
    response_contract = (claims if episode["response"] in ("compliant", "delayed")
                         else not claims)
    return {"response": episode["response"], "depth": episode["depth"],
            "outcome": episode["outcome"], "route_steps": len(nav),
            "minimum_signed_clearance_m_capped_at_10cm": closest,
            "proxy_margin_pass": closest["distance_m"] >= .05,
            "fresh_claim_evidence_pass": bool(claim_evidence),
            "response_contract_pass": bool(response_contract),
            "video_sha_pass": sha256(video) == episode["video"]["sha256"],
            "video_frame_count_pass": frames == episode["video"]["frames"],
            "video_frames": frames,
            "claim_iou": accepted[-1]["evaluator_iou"] if accepted else None,
            "claim_frame_age_s": accepted[-1]["frame_age_s"] if accepted else None,
            "depth_point_error_m": episode["depth_point_error_m"]}


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    cases = [audit_case(episode) for episode in report["episodes"]]
    fingerprints = {episode["scene_fingerprint"] for episode in report["episodes"]}
    output = {"source_sha256": sha256(Path(__file__)),
              "joined_source_sha_pass": report["source_sha256"] == sha256(SOURCE),
              "scene_pair_fingerprint_pass": len(fingerprints) == 1,
              "cases": cases,
              "all_pass": all(case["proxy_margin_pass"] and
                              case["fresh_claim_evidence_pass"] and
                              case["response_contract_pass"] and
                              case["video_sha_pass"] and
                              case["video_frame_count_pass"] for case in cases)}
    (OUT / "audit-v411.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["response"], "RGB-D" if case["depth"] else "RGB",
              case["outcome"], "clearance >=", case["minimum_signed_clearance_m_capped_at_10cm"]["distance_m"],
              "claim IoU", case["claim_iou"], flush=True)
    print("all_pass", output["all_pass"], flush=True)


if __name__ == "__main__":
    main()
