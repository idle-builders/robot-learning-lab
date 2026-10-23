"""Audit longer real-splat owner episodes against frozen pose and view gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SCREEN = ROOT / "assets/real_indoor_scenes/habitat_gs/owner_placement_screen/v3/screen.json"
ARTIFACTS = ROOT / "artifacts/ape-real-room-next-captures-20261014"
ARMS = {
    "rgbd_yaw": "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_delay850ms_rangeprojected_farconverse_habgs4v1_v1.json",
    "native_shaped": "owner_batch_native_shaped_arbitrated_delay850ms_rangeprojected_farconverse_varied150-forward-seed144_habgs4v1_v1.json",
}
CANDIDATE_ARMS = {
    "rgbd_yaw": "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_delay850ms_rangeprojected_farconverse_recedingowner_habgs4recedingv1_v1.json",
    "native_shaped": "owner_batch_native_shaped_arbitrated_delay850ms_rangeprojected_farconverse_recedingowner_varied150-forward-seed144_habgs4recedingv1_v1.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    precommit_name = ("habitat_gs_receding_owner_precommit.json" if args.candidate
                      else "habitat_gs_owner_loop_precommit.json")
    arms = CANDIDATE_ARMS if args.candidate else ARMS
    json.loads((ARTIFACTS / precommit_name).read_text())
    screen = json.loads(SCREEN.read_text())
    assert screen["scene_count"] == 10 and screen["supported_count"] == 4
    rows = {}
    for arm, filename in arms.items():
        batch = json.loads((SCREEN.parent / filename).read_text())
        assert batch["scene_count"] == 10 and batch["completed_count"] == 4
        results = []
        for row in batch["scenes"]:
            if row["task_status"] != "completed":
                continue
            trace = json.loads((ROOT / row["trace"]).read_text())
            folder = (ROOT / row["trace"]).parent
            black_fractions = []
            valid_depth_fractions = []
            for frame in trace["trace"]:
                number = int(frame["frame"])
                rgb = np.asarray(Image.open(folder / f"rgb_{number:03d}.png").convert("RGB"))
                depth = np.asarray(Image.open(folder / f"depth_{number:03d}.png"))
                black_fractions.append(float(np.mean(np.max(rgb, axis=-1) < 8)))
                valid_depth_fractions.append(float(np.mean(depth > 0)))
            errors = [float(f["evaluator_pose_error_at_capture_m"])
                      for f in trace["trace"]]
            observed_view = bool(row.get("observed_standoff_band_met") or
                                 row.get("observed_far_conversation_band_met"))
            healthy_view = bool(observed_view and trace["trace"][-1]["pose_healthy"] and
                                row.get("outcome") in ("hold_viewpoint", "hold_viewpoint_far"))
            pose_qualified = healthy_view and max(errors) <= .08
            long_route = float(row["accepted_travel_m"]) >= .9
            appearance_coverage = (max(black_fractions) <= .05 and
                                   min(valid_depth_fractions) >= .70)
            result = {"scene_id": row["scene_id"], "name": row["name"],
                      "arm": arm, "outcome": row["outcome"],
                      "stop_reason": row["stop_reason"],
                      "accepted_travel_m": row["accepted_travel_m"],
                      "frames": row["frames"],
                      "healthy_viewpoint": healthy_view,
                      "pose_qualified_viewpoint": pose_qualified,
                      "false_healthy_viewpoint": healthy_view and not pose_qualified,
                      "long_route_qualified_viewpoint": bool(pose_qualified and long_route),
                      "maximum_capture_aligned_pose_error_m": max(errors),
                      "maximum_near_black_rgb_fraction": max(black_fractions),
                      "minimum_valid_mesh_depth_fraction": min(valid_depth_fractions),
                      "appearance_coverage_gate": appearance_coverage,
                      "first_rgb": str((folder / "rgb_000.png").relative_to(ROOT)),
                      "middle_rgb": str((folder / f"rgb_{len(errors)//2:03d}.png").relative_to(ROOT)),
                      "final_rgb": str((folder / f"rgb_{len(errors)-1:03d}.png").relative_to(ROOT)),
                      "video": row["video"], "trace": row["trace"]}
            results.append(result)
        rows[arm] = results
    report = {"schema": "ape.habitat-gs-owner-loop-audit.v1",
              "precommit": str((ARTIFACTS / precommit_name).relative_to(ROOT)),
              "source_screen": str(SCREEN.relative_to(ROOT)),
              "source_batches": {arm: str((SCREEN.parent / filename).relative_to(ROOT))
                                 for arm, filename in arms.items()},
              "summary": {arm: {
                  "completed": len(results),
                  "healthy_viewpoints": sum(r["healthy_viewpoint"] for r in results),
                  "pose_qualified_viewpoints": sum(r["pose_qualified_viewpoint"] for r in results),
                  "long_route_qualified_viewpoints": sum(
                      r["long_route_qualified_viewpoint"] for r in results),
                  "appearance_coverage_pass": sum(r["appearance_coverage_gate"] for r in results),
                  "false_healthy_viewpoints": sum(r["false_healthy_viewpoint"] for r in results),
              } for arm, results in rows.items()},
              "results": rows,
              "limits": "Four separate real-recorded capture groups, but reused released Gaussian model/mesh and prior generic route starts; scripted rigid person, ideal range/yaw, historical motion errors, 17 cm kinematic proxy, no selected 50 cm body. Coverage numbers cannot certify photoreal moving views without image review."}
    out = ARTIFACTS / ("habitat_gs_receding_owner_report.json" if args.candidate
                       else "habitat_gs_owner_loop_report.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
