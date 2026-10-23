"""Exercise the live blocked-route recovery with medium depth or depth loss.

This keeps V322's owner task, active looks, hypothetical near-body range,
four-shin rescan, depth-spike filter and motion gate. Only the head-depth
valid window/health changes. Extra cases vary the scripted prop timing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_quality import reject_isolated_depth_spikes
from depth_voxel import DepthVoxelMap
from run import ROOT, sha
import run_blocked_forward_viewpoint_v321 as live


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v246p"
CASES = (
    (170, "present", .30, 9.8, 10.4, 10.8, 12.8),
    (172, "present", .30, 9.8, 10.4, 10.8, 12.8),
    (174, "present", .30, 9.4, 10.2, 10.8, 12.8),
    (176, "present", .30, 9.8, 10.4, 12.2, 14.2),
)
PROFILES = ("medium_30cm_1m", "head_depth_dropout")
DEPTH_WINDOW_M = (.30, 1.00)


class MediumWindowMap(DepthVoxelMap):
    """Let the same ray-map code use the full virtual medium depth window."""

    def observe(self, depth, camera_pose, time_s, *, fovy_deg,
                stride=8, max_depth_m=1.4, self_hit=None,
                self_masked=False):
        return super().observe(
            depth, camera_pose, time_s, fovy_deg=fovy_deg,
            stride=stride, max_depth_m=DEPTH_WINDOW_M[1],
            self_hit=self_hit, self_masked=self_masked)


def profile_depth(profile: str, original):
    def measured(ideal: np.ndarray, mode: str, seed: int,
                 serial: int) -> np.ndarray:
        if mode != "short":
            return original(ideal, mode, seed, serial)
        if profile == "head_depth_dropout":
            return np.full_like(ideal, np.nan)
        depth = ideal.copy()
        rng = np.random.default_rng(seed * 1009 + 900 + serial)
        depth += rng.normal(0., .01, depth.shape).astype(depth.dtype)
        depth[(depth < DEPTH_WINDOW_M[0]) |
              (depth > DEPTH_WINDOW_M[1]) |
              (rng.random(depth.shape) < .15)] = np.nan
        return depth
    return measured


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original_cases = live.CASES
    original_depth = live.noisy_depth
    original_map = live.DepthVoxelMap
    profiles = []
    try:
        live.CASES = CASES
        live.DepthVoxelMap = MediumWindowMap
        for profile in PROFILES:
            live.noisy_depth = profile_depth(profile, original_depth)
            branch = OUT / profile
            live.main(output_root=branch, source=SOURCE,
                      map_depth_filter=reject_isolated_depth_spikes,
                      experiment_label=f"V246p {profile}")
            report_path = branch / "blocked-forward-viewpoint-audit.json"
            report = json.loads(report_path.read_text())
            report["virtual_head_depth_profile"] = profile
            report["virtual_head_depth_range_m"] = (
                None if profile == "head_depth_dropout" else DEPTH_WINDOW_M)
            report["case_variation"] = (
                "Seeds 170/172 retain V322's trajectory; seed174 enters "
                "earlier and seed176 departs later. All four have the same "
                "8cm prop and furnished screen topology. Geometry preflight "
                "is evaluator-only.")
            report["scope_override"] = (
                "V322 owner-loop continuation with a 30cm-1m virtual head "
                "depth range or a completely missing head-depth stream. "
                "The original report scope's short-depth description is "
                "superseded by this field; identical hypothetical near-body "
                "range, shin, local-depth quality and action logic remain.")
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            profiles.append({"name": profile,
                             "report_path": str(report_path.relative_to(ROOT)),
                             "report_sha256": sha(report_path),
                             "rows": [{"case": row["case"],
                                       "status": row["status"],
                                       "recovery_gate": row["recovery_gate"]["status"],
                                       "clearance_m": row[
                                           "evaluator_path_min_clearance_m"]}
                                      for row in report["rows"]]})
    finally:
        live.CASES = original_cases
        live.noisy_depth = original_depth
        live.DepthVoxelMap = original_map
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.medium-depth-recovery.v1",
        "source_sha256": sha(SOURCE),
        "recovery_runner_sha256": sha(Path(live.__file__)),
        "cases": CASES, "profiles": profiles,
        "scope": "Complete historical articulated 50cm ape owner episodes; "
                 "medium head depth versus complete head-depth loss with "
                 "matched hypothetical range and shin sources. Kinematic "
                 "motion, assumed obstacle speed and unbuilt mounts do not "
                 "qualify a sensor or physical stopping."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
