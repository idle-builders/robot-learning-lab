"""Paired fault: lose head depth only in recovery map captures.

V261p scenes, owner responses, early search and actor thresholds are fixed.
Serials 400+ are the timed recovery/map head packets; their depth pixels are
missing while RGB, pose, near-range and shin sources remain available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
import run_no_supported_view_owner_help_v256p as help_policy
import run_varied_owner_route_v258p as challenge


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v263p"
BASE = ROOT / "artifacts/ape-articulated-depth-route-v261p"
VARIATION_SOURCE = Path(challenge.__file__)
HELP_SOURCE = Path(help_policy.__file__)


def main() -> None:
    old_scene = {seed: spec.copy() for seed, spec in challenge.SCENE.items()}
    old_out, old_source = challenge.OUT, challenge.SOURCE
    old_profile, old_description = challenge.profile_depth, challenge.SENSOR_PROFILE

    def recovery_loss(profile: str, original):
        healthy = old_profile(profile, original)

        def measured(ideal: np.ndarray, mode: str, seed: int,
                     serial: int) -> np.ndarray:
            if mode == "short" and serial >= 400:
                return np.full_like(ideal, np.nan)
            return healthy(ideal, mode, seed, serial)

        return measured

    try:
        challenge.SCENE[180]["prop_x_m"] = -.20
        challenge.SCENE[184]["prop_x_m"] = -.20
        challenge.OUT, challenge.SOURCE = OUT, SOURCE
        challenge.profile_depth = recovery_loss
        challenge.SENSOR_PROFILE = (
            "virtual head depth 30cm-1m through earlier search; all "
            "serial>=400 recovery/map head-depth pixels missing; RGB, "
            "near-range and shin preserved")
        challenge.main()
    finally:
        challenge.SCENE.clear()
        challenge.SCENE.update(old_scene)
        challenge.OUT, challenge.SOURCE = old_out, old_source
        challenge.profile_depth = old_profile
        challenge.SENSOR_PROFILE = old_description
    for source in (VARIATION_SOURCE, HELP_SOURCE):
        (OUT / source.name).write_bytes(source.read_bytes())
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report.update({
        "variation_source_sha256": sha(VARIATION_SOURCE),
        "help_policy_source_sha256": sha(HELP_SOURCE),
        "healthy_v261_report_sha256": sha(BASE / "report.json"),
        "fault": "All timed recovery head-depth map captures are missing; "
                 "earlier side-search head RGB-D, near-range and shin "
                 "sources stay healthy. No scene truth enters actor choice.",
    })
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
