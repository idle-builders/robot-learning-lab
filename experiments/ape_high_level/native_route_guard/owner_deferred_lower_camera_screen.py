"""V71: lower, wider fixed near-body RGB-D mount counterfactual."""

from __future__ import annotations

import json

import owner_deferred_camera_direction_screen as v70


OUT = v70.ROOT / "artifacts/ape-native-route-guard/v71-lower-camera-screen"
CAMERA_HEIGHT_M = .25
FOVY_DEG = 90.
PITCH_DEG = (25, 40, 55, 70)


def main():
    v70.main(out=OUT, camera_height_m=CAMERA_HEIGHT_M,
             fovy_deg=FOVY_DEG, pitch_degrees=PITCH_DEG)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.ape-lower-camera-screen.v1"
    report["variant_source_sha256"] = v70.v52.sha(__file__)
    report["v70_report_sha256"] = v70.v52.sha(v70.OUT / "report.json")
    report["scope"] = ("Uniform fixed-yaw/pitch lower 25cm, 90-degree RGB-D "
                       "counterfactual on the same V68 stale-range event; "
                       "no selected physical mount, shell/self-occlusion "
                       "qualification or actor hazard decision.")
    path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
