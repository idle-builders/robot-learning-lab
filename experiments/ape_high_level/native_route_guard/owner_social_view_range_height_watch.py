"""V89: low/mid height ablation only during the V87/V88 owner-image watch.

The fused 5+30 cm plane controls the entire pre-show route in every run.
The held baseline and inference watch alone switch to 5 cm-only or 30 cm-only.
The full owner task, actor view pose, cameras, prop scripts and near-body
controller otherwise stay fixed.
"""

from __future__ import annotations

import json

import owner_radial_actor_episode as v86
import owner_radial_social_view_episode as v87
import owner_radial_social_view_new_turns as v88


OUT = v86.ROOT / "artifacts/ape-native-route-guard/v89-social-view-range-height"
CASES = ("outward_normal", "grid_x0_y2", "grid_x1_y1")
PATHS = tuple(item for item in v86.v85.v81.PATHS
              if item[0] == "outward_normal") + tuple(
                  item for item in v88.PATHS
                  if item[0] in ("grid_x0_y2", "grid_x1_y1"))
MODES = {"low_5cm": (.05,), "mid_30cm": (.30,)}


def run_mode(name, held_heights):
    factory_owner = v86.v82.v77.v72.v64.v51.v46
    original_factory = factory_owner.plane_packet
    original_class, original_cases, original_out, original_paths = (
        v86.ActorRadialPose, v86.CASES, v86.OUT, v86.v85.v81.PATHS)

    def staged_plane_packet(base, requested_heights):
        assert requested_heights == (.05, .30)
        fused = original_factory(base, (.05, .30))
        held = original_factory(base, held_heights)

        def observe(model, data, actual, case, tick, **kwargs):
            sensor = held if tick >= 900 else fused
            return sensor(model, data, actual, case, tick, **kwargs)

        return observe

    try:
        factory_owner.plane_packet = staged_plane_packet
        v86.ActorRadialPose = v87.SocialViewPose
        v86.CASES = CASES
        v86.OUT = OUT / name
        v86.v85.v81.PATHS = PATHS
        v86.main()
    finally:
        factory_owner.plane_packet = original_factory
        (v86.ActorRadialPose, v86.CASES, v86.OUT,
         v86.v85.v81.PATHS) = (
            original_class, original_cases, original_out, original_paths)
    path = OUT / name / "report.json"
    report = json.loads(path.read_text())
    sha = v86.v82.v77.v72.v52.sha
    report["schema"] = "robot-learning-lab.ape-social-view-range-height.v1"
    report["v86_source_sha256"] = report["source_sha256"]
    report["source_sha256"] = sha(__file__)
    report["v87_report_sha256"] = sha(v87.OUT / "report.json")
    report["v88_report_sha256"] = sha(v88.OUT / "report.json")
    report["height_mode"] = name
    report["pre_show_heights_m"] = (.05, .30)
    report["held_watch_heights_m"] = held_heights
    report["scope"] = (
        "Matched full owner episodes: fused 5+30cm pre-show pose and "
        "movement, then only the image-watch range plane switched. Ideal "
        "rays in one stylized kitchen and historical kinematic body; "
        "no physical range sensor or clearance qualification.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    reports = [run_mode(name, heights) for name, heights in MODES.items()]
    sha = v86.v82.v77.v72.v52.sha
    report = {
        "schema": "robot-learning-lab.ape-social-view-range-height-suite.v1",
        "source_sha256": sha(__file__),
        "v87_report_sha256": sha(v87.OUT / "report.json"),
        "v88_report_sha256": sha(v88.OUT / "report.json"),
        "cases": CASES,
        "modes": [{"height_mode": x["height_mode"],
                   "report_path": str(OUT / x["height_mode"] / "report.json"),
                   "report_sha256": sha(OUT / x["height_mode"] / "report.json"),
                   "outcomes": [
                       {"path": y["path"], "outcome": y["run"]["outcome"],
                        "minimum_prop_gap_m": y["run"]["minimum_prop_gap_m"]}
                       for y in x["results"]]}
                  for x in reports],
        "scope": "Two held-watch ideal range heights, with identical fused pre-show route and owner task on one control and two changed prop turns."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
