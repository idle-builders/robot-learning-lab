"""Independent source-split and warning audit for V165p layout comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
import compare_standoff_range_layouts_v165p as v165p


SOURCE = Path(__file__)
REPORT = v165p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = packets = samples = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v165p.__file__)) == report["source_sha256"],
          "Layout comparison source changed")
    check(sha(v165p.v162p.OUT / "report.json") ==
          report["v162_report_sha256"],
          "Source case report changed")
    check(tuple(report["layouts"]) == v165p.LAYOUTS and
          report["stresses"] == v165p.STRESSES,
          "Layout or stress definitions changed")
    check(len(report["source_cases"]) == 16,
          "Source case matrix incomplete")
    for case in report["source_cases"]:
        monitor_file = ROOT / case["monitor_path"]
        check(sha(monitor_file) == case["monitor_sha256"],
              "Source monitor changed")
        monitor = json.loads(monitor_file.read_text())
        for frame in monitor["frames"]:
            packet_file = ROOT / frame["packet_path"]
            check(sha(packet_file) == frame["packet_sha256"],
                  "Source range packet changed")
            with np.load(packet_file, allow_pickle=False) as packet:
                front = packet["front_four_points_xyz_m"]
            split = frame["front_counts"].get(
                "two_environment_points", 0)
            side = frame["front_counts"].get(
                "four_environment_points", 0)
            check(len(front) == split+side,
                  "Forward/angled sensor source split invalid")
            packets += 1
    expected = (16*sum(profile["draws"]
                       for profile in v165p.STRESSES.values())*
                len(v165p.LAYOUTS))
    check(len(report["rows"]) == expected == 336,
          "Layout comparison matrix incomplete")
    by_trial = {}
    for row in report["rows"]:
        key = (row["source_case_id"], row["stress"], row["draw"])
        check(row["layout"] not in by_trial.setdefault(key, {}),
              "Duplicate layout trial")
        by_trial[key][row["layout"]] = row
        sequence = row["samples"]
        check(len(sequence) >= 40, "Layout packet sequence incomplete")
        first = next((s["time_s"] for s in sequence if s["cue_now"]), None)
        check(first == row["first_front_left_cue_s"],
              "First cue differs from saved samples")
        support_pre = sum(s["front_left_support"] >= 5
                          for s in sequence[:9])
        check(support_pre ==
              row["supported_pre_show_packets_16p2_to_17p0"] and
              (support_pre >= 5) == row["illustrative_admission_5_of_9"],
              "Illustrative admission differs from saved packets")
        for index, sample in enumerate(sequence):
            if sample["front_left_support"] < 5:
                check(sample["front_left_p10_x_m"] is None,
                      "Unsupported range treated as valid")
            if index >= 2 and sample["closing_0p2s_m"] is not None:
                older = sequence[index-2]["front_left_p10_x_m"]
                current = sample["front_left_p10_x_m"]
                check(older is not None and current is not None and
                      abs(sample["closing_0p2s_m"]-
                          round(older-current, 5)) < 1e-8,
                      "Closing value differs from packet statistics")
                check(sample["cue_now"] == (
                    sample["closing_0p2s_m"] >=
                    report["frozen_front_cue_threshold_m_per_0p2s"]),
                    "Cue differs from frozen threshold")
            samples += 1
        breach = row["evaluator_first_below_5cm_s"]
        check(row["timely_hazard_warning"] == (
            row["class"] == "hazard" and first is not None and
            breach is not None and first <= breach),
            "Timely-warning classification changed")
        check(row["false_control_warning"] == (
            row["class"] == "control" and first is not None),
            "False-warning classification changed")
    check(len(by_trial) == 112 and
          all(set(trial) == set(v165p.LAYOUTS)
              for trial in by_trial.values()),
          "Matched layout trials incomplete")
    rear_differences = sum(
        any(a["front_left_support"] != b["front_left_support"] or
            a["front_left_p10_x_m"] != b["front_left_p10_x_m"]
            for a, b in zip(trial["four_front_side"]["samples"],
                            trial["four_plus_right_rear"]["samples"]))
        for trial in by_trial.values())
    check(rear_differences == 0,
          "Rear array changed this front-left ROI")
    for stress in v165p.STRESSES:
        for layout in v165p.LAYOUTS:
            group = [row for row in report["rows"]
                     if row["stress"] == stress and row["layout"] == layout]
            hazard = [row for row in group if row["class"] == "hazard"]
            control = [row for row in group if row["class"] == "control"]
            item = report["summary"][stress][layout]
            check(item["hazard_trials"] == len(hazard) and
                  item["timely_hazard_warnings"] == sum(
                      row["timely_hazard_warning"] for row in hazard) and
                  item["controls"] == len(control) and
                  item["false_control_warnings"] == sum(
                      row["false_control_warning"] for row in control) and
                  item["illustrative_pre_show_admissions"] == sum(
                      row["illustrative_admission_5_of_9"] for row in group),
                  "Layout summary differs from rows")
    result = {
        "schema": "robot-learning-lab.ape-standoff-range-layout-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "source_actor_packets_checked": packets,
        "layout_packet_samples_checked": samples,
        "matched_trials": len(by_trial),
        "rear_addition_changed_front_left_trials": rear_differences,
        "outcome": "pass",
    }
    (v165p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
