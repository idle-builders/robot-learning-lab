"""Independent provenance and derived-outcome audit for V162p packet stress."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from run import ROOT, sha
import stress_standoff_range_packets_v162p as v162p


SOURCE = Path(__file__)
REPORT = v162p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = packets = videos = sample_checks = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v162p.__file__)) == report["source_sha256"],
          "Packet stress source changed")
    check(sha(v162p.v160p.OUT / "report.json") ==
          report["v160_report_sha256"], "Variant source report changed")
    check(sha(v162p.v148p.OUT / "report.json") ==
          report["v148_report_sha256"], "Control source report changed")
    check(report["profiles"] == v162p.PROFILES,
          "Perturbation profiles changed")
    cases = report["source_cases"]
    check(len(cases) == 16 and
          len({case["id"] for case in cases}) == 16,
          "Source case matrix incomplete")
    check(sum(case["class"] == "hazard" for case in cases) == 8 and
          sum(case["class"] == "control" for case in cases) == 8,
          "Source cases lack controls or hazards")
    for case in cases:
        monitor_file = ROOT / case["monitor_path"]
        check(sha(monitor_file) == case["monitor_sha256"],
              "Source monitor changed")
        monitor = json.loads(monitor_file.read_text())
        for frame in monitor["frames"]:
            check(sha(ROOT / frame["packet_path"]) ==
                  frame["packet_sha256"], "Source actor packet changed")
            packets += 1
        video = ROOT / case["video_path"]
        check(sha(video) == case["video_sha256"],
              "Source case video changed")
        capture = cv2.VideoCapture(str(video))
        decoded = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            decoded += 1
            check(image.ndim == 3 and image.shape[0] > 0,
                  "Source video frame malformed")
        capture.release()
        check(decoded >= 20, "Source video incomplete")
        videos += 1
        if case["class"] == "hazard":
            check(case["first_below_5cm_s"] is not None,
                  "Hazard has no modeled gap crossing")
        else:
            check(case["first_below_5cm_s"] is None,
                  "Control crosses modeled gap margin")
    expected_rows = len(cases)*sum(profile["draws"]
                                   for profile in v162p.PROFILES.values())
    check(len(report["rows"]) == expected_rows == 256,
          "Perturbation matrix incomplete")
    seen = set()
    by_id = {case["id"]: case for case in cases}
    for row in report["rows"]:
        key = (row["source_case_id"], row["profile"], row["draw"])
        check(key not in seen, "Duplicate perturbation row")
        seen.add(key)
        case = by_id[row["source_case_id"]]
        check(row["class"] == case["class"] and
              row["evaluator_first_below_5cm_s"] ==
              case["first_below_5cm_s"],
              "Row source identity changed")
        samples = row["samples"]
        check(len(samples) >= 40, "Perturbed packet sequence incomplete")
        first = next((s["time_s"] for s in samples if s["cue_now"]), None)
        check(first == row["first_front_left_cue_s"],
              "First warning differs from sample sequence")
        for index, sample in enumerate(samples):
            if sample["front_left_support"] < 5:
                check(sample["front_left_p10_x_m"] is None,
                      "Unsupported points produced a valid range")
            if index >= 2 and sample["closing_0p2s_m"] is not None:
                older = samples[index-2]["front_left_p10_x_m"]
                current = sample["front_left_p10_x_m"]
                check(older is not None and current is not None and
                      abs(sample["closing_0p2s_m"]-
                          round(older-current, 5)) < 1e-8,
                      "Closing cue differs from packet statistics")
                check(sample["cue_now"] == (
                    sample["closing_0p2s_m"] >=
                    report["frozen_front_cue_threshold_m_per_0p2s"]),
                    "Warning differs from frozen threshold")
            sample_checks += 1
        lead = (case["first_below_5cm_s"]-first
                if case["first_below_5cm_s"] is not None and
                first is not None else None)
        check(row["timely_hazard_warning"] == (
            case["class"] == "hazard" and
            lead is not None and lead >= -1e-8),
            "Timely-warning classification changed")
        check(row["false_control_warning"] == (
            case["class"] == "control" and first is not None),
            "False-warning classification changed")
    for name in v162p.PROFILES:
        rows = [row for row in report["rows"]
                if row["profile"] == name]
        hazards = [row for row in rows if row["class"] == "hazard"]
        controls = [row for row in rows if row["class"] == "control"]
        summary = report["summary"][name]
        check(summary["hazard_trials"] == len(hazards) and
              summary["timely_hazard_warnings"] == sum(
                  row["timely_hazard_warning"] for row in hazards) and
              summary["controls"] == len(controls) and
              summary["false_control_warnings"] == sum(
                  row["false_control_warning"] for row in controls),
              "Summary counts differ from replay rows")
    result = {
        "schema": "robot-learning-lab.ape-standoff-range-stress-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "source_actor_packets_checked": packets,
        "perturbed_packet_samples_checked": sample_checks,
        "source_case_videos_decoded": videos,
        "outcome": "pass",
    }
    (v162p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
