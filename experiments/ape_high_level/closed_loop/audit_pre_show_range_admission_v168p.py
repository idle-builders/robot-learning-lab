"""Independent packet and joined-label audit for V167p pre-show admission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
from screen_standoff_front_cue_v149p import front_left_stat
import screen_pre_show_range_admission_v167p as v167p


SOURCE = Path(__file__)
REPORT = v167p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = 0
    packets = {}

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v167p.__file__)) == report["source_sha256"],
          "Pre-show source changed")
    check(sha(v167p.v100.V98 /
              "live-moving-owner-offer-audit.json") ==
          report["v98_report_sha256"], "Staging source changed")
    future_file = v167p.v165p.OUT / "report.json"
    check(sha(future_file) == report["v165_report_sha256"],
          "Joined future packet replay changed")
    check(report["current_design_status"] ==
          "rejected_no_selected_mechanical_design",
          "Historical-body scope changed")
    check(len(report["rows"]) == 32, "Pre-show matrix incomplete")
    check({(r["variant"], r["seed"], r["station_x_m"],
            r["layout"], r["occlusion"]) for r in report["rows"]} ==
          {(variant, seed, station, layout, occlusion)
           for variant in v167p.v160p.VARIANTS
           for seed in v167p.v160p.SEEDS
           for station in v167p.v160p.STATIONS_X_M
           for layout in v167p.LAYOUTS
           for occlusion in v167p.OCCLUSIONS},
          "Pre-show cases changed")
    future = json.loads(future_file.read_text())
    future_by_key = {(r["source_case_id"], r["stress"], r["layout"]): r
                     for r in future["rows"] if r["draw"] == 0}
    for row in report["rows"]:
        check(sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"],
              "Staging joint/pose packet changed")
        sources = row["pre_show_packets"]
        check(len(sources) == 9 and
              [p["time_s"] for p in sources] ==
              list(v167p.SCAN_TIMES_S),
              "Pre-show timing incomplete")
        samples = row["pre_show_decision"]["samples"]
        check(len(samples) == 9, "Pre-show sample count changed")
        for index, source in enumerate(sources):
            path = ROOT / source["packet_path"]
            check(sha(path) == source["packet_sha256"],
                  "Pre-show actor packet changed")
            packets[source["packet_path"]] = source["packet_sha256"]
            with np.load(path, allow_pickle=False) as data:
                front = data["front_four_points_xyz_m"]
                pose = data["estimated_pose_xyyaw"]
                forward_count = int(data["forward_source_count"])
                time_s = float(data["capture_time_s"])
            check(time_s == source["time_s"] and
                  len(front) == (source["front_counts"].get(
                      "two_environment_points", 0)+
                      source["front_counts"].get(
                          "four_environment_points", 0)) and
                  forward_count == source["forward_count"],
                  "Pre-show packet source split changed")
            ids = np.arange(len(front))
            if row["occlusion"] == "forward_pair_occluded":
                ids = ids[~(v167p.v165p.front_left_roi(front, pose) &
                            (ids < forward_count))]
            if row["layout"] == "two_forward":
                ids = ids[ids < forward_count]
            support, p10 = front_left_stat(front[ids], pose)
            check(samples[index] == {
                "time_s": time_s,
                "front_left_support": support,
                "front_left_p10_x_m": p10,
            }, "Pre-show actor statistic differs from saved packet")
        supported = sum(sample["front_left_support"] >= 5
                        for sample in samples)
        check(row["pre_show_decision"]["supported_packets"] == supported and
              row["pre_show_decision"]["latest_packet_supported"] ==
              (samples[-1]["front_left_support"] >= 5) and
              row["pre_show_decision"]["provisional_admission"] ==
              (supported >= 5 and
               samples[-1]["front_left_support"] >= 5),
              "Admission differs from saved packets")
        stress = ("saved" if row["occlusion"] == "clear_view"
                  else "forward_pair_occluded")
        future_row = future_by_key[(row["later_saved_source_case_id"],
                                    stress, row["layout"])]
        check(row["later_packet_replay_first_cue_s"] ==
              future_row["first_front_left_cue_s"] and
              row["later_packet_replay_timely_warning"] ==
              future_row["timely_hazard_warning"] and
              row["later_evaluator_5cm_crossing_s"] ==
              future_row["evaluator_first_below_5cm_s"],
              "Joined later warning label changed")
    check(len(packets) == 72,
          "Pre-show packet provenance incomplete")
    continued = json.loads((v167p.v160p.OUT / "report.json").read_text())
    handoffs = 0
    for row in report["rows"]:
        if (row["layout"] != "four_front_side" or
                row["occlusion"] != "clear_view"):
            continue
        later = next(r for r in continued["rows"]
                     if r["variant"] == row["variant"] and
                     r["seed"] == row["seed"] and
                     r["station_x_m"] == row["station_x_m"] and
                     r["scripted_prop_stop_delay_s"] is None)
        monitor = json.loads((ROOT / later["monitor"]["path"]).read_text())
        with np.load(ROOT / row["pre_show_packets"][-1]["packet_path"],
                     allow_pickle=False) as before, np.load(
                ROOT / monitor["frames"][0]["packet_path"],
                allow_pickle=False) as after:
            check(np.array_equal(before["actor_qpos"],
                                 after["actor_qpos"]) and
                  np.array_equal(before["estimated_pose_xyyaw"],
                                 after["estimated_pose_xyyaw"]) and
                  float(before["capture_time_s"]) ==
                  float(after["capture_time_s"]) == 16.2,
                  "Pre-show and later actor state do not match")
        handoffs += 1
    check(handoffs == 8, "Pre-show handoff states incomplete")
    for layout in v167p.LAYOUTS:
        for occlusion in v167p.OCCLUSIONS:
            group = [r for r in report["rows"]
                     if r["layout"] == layout and
                     r["occlusion"] == occlusion]
            summary = report["summary"][f"{layout}/{occlusion}"]
            check(summary["cases"] == len(group) == 8 and
                  summary["provisional_admissions"] == sum(
                      r["pre_show_decision"]["provisional_admission"]
                      for r in group) and
                  summary["admitted_without_later_timely_warning"] == sum(
                      r["pre_show_decision"]["provisional_admission"] and
                      not r["later_packet_replay_timely_warning"]
                      for r in group) and
                  summary["deferred_despite_later_timely_warning"] == sum(
                      not r["pre_show_decision"]["provisional_admission"] and
                      r["later_packet_replay_timely_warning"]
                      for r in group),
                  "Pre-show summary differs from packet decisions")
    result = {
        "schema": "robot-learning-lab.ape-pre-show-range-admission-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "unique_pre_show_actor_packets_checked": len(packets),
        "actor_decisions_checked": len(report["rows"]),
        "exact_actor_state_handoffs_checked": handoffs,
        "outcome": "pass",
    }
    (v167p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
