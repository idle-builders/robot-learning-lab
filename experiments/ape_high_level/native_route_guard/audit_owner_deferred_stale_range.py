"""Independent stale-packet and full deferred-action audit for V68."""

from __future__ import annotations

import json
from pathlib import Path

import imageio

import audit_owner_deferred_multi_surface as base_audit
import owner_deferred_stale_range as v68


v64, runner, v52 = v68.v64, v68.runner, v68.v52
OUT = v68.OUT


def main():
    report = json.loads((OUT / "report.json").read_text())
    assert report["source_sha256"] == v52.sha(v68.__file__)
    assert report["v64_report_sha256"] == v52.sha(v64.OUT / "report.json")
    assert report["scene_sha256"] == v52.sha(runner.v35.moving.SCENE_XML)
    assert report["current_design_pointer_sha256"] == v52.sha(
        v68.ROOT / "robot/ape-current-design.json")
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    scanner = v64.v51.v46.plane_packet(runner.observed_packet,
                                       (.05, .30))
    healthy = json.loads((v64.OUT / "report.json").read_text())
    healthy_by_path = {x["path"]: x["run"] for x in healthy["results"]}
    checked = []
    for item in report["results"]:
        last = None
        difference = []

        def expected(model, data, pose, case, tick, *, person_xy=None,
                     hidden_return=False):
            nonlocal last
            fresh = scanner(model, data, pose, case, tick,
                            person_xy=person_xy,
                            hidden_return=hidden_return)
            if tick in v68.STALE_TICKS:
                assert last is not None
                difference.append(max(abs(a-b) for a, b in zip(last, fresh)
                                      if a is not None and b is not None))
                return list(last)
            last = list(fresh)
            return fresh

        checked_row = base_audit.check(
            item, report, source, scenario, monitor_scanner=expected,
            require_margin=False)
        assert len(difference) == len(v68.STALE_TICKS)
        assert max(difference) > .10
        path = item["path"]
        rows = json.loads(Path(item["run"]["trace"]["path"]).read_text())
        fresh_rows = json.loads(Path(healthy_by_path[path]["trace"]["path"])
                                .read_text())
        # Match every actor row before the injected watch packet. The
        # movement trajectory diverges only after the sensor fault.
        first_watch = next(i for i, row in enumerate(rows) if row["stage"] ==
                           "deferred_range_watch")
        assert rows[:first_watch] == fresh_rows[:first_watch]
        watch = [row for row in rows if row["stage"] ==
                 "deferred_range_watch"]
        base_watch = [row for row in fresh_rows if row["stage"] ==
                      "deferred_range_watch"]
        assert watch[:v68.STALE_TICKS[0]-1000] == \
            base_watch[:v68.STALE_TICKS[0]-1000]
        for tick in v68.STALE_TICKS:
            assert watch[tick-1000]["actor_packet"] == \
                watch[v68.STALE_TICKS[0]-1001]["actor_packet"]
        checked_row["maximum_hidden_packet_difference_m"] = round(
            max(difference), 5)
        checked_row["matched_before_stale_watch_rows"] = \
            v68.STALE_TICKS[0]-1000
        checked.append(checked_row)
    assert checked[0]["alarm_s"] is None
    comparison = json.loads((OUT / "comparison.json").read_text())
    assert comparison["v68_report_sha256"] == v52.sha(OUT / "report.json")
    reader = imageio.get_reader(comparison["video"]["path"])
    try:
        assert reader.count_frames() == comparison["video"]["frames"]
        assert reader.get_data(reader.count_frames()-1).shape == \
            tuple(comparison["video"]["shape"])
    finally:
        reader.close()
    assert v52.sha(comparison["video"]["path"]) == \
        comparison["video"]["sha256"]
    audit = {"passed": True, "rows": checked,
             "comparison_frames": comparison["video"]["frames"],
             "scope": "matched prefixes, independently regenerated true and stale scans, cluster forecasts, prior-packet commands, 20ms MuJoCo gaps and decoded videos"}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(OUT / "audit.json")


if __name__ == "__main__":
    main()
