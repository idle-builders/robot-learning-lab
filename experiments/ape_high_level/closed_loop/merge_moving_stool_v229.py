"""Merge V229's independent seed107 and seed111/113 invocations."""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v229"


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def write(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    first_report_name = "seed107-first-pass-live-audit.json"
    first_route_name = "seed107-first-pass-route-audit.json"
    first_geometry_name = "seed107-first-pass-geometry-audit.json"
    report_name = "moving-stool-short-depth-live-audit.json"
    route_name = "rgb-floor-limited-depth-route-audit.json"
    geometry_name = "moving-stool-geometry-audit.json"
    first = read(first_report_name)
    later = read(report_name)
    assert [r["seed"] for r in first["rows"]] == [107]
    assert [r["seed"] for r in later["rows"]] == [111, 113]
    report = {**later,
              "selected_routes": first["selected_routes"] + later["selected_routes"],
              "rows": first["rows"] + later["rows"],
              "source_invocations": [
                  {"seeds": [107], "report_sha256": sha(OUT / first_report_name)},
                  {"seeds": [111, 113], "report_sha256": sha(OUT / report_name)}]}
    route = read(route_name)
    route["rows"] = read(first_route_name)["rows"] + route["rows"]
    assert [r["seed"] for r in route["rows"]] == [107, 111, 113]
    write(route_name, route)
    geometry = read(geometry_name)
    geometry["rows"] = read(first_geometry_name)["rows"] + geometry["rows"]
    assert [r["seed"] for r in geometry["rows"]] == [107, 111, 113]
    write(geometry_name, geometry)
    report["floor_limited_route_sha256"] = sha(OUT / route_name)
    report["moving_stool_geometry_sha256"] = sha(OUT / geometry_name)
    report["merge_source_sha256"] = sha(SOURCE)
    write(report_name, report)
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print([(r["seed"], r["status"], r["moves"]) for r in report["rows"]])


if __name__ == "__main__":
    main()
