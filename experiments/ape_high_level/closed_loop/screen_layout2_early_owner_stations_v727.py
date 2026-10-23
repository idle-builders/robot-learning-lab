"""Evaluator-only station screen on the saved layout-2 sensor survey.

This ranks *where to test next*. Scripted prop paths and the saved survey are
not observations available to the live actor, and no image or episode is run.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout2-early-station-screen-v727"
START = np.array([1.05, -4.20])
PERSON_SURFACE = np.array([.82104, -2.52133])  # V697 actor's first RGB-D estimate
PROP_START = np.array([0., -4.7])
STRAIGHT_END = np.array([.8, -3.5])
TURN_MID = PROP_START + ((.8 - .15) / 2.5) * (STRAIGHT_END - PROP_START)
TURN_END = np.array([1., -3.9])
BODY_RADIUS_M = .17
PROP_RADIUS_M = .24
REQUIRED_GAP_M = .05
UNCERTAINTY_RESERVE_M = .10  # screen only; not a calibrated uncertainty bound


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_to(free, xs, ys, goal):
    def cell(xy):
        return (int(np.argmin(abs(ys - xy[1]))),
                int(np.argmin(abs(xs - xy[0]))))

    origin, dest = cell(START), cell(goal)
    if not free[origin] or not free[dest]:
        return None
    queue = [(0., origin)]
    cost = {origin: 0.}
    parent = {}
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        value, at = heapq.heappop(queue)
        if at == dest:
            break
        if value > cost[at] + 1e-8:
            continue
        for dy, dx in offsets:
            nxt = at[0] + dy, at[1] + dx
            if not (0 <= nxt[0] < len(ys) and 0 <= nxt[1] < len(xs)) or not free[nxt]:
                continue
            if dx and dy and (not free[at[0], nxt[1]] or not free[nxt[0], at[1]]):
                continue
            extra = math.hypot(dx, dy) * .1
            if value + extra >= cost.get(nxt, math.inf):
                continue
            cost[nxt], parent[nxt] = value + extra, at
            heapq.heappush(queue, (value + extra, nxt))
    if dest not in cost:
        return None
    path = [dest]
    while path[-1] != origin:
        path.append(parent[path[-1]])
    return np.array([[xs[c[1]], ys[c[0]]] for c in reversed(path)]), cost[dest]


def gap_to_path(point, waypoints):
    closest = math.inf
    for first, last in zip(waypoints[:-1], waypoints[1:]):
        delta = last - first
        fraction = np.clip(np.dot(point - first, delta) / np.dot(delta, delta), 0., 1.)
        closest = min(closest, np.linalg.norm(point - first - fraction * delta))
    return float(closest - BODY_RADIUS_M - PROP_RADIUS_M)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(PRIOR) as survey:
        xs, ys = survey["xs"], survey["ys"]
        free = survey["free"] & ~survey["occupied"]
    bearing = START - PERSON_SURFACE
    bearing /= np.linalg.norm(bearing)
    lateral = np.array([-bearing[1], bearing[0]])
    paths = {"straight": np.array([PROP_START, STRAIGHT_END]),
             "turned": np.array([PROP_START, TURN_MID, TURN_END])}
    rows = []
    for standoff in np.arange(.30, .86, .05):
        for offset in np.arange(-.90, -.34, .05):
            goal = PERSON_SURFACE + standoff * bearing + offset * lateral
            route = route_to(free, xs, ys, goal)
            gaps = {name: gap_to_path(goal, path) for name, path in paths.items()}
            rows.append({"standoff_m": round(float(standoff), 3),
                         "lateral_m": round(float(offset), 3),
                         "goal_xy_m": goal.round(4).tolist(),
                         "known_free_route": route is not None,
                         "route_length_m": None if route is None else round(route[1], 3),
                         "scripted_hold_gap_m": {key: round(value, 4)
                                                 for key, value in gaps.items()},
                         "screen_pass": route is not None and
                           min(gaps.values()) >= REQUIRED_GAP_M + UNCERTAINTY_RESERVE_M})
    viable = sorted((row for row in rows if row["screen_pass"]),
                    key=lambda row: (-min(row["scripted_hold_gap_m"].values()),
                                     row["route_length_m"]))
    report = {"schema": "robot-learning-lab.layout2-early-station-screen.v1",
              "scope": "Evaluator-only candidate search, no new simulation episode or camera test",
              "prior_map_sha256": sha(PRIOR),
              "source_sha256": sha(Path(__file__)),
              "person_surface_source": "V697 first actor RGB-D person estimate",
              "scripted_prop_paths": {key: value.tolist() for key, value in paths.items()},
              "provisional_body_radius_m": BODY_RADIUS_M,
              "prop_radius_m": PROP_RADIUS_M,
              "required_gap_m": REQUIRED_GAP_M,
              "uncalibrated_screen_reserve_m": UNCERTAINTY_RESERVE_M,
              "n_candidates": len(rows), "n_screen_pass": len(viable),
              "ranked_screen_pass": viable,
              "all_candidates": rows,
              "limits": ["The actor cannot use the future scripted prop path.",
                         "Reachable survey cells do not certify an executed route or view.",
                         "Pose, prop speed, sensor loss and response delay are not qualified."]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    scale = 110
    image = Image.new("RGB", (len(xs) * 5, len(ys) * 5), "#222631")
    pixels = image.load()
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            color = (226, 232, 220) if free[iy, ix] else (46, 52, 60)
            for yy in range(5):
                for xx in range(5):
                    pixels[ix * 5 + xx, (len(ys) - 1 - iy) * 5 + yy] = color
    draw = ImageDraw.Draw(image)
    def pixel(point):
        return (float((point[0] - xs[0]) / .1 * 5),
                float((ys[-1] - point[1]) / .1 * 5))
    for path in paths.values():
        draw.line([pixel(p) for p in path], fill="#df705d", width=3)
    for row in rows:
        x, y = pixel(row["goal_xy_m"])
        color = "#2f9a6d" if row["screen_pass"] else "#8f6570"
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    for point, color in ((START, "#2383dc"), (PERSON_SURFACE, "#e6b348")):
        x, y = pixel(point)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    if viable:
        path, _ = route_to(free, xs, ys, viable[0]["goal_xy_m"])
        draw.line([pixel(p) for p in path], fill="#326fd3", width=2)
    image.resize((len(xs) * 10, len(ys) * 10)).save(OUT / "station-screen.png")
    print(json.dumps({"screen_pass": len(viable),
                      "top_candidates": viable[:5]}, indent=2))


if __name__ == "__main__":
    main()
