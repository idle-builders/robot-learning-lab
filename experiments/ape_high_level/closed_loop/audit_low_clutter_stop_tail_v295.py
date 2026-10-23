"""V295: source-body stop-tail stress on the low-clutter live route."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_true_low_obstacle_v289 import set_full_height
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from preflight_lateral_clutter_v282 import make_room
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v295"
STOP = ROOT / "artifacts/ape-skill-contract-cove50-20261009/stop-trace.json"
V284 = ROOT / "artifacts/ape-closed-loop-cove50-v284/shin-search-audit.json"
V290 = ROOT / "artifacts/ape-closed-loop-cove50-v290/true4cm-shin-search-audit.json"
SCENARIOS = ((.04, -.0704, "executed_shin_search"),
             (.04, -.1408, "counterfactual_second_step"),
             (.08, -.0704, "executed_shin_search"))
SAMPLES = 51


def panel(room, label: str) -> np.ndarray:
    rgb, _ = room.capture("lower_rgb", depth=False)
    external = room.capture_external()
    image = Image.fromarray(np.concatenate((np.asarray(rgb),
                                             np.asarray(external)), axis=1))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 29), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255))
    return np.asarray(image)


def audit(height: float, start_y: float, label: str, tail_m: float) -> dict:
    room = make_room(138, "present")
    set_full_height(room, height)
    obstacles = scene_geoms(room)
    bodies = body_geoms(room, obstacles)
    room.actual[:] = (0., start_y, 0.)
    room.estimated[:] = room.actual
    room._forward()
    rows = []
    video_path = OUT / f"height{int(height*100)}cm-{label}-source-tail-stress.mp4"
    try:
        with imageio.get_writer(video_path, fps=5, codec="libx264", quality=7) as video:
            for index, extra in enumerate(np.linspace(0., tail_m, SAMPLES)):
                room.actual[:] = (0., start_y - extra, 0.)
                room._forward()
                gap, robot, scene = full_body_clearance(room, obstacles, bodies)
                rows.append({"extra_m": round(float(extra), 5),
                             "model_clearance_m": round(float(gap), 5),
                             "limiting_robot_geom": robot,
                             "limiting_scene_geom": scene})
                if index % 3 == 0 or index == SAMPLES - 1:
                    video.append_data(panel(
                        room, f"{int(height*100)}cm floor object | source-body stop-tail stress {extra*100:.1f}cm | model gap {gap*100:.1f}cm"))
    finally:
        room.close()
    return {"physical_object_height_m": height,
            "route_state": label,
            "start_y_m": start_y,
            "start_clearance_m": rows[0]["model_clearance_m"],
            "minimum_clearance_m": min(row["model_clearance_m"] for row in rows),
            "first_5cm_violation_extra_m": next((row["extra_m"] for row in rows
                                                  if row["model_clearance_m"] < .05), None),
            "first_contact_extra_m": next((row["extra_m"] for row in rows
                                           if row["model_clearance_m"] < 0.), None),
            "rows": rows, "video_sha256": sha(video_path)}


def main() -> None:
    stop = json.loads(STOP.read_text())
    upper = float(stop["summary"]["path_to_observed_stop_m_range"][1])
    if not .10 < upper < .13:
        raise ValueError("Source stop trace changed; stress distance needs review")
    live4 = json.loads(V290.read_text())
    live8 = json.loads(V284.read_text())
    for report, height in ((live4, .04), (live8, .08)):
        if not all(row["evaluator_min_model_clearance_m"] >= .09
                   for row in report["rows"]):
            raise ValueError(f"The {height}m live route changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [audit(height, y, label, upper) for height, y, label in SCENARIOS]
    report = {"scope": "V295 evaluator-only hypothetical straight stop-tail transfer on the exact selected Direct Frame live V284/V290 low-clutter search. Start after the executed first-side step y=-7.04cm for both 4cm and 8cm objects, plus a separate counterfactual y=-14.08cm true4cm state that a less conservative head gate could reach. Continue selected-body pose toward the object over the upper 11.314cm path length measured after zero twist in a different source-body native policy; sample 51 positions and render videos. This is neither the selected Direct Frame's native stop trajectory nor a physical bound: source masses/policy, yaw/slip/contact, capture latency and dynamic obstacles differ. A failing stress screen invalidates an instant-stop safety interpretation; a pass would not certify deployment.",
              "source_sha256": sha(SOURCE), "stop_trace_sha256": sha(STOP),
              "v284_sha256": sha(V284), "v290_sha256": sha(V290),
              "source_body_sha256": stop["source_body_sha256"],
              "source_upper_path_to_observed_stop_m": upper,
              "rows": rows}
    (OUT / "low-clutter-stop-tail-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps([{key: value for key, value in row.items() if key != "rows"}
                      for row in rows], indent=2), flush=True)


if __name__ == "__main__":
    main()
