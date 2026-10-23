"""Stress the V497 stationary social stop with an approaching owner proxy.

The owner translates south during the real interval occupied by slow image
recognition. The ape remains stopped. Paired fixed-owner scans use identical
noise, so changed-beam counts describe available range evidence; they are
not a safety response. Full geometry labels contact separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import robocasa_owner_person as person
import run_robocasa_layout2_route_v407 as route
from audit_robocasa_layout2_routes_v409 import sha256
from multi_novel_range_objects import MultiNovelRangeObjects
from run_robocasa_scan_height_v433 import height_scan
from screen_robocasa_owner_safe_stop_v494 import aim_camera
from screen_robocasa_owner_person_v489 import gap_to_room


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-owner-approach-v500"
ROUTE = ROOT / "artifacts/ape-robocasa-person-multitrack-v492/report.json"
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
TORSO = np.array([1., -2.4])
HAND = np.array([1.30, -2.80, .55])
BEGIN_S, END_S = 4., 6.


def place_owner(model, south_m):
    torso = TORSO + [0., -south_m]
    hand = HAND + [0., -south_m, 0.]
    person.set_owner_pose(model, torso, hand, show_can=True)
    reach = hand[:2] - torso
    reach /= np.linalg.norm(reach)
    model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14 * reach


def changed_beams(a, b):
    return sum((x is None) != (y is None) or
               (x is not None and y is not None and abs(x-y) >= .05)
               for x, y in zip(a, b))


def nearest(model, data, ape_geom, names):
    best = {"signed_gap_m_capped_at_50cm": .5, "geom": None}
    for name in names:
        other = model.geom(name).id
        gap = float(mujoco.mj_geomDistance(model, data, ape_geom, other, .5, None))
        if gap < best["signed_gap_m_capped_at_50cm"]:
            best = {"signed_gap_m_capped_at_50cm": gap, "geom": name}
    return best


def scene(ape_xy):
    model = person.scene_with_person(prop_xy=[3., -1.05])
    model.body_pos[model.body("audit_ape").id, :2] = ape_xy
    place_owner(model, 0.)
    aim_camera(model, "owner_wide20", ape_xy, TORSO, 20)
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    route_report = json.loads(ROUTE.read_text())
    ape = np.asarray(route_report["cases"][2]["episode"]["end_true_xy_m"])
    prior = np.load(PRIOR)
    lowmid = height_scan((.05, .30))
    upper = height_scan((.55,))
    cases = []
    for label, final_south in (("stationary", 0.), ("approach_35cm", .35),
                               ("approach_40cm", .40)):
        model = scene(ape)
        fixed = scene(ape)
        data, fixed_data = mujoco.MjData(model), mujoco.MjData(fixed)
        mujoco.mj_forward(fixed, fixed_data)
        renderer = mujoco.Renderer(model, height=480, width=640)
        tracker = MultiNovelRangeObjects(prior["xs"], prior["ys"],
                                         prior["free"], prior["occupied"])
        ape_geom = model.geom("audit_ape_geom").id
        samples, frames = [], []
        for index, t in enumerate(np.arange(BEGIN_S, END_S + .011, .02)):
            fraction = np.clip((t - BEGIN_S) / (END_S - BEGIN_S), 0., 1.)
            south = float(final_south * fraction)
            place_owner(model, south)
            mujoco.mj_forward(model, data)
            hand = nearest(model, data, ape_geom,
                           ("owner_hand_geom", "owner_forearm_geom",
                            "owner_upper_arm_geom", "owner_torso_geom",
                            "owner_head_geom"))
            can = nearest(model, data, ape_geom, ("shown_can_geom",))
            row = {"time_s": round(float(t), 4), "south_m": south,
                   "person": hand, "can": can}
            if index % 5 == 0:
                rng_seed = 5000 + index
                lm = lowmid(model, data, ape, 5., np.random.default_rng(rng_seed))
                lm0 = lowmid(fixed, fixed_data, ape, 5., np.random.default_rng(rng_seed))
                up = upper(model, data, ape, 5., np.random.default_rng(rng_seed))
                up0 = upper(fixed, fixed_data, ape, 5., np.random.default_rng(rng_seed))
                belief = tracker.update(lm, ape, float(t))
                row["changed_lowmid_beams_vs_fixed"] = changed_beams(lm, lm0)
                row["changed_upper_beams_vs_fixed"] = changed_beams(up, up0)
                row["lowmid_ranges_m"] = lm
                row["upper_ranges_m"] = up
                row["range_track_centers_xy_m"] = [track["center_xy_m"]
                                                    for track in belief["tracks"]]
                renderer.update_scene(data, camera="owner_wide20")
                rgb = renderer.render().copy()
                canvas = Image.fromarray(rgb)
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 640, 38), fill="#111b20")
                color = "#ff736b" if min(hand["signed_gap_m_capped_at_50cm"],
                                          can["signed_gap_m_capped_at_50cm"]) < .05 else "#fff4c4"
                draw.text((10, 12), f"{label} t={t:.1f}s  owner shift={south:.2f}m"
                          f"  hand={hand['signed_gap_m_capped_at_50cm']:.2f}m"
                          f"  item={can['signed_gap_m_capped_at_50cm']:.2f}m", fill=color)
                frames.append(np.asarray(canvas))
            samples.append(row)
        renderer.close()
        video = OUT / f"{label}.mp4"
        imageio.mimsave(video, frames, fps=10, codec="libx264", quality=7)
        room, person_ape = gap_to_room(model, data)
        first_can_5cm = next((x for x in samples if x["can"]["signed_gap_m_capped_at_50cm"] < .05), None)
        first_person_5cm = next((x for x in samples if x["person"]["signed_gap_m_capped_at_50cm"] < .05), None)
        first_lowmid_change = next((x for x in samples if x.get("changed_lowmid_beams_vs_fixed", 0) >= 2), None)
        first_upper_change = next((x for x in samples if x.get("changed_upper_beams_vs_fixed", 0) >= 2), None)
        cases.append({"id": label, "owner_total_south_m": final_south,
                      "first_can_below_5cm": first_can_5cm,
                      "first_person_below_5cm": first_person_5cm,
                      "first_lowmid_2beam_change": first_lowmid_change,
                      "first_upper_2beam_change": first_upper_change,
                      "min_can_gap_m": min(x["can"]["signed_gap_m_capped_at_50cm"] for x in samples),
                      "min_person_gap_m": min(x["person"]["signed_gap_m_capped_at_50cm"] for x in samples),
                      "final_person_room": room, "final_person_ape": person_ape,
                      "samples": samples,
                      "video": {"path": str(video), "sha256": sha256(video),
                                "frames": len(frames)}})
        print(label, "min person", cases[-1]["min_person_gap_m"],
              "min can", cases[-1]["min_can_gap_m"],
              "first can <5cm", first_can_5cm["time_s"] if first_can_5cm else None,
              "first range change", first_lowmid_change["time_s"] if first_lowmid_change else None,
              flush=True)
    route.LIDAR_Z_M = .30
    output = {"source_sha256": sha256(Path(__file__)),
              "person_source_sha256": sha256(Path(person.__file__)),
              "route_report_sha256": sha256(ROUTE),
              "prior_map_sha256": sha256(PRIOR),
              "scope": "scripted 35/40cm whole-owner approaches over 2s from joined safe stop while ape holds; matched fixed-owner low+mid and upper ideal range with identical noise; geometry/contact labels evaluator-only; no gait, braking or live actor interrupt",
              "ape_xy_m": ape.tolist(), "cases": cases}
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
