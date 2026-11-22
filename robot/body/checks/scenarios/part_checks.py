"""Scenarios for the per-part printability checks (geometry, holes, press, bed, overhang) on the selected body: the defects the
body sessions reported on 2026-11-22 (T-r96hnj, T-tqarya), each as the run that shows it. Measurements only."""
from __future__ import annotations

import numpy as np

from evaluate import basics, run_tool, scenario, selected_manifest, truth_pair

SHOULDER = ["R_shoulder_pitch_to_roll_combined", "R_shoulder_roll_to_yaw_combined"]      # U connectors, loadpath_slices of the old ones
TEARDROP = ["R_hip_pitch_to_roll_roll_flip", "R_thigh_roll_flip", "R_shoulder_roll_to_yaw_combined"]
MESH_ONLY = ["R_thigh_shell_front", "R_thigh_shell_back", "R_shin_back_hard_cover", "combined__stage2_lower_pelvis_battery_cradle"]
NARROW_SEAT = ["combined__neck_root_cradle", "combined__stage3_front_power_electronics_carrier"]
HIP_YAW = ["R_hip_yaw_to_pitch_combined", "L_hip_yaw_to_pitch_combined"]


def _check(r, name):
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return rep.get("checks", {}).get(name, {})


@scenario("geometry-stale-loadpath", "truth", "shoulder U connectors whose loadpath_slices were set for the old connectors: slabs outside "
          "the part counted apart, not as severed branches")
def s_loadpath(sid):
    r = run_tool(sid, ["--parts", ",".join(SHOULDER), "--checks", "geometry"])
    g = _check(r, "geometry")
    lp = {n: {k: (p.get("loadpath") or {}).get(k) for k in ("range_mm", "part_extent_mm", "range_inside_part_pct", "slabs_outside_part",
                                                             "severed_or_deadend_branches", "min_slab_area_mm2")}
          for n, p in (g.get("per_part") or {}).items()}
    return dict(basics(r), loadpath=lp, severed_total=(g.get("bad_events") or {}).get("loadpath_severed_branches"),
                slabs_outside_total=(g.get("bad_events") or {}).get("loadpath_slabs_outside_part"))


@scenario("holes-teardrop-boss", "truth", "teardropped and slotted sleeve / screw bores: boss wall over the rays that start in material "
          "(was 0 or 0.079 mm from the open sector)")
def s_teardrop(sid):
    r = run_tool(sid, ["--parts", ",".join(TEARDROP), "--checks", "holes"])
    h = _check(r, "holes")
    out = {}
    for n, p in (h.get("per_part") or {}).items():
        rows = [x for x in p["hole_list"] if x.get("boss_wall_mm") is not None]
        opened = [x for x in rows if x.get("boss_wall_open_pct")]
        out[n] = dict(hardware_holes_with_wall=len(rows), with_open_sector=len(opened),
                      walls_at_0=sum(1 for x in rows if x["boss_wall_mm"] == 0),
                      open_sector_wall_min_mm=min([x["boss_wall_mm"] for x in opened] or [None]),
                      thinnest_mm=min([x["boss_wall_mm"] for x in rows] or [None]))
    return dict(basics(r), per_part=out, boss_walls_under_2_lines=(h.get("bad_events") or {}).get("boss_walls_under_2_lines"))


@scenario("mesh-only-parts", "robust", "printed parts without a native STEP (thigh shells, shin back cover, battery cradle) in geometry, "
          "holes and press: measured from the closed mesh, or counted as not measured with the reason")
def s_mesh_only(sid):
    r = run_tool(sid, ["--parts", ",".join(MESH_ONLY), "--checks", "geometry,holes,press"])
    out = {}
    for c in ("geometry", "holes", "press"):
        res = _check(r, c)
        out[c] = dict(measured=sorted(res.get("per_part") or {}), not_measured=res.get("not_measured"),
                      from_own_mesh=res.get("measured_from_own_mesh") or sorted(n for n, p in (res.get("per_part") or {}).items()
                                                                               if "own closed mesh" in str(p.get("method") or p.get("source"))))
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return dict(basics(r), checks=out, not_applicable=rep.get("not_applicable"))


@scenario("press-narrow-seat", "truth", "T-tqarya: the lowest spacer support of the neck root cradle / power carrier sleeves: open share, "
          "and the nearest other B-rep hole to the sleeve exit (a bore under the exit would sit within the dia 10 annulus)")
def s_press_open(sid):
    r = run_tool(sid, ["--parts", ",".join(NARROW_SEAT), "--checks", "press,holes"])
    pr, ho = _check(r, "press"), _check(r, "holes")
    out = {}
    for n, p in (pr.get("per_part") or {}).items():
        holes = (ho.get("per_part") or {}).get(n, {}).get("hole_list", [])
        for s in sorted(p["sleeves"], key=lambda s: s["counter_face"]["support_pct"])[:2]:
            ex = np.asarray(s["entry_end_mm"], float) + s["length_mm"] * np.asarray(s["axis"], float)
            other = [float(np.linalg.norm(np.asarray(h["centre_mm"]) - ex)) for h in holes
                     if not str(h.get("role", "")).startswith("sleeve bore") or np.linalg.norm(np.asarray(h["centre_mm"]) - np.asarray(s["centre_mm"])) > 1.0]
            cf = s["counter_face"]
            out[f"{n}#{s['index']}"] = dict(support_pct=cf["support_pct"], flat_pct=cf["flat_pct"], open_pct=cf["open_pct"],
                                            nearest_other_hole_mm=round(min(other), 2) if other else None)
    return dict(basics(r), lowest_support=out)


@scenario("overhang-bed-directions", "truth", "roll-flip hip and thigh connectors and the hip yaw connector in all six print directions: "
          "bed contact (and failures), floating flat ceilings at the bed, bed contact by axis from one run")
def s_directions(sid):
    parts = ["R_hip_pitch_to_roll_roll_flip", "R_thigh_roll_flip", "R_hip_yaw_to_pitch_combined"]
    per = {}
    for ax in ("+x", "-x", "+y", "-y", "+z", "-z"):
        r = run_tool(f"{sid}-{ax.replace('+', 'p').replace('-', 'm')}", ["--parts", ",".join(parts), "--checks", "bed,overhang", "--process", f"up={ax}"])
        b, o = _check(r, "bed"), _check(r, "overhang")
        per[ax] = dict(rc=r["rc"], bed_failed=b.get("failed"), bed_failed_parts=sorted(b.get("failed_parts") or {}),
                       bed_contact_mm2={n: p.get("contact_mm2") for n, p in (b.get("per_part") or {}).items()},
                       floating_ceilings_under_0p5_mm={n: sum(1 for f in p["flat_ceilings"] if f["floats"] and f["height_mm"] < 0.5)
                                                       for n, p in (o.get("per_part") or {}).items()},
                       longest_bridge_mm={n: p["longest_bridge_span_mm"] for n, p in (o.get("per_part") or {}).items()})
    r = run_tool(sid, ["--parts", ",".join(parts), "--checks", "bed"])
    one = {n: p.get("contact_by_axis_mm2") for n, p in (_check(r, "bed").get("per_part") or {}).items()}
    agree = {n: all(one.get(n, {}).get(ax) == per[ax]["bed_contact_mm2"].get(n) for ax in per) for n in parts}
    return dict(per_direction=per, contact_by_axis_one_run=one, by_axis_equals_direction_runs=agree)


@scenario("overhang-shell-mating", "truth", "thigh shells printed +-y: beyond-limit area on mating faces at standing (the inner seat "
          "on the thigh connector) and its distance to the connector")
def s_shell_mating(sid):
    r = run_tool(sid, ["--parts", "R_thigh_shell_front,R_thigh_shell_back", "--checks", "overhang"])
    o = _check(r, "overhang")
    return dict(basics(r), on_mating_faces={n: dict(up=p["up"], mm2=p["beyond_limit_on_mating_faces_mm2"], by_part=p["beyond_limit_on_mating_faces_by_part_mm2"],
                                                    beyond_limit_mm2=p["beyond_limit_mm2"]) for n, p in (o.get("per_part") or {}).items()})


@scenario("geometry-sleeve-overlap", "truth", "hip yaw connectors vs their own press-fit sleeves: reported overlap vs the mesh boolean vs "
          "exact OCC truth computed here")
def s_sleeve_overlap(sid):
    r = run_tool(sid, ["--parts", ",".join(HIP_YAW), "--checks", "geometry"])
    g = _check(r, "geometry").get("per_part") or {}
    man = selected_manifest()
    return dict(basics(r), per_part={n: dict(reported_mm3=g.get(n, {}).get("sleeve_overlap_mm3"), mesh_mm3=g.get(n, {}).get("sleeve_overlap_mesh_mm3"),
                                             method=g.get(n, {}).get("sleeve_overlap_method"), truth_mm3=truth_pair(man, n, f"{n}_steel_sleeves")[0])
                                     for n in HIP_YAW})
