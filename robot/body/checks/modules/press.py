"""press: sleeves of a scoped part (its sleeve sets: process.relations, the definition's joined_to {how: pressed}, else the sets whose
sleeves touch the part; never a name): press tool / anvil path obstruction (mm3), counter-face coverage, flatness. Screen = mesh booleans, exact = press_lib.score_part. Raw function moved from leg/fast_check.py.
A part without a native STEP (mesh-only shells and carriers) is screened on its own mesh when that mesh is closed, its sleeves from their
STEP or, without one, from their mesh's revolution axes; a part it cannot measure is listed in not_measured with the reason and counted.
A tube that is a heat-set insert by its sleeve set's definition (contains[].spec.kind insert, matched by length and diameters) gets the
insert model (press_lib.insert_row: catalogue tip per thread, entry face, no exit-plane support); a tube matching no item is counted."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks import bodylib as BL
from checks.core import ROOT, R, kit, screen_solid, step_mesh, tick
from checks.registry import register


def _sleeve_axes(body, sn):
    """[(end 0, end 1, unit axis, outer r, inner r) or None per sleeve] of a sleeve part and the source: the B-rep cylinders of its
    STEP, else the revolution axes of its mesh (mount_check.tubes; mesh-only sleeves of the shells)."""
    from checks import press_lib as PC
    ps = body.parts[sn]
    if ps.get("native_step") and BL.src(ps["native_step"]).exists():
        from build123d import Location, import_step
        sl = import_step(str(BL.src(ps["native_step"]))).moved(Location(tuple(float(v) for v in (ps.get("native_step_offset_mm") or (0, 0, 0)))))
        return [PC.sleeve_axis(x) for x in sl.solids()], "STEP"
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from mount_check import tubes
    return [(c - L / 2 * a, c + L / 2 * a, a, od / 2, idd / 2) for c, a, L, od, idd in tubes(body.mesh(sn))], "mesh"


def _part_mesh(body, n):
    """(closed mesh of part n, source) for the screen: its STEP tessellation, else its own mesh when that is closed; (None, why)."""
    p = body.parts[n]
    if p.get("native_step") and BL.src(p["native_step"]).exists():
        return step_mesh(BL.src(p["native_step"]), p.get("native_step_offset_mm"), 0.03, 0.2, largest=True), "STEP tessellation 0.03 mm"
    m = body.mesh(n)
    if m.volume < 0:
        m = m.copy(); m.invert()
    if len(m.faces) and kit.mf(m).status().name == "NoError" and not kit.mf(m).is_empty():
        return m, "own closed mesh (no native STEP)"
    return None, "no native STEP and its mesh is not closed: counter faces and path volumes need a closed solid"


def sleeve_sets(body, n):
    """The sleeve sets pressed into part n (process.relations: the definition's joined_to {how: pressed}, else the sets whose sleeves'
    surface touches the part)."""
    from checks import process as P
    return P.relations(body).sleeve_sets_of(n)


def set_items(rel, sn):
    """The bought items a sleeve set's body definition says it holds (contains, the items for this set), [] without a definition.
    An item whose spec.kind is insert is a heat-set insert (press_lib.insert_row), never told by its name."""
    e = (getattr(rel, "claims", None) or {}).get(sn) if rel is not None else None
    return [it for it in (e or {}).get("contains") or [] if not it.get("part") or it["part"] == sn]


def check_press(body, changed, exact=False, skipped=None, rel=None):
    """{part: result} for the scoped parts with steel sleeves; skipped (dict, optional) collects {part: why} of those not measured.
    A part with several sleeve sets gets the rows of all of them, numbered on. rel (ctx.rel): the sleeve sets' definitions, which
    tell heat-set inserts (spec.kind insert) from sleeves."""
    from checks import press_lib as PC
    import trimesh
    t0 = time.time()
    out = {}
    skipped = {} if skipped is None else skipped
    for n in changed:
        sets = sleeve_sets(body, n)
        if not sets:
            continue
        p = body.parts[n]
        has_step = bool(p.get("native_step")) and BL.src(p["native_step"]).exists()
        if exact and has_step and all(body.parts[sn].get("native_step") for sn in sets):
            from build123d import Compound, Location, import_step
            part = Compound(children=list(import_step(str(BL.src(p["native_step"]))).moved(
                Location(tuple(float(v) for v in (p.get("native_step_offset_mm") or (0, 0, 0))))).solids()))
            rows = []
            for sn in sets:
                ps = body.parts[sn]
                sl = import_step(str(BL.src(ps["native_step"]))).moved(Location(tuple(float(v) for v in (ps.get("native_step_offset_mm") or (0, 0, 0)))))
                rows += [{**r, "index": len(rows) + r["index"]} for r in PC.score_part(max(part.solids(), key=lambda x: x.volume), sl, set_items(rel, sn))]
            out[n] = dict(method="exact OCC (press_lib.score_part)", summary=PC.summary(rows), rows=rows, sleeve_sets=sets)
            continue
        mesh, src = _part_mesh(body, n)
        if mesh is None:
            skipped[n] = src
            continue
        axes, ssrc, items = [], set(), []
        for sn in sets:
            a, w = _sleeve_axes(body, sn)
            axes += a; ssrc.add(w); items += [set_items(rel, sn)] * len(a)
        ssrc = " + ".join(sorted(ssrc))
        pm = kit.mf(mesh); rays = PC._rays(mesh)

        def cyl(p0, d, r, L):
            import manifold3d as M
            d = np.asarray(d, float) / np.linalg.norm(d)
            Rm = trimesh.geometry.align_vectors([0, 0, 1], d)[:3, :3]
            return M.Manifold.cylinder(L, r, r, 96).transform(np.c_[Rm, np.asarray(p0, float)])

        def orient(e_in, e_out, d, r_out):
            tip = (pm ^ cyl(e_in - 0.05 * d, -d, PC.TIP_D / 2, PC.TIP_L)).volume()
            bod = (pm ^ cyl(e_in - (0.05 + PC.TIP_L) * d, -d, PC.BODY_D / 2, PC.TOOL_L - PC.TIP_L)).volume()
            anv = (pm ^ cyl(e_out + 0.2 * d, d, PC.ANVIL_PATH_D / 2, PC.ANVIL_L)).volume()
            a10 = (pm ^ cyl(e_out + 0.2 * d, d, PC.ANVIL_D / 2, PC.TIP_L)).volume()
            return dict(tool_tip_obstruction_mm3=round(tip, 3), tool_body_obstruction_mm3=round(bod, 3),
                        anvil_path_obstruction_mm3=round(anv, 3), anvil_dia10_first15mm_obstruction_mm3=round(a10, 3),
                        counter_face=PC.counter_face(mesh, rays, e_out, d, r_out))
        def vol(p0, d, r, L):
            return float((pm ^ cyl(p0, d, r, L)).volume())
        rows = []
        for i, ax in enumerate(axes):
            if ax is None:
                rows.append(dict(index=i, error="no cylindrical face")); continue
            e0, e1, d, r_out, r_in = ax
            it, sc = PC.item_of((float(np.linalg.norm(e1 - e0)), 2 * r_out, 2 * r_in), items[i])
            if it is not None and (it.get("spec") or {}).get("kind") == "insert":
                rows.append(PC.insert_row(vol, mesh, rays, i, e0, e1, d, r_out, r_in, it["spec"],
                                          f"definition item {it.get('line')} (nearest length / diameters, {sc} mm)"))
                continue
            ksrc = (f"definition item {it.get('line')} ({sc} mm)" if it is not None else
                    "no definition item" + (f" within {PC.ITEM_MATCH_MM} mm (nearest {sc} mm): the set's STEP and its definition's contains disagree" if sc is not None else " (no contains)"))
            a, b = orient(e0, e1, d, r_out), orient(e1, e0, -d, r_out)
            ka = (a["tool_tip_obstruction_mm3"] + a["tool_body_obstruction_mm3"], a["anvil_path_obstruction_mm3"])
            kb = (b["tool_tip_obstruction_mm3"] + b["tool_body_obstruction_mm3"], b["anvil_path_obstruction_mm3"])
            best, entry, other = (a, e0, b) if ka <= kb else (b, e1, a)
            rows.append(dict(index=i, kind="sleeve", kind_source=ksrc, centre_mm=np.round((e0 + e1) / 2, 2).tolist(), axis=np.round(d if best is a else -d, 3).tolist(),
                             entry_end_mm=np.round(entry, 2).tolist(), length_mm=round(float(np.linalg.norm(e1 - e0)), 2),
                             od_mm=round(2 * r_out, 2), id_mm=round(2 * r_in, 2), **best,
                             other_side=dict(tool_mm3=round(other["tool_tip_obstruction_mm3"] + other["tool_body_obstruction_mm3"], 3),
                                             anvil_mm3=other["anvil_path_obstruction_mm3"], coverage_pct=other["counter_face"]["coverage_pct"],
                                             support_pct=other["counter_face"]["support_pct"], free_depth_mm=other["counter_face"]["free_depth_mm"])))
        out[n] = dict(method=f"mesh booleans (manifold3d, 96-gon cylinders) on the part's {src}, sleeve axes from the sleeves' {ssrc}"
                      + (" (exact needs both STEPs)" if exact else ""), summary=PC.summary(rows), rows=rows, sleeve_sets=sets)
    tick("press", t0)
    return out


def _applies(ctx):
    """A sleeve set on a scoped part's link whose box meets the part's (the cheap test; run() asks which part each set is pressed into)."""
    P = ctx.parts
    box = lambda n: np.array(P[n]["bounds_mm"], float)  # noqa: E731
    sets = [s for s in P if ctx.rel.is_sleeve_set(s)]
    return any(P[s]["owning_link"] == P[n]["owning_link"] and not (np.any(box(s)[0] > box(n)[1] + 1) or np.any(box(s)[1] < box(n)[0] - 1))
               for n in ctx.scope if n in P for s in sets)


@register("press", _applies)
def run(ctx):
    """Steel sleeves of a scoped part: counter-side spacer support (flat / wall / recess / open %, free depth mm) and entry-side press tool path obstruction (mm3); the old fixed-post anvil fields stay in the rows."""
    skipped = {}
    raw = check_press(ctx.body, ctx.scope, ctx.exact_press, skipped, ctx.rel)
    per_part, cases = {}, []
    bad = dict(sleeves_with_tool_obstruction=0, sleeves_with_anvil_obstruction=0, sleeves_with_counter_face_below_90pct=0, sleeves_with_support_below_90pct=0,
               inserts=0, inserts_with_face_obstruction=0, inserts_entry_face_flat_below_90pct=0)
    for n, r in raw.items():
        per_part[n] = dict(method=r["method"], summary=r["summary"], sleeves=r["rows"],
                           sleeve_sets={sn: ctx.rel.source(sn, "sleeve_owner") for sn in r["sleeve_sets"]})
        s = r["summary"]
        bad["sleeves_with_tool_obstruction"] += s["with_tool_obstruction"]
        bad["sleeves_with_anvil_obstruction"] += s["with_anvil_obstruction"]
        bad["sleeves_with_counter_face_below_90pct"] += s["counter_face_coverage_below_90pct"]
        bad["sleeves_with_support_below_90pct"] += s["counter_face_support_below_90pct"]
        bad["inserts"] += s["inserts"]; bad["inserts_with_face_obstruction"] += s["inserts_with_face_obstruction"]
        bad["inserts_entry_face_flat_below_90pct"] += s["inserts_entry_face_flat_below_90pct"]
        exp = ctx.settings(n).get("expected_hardware", {}).get("steel_sleeves")
        per_part[n]["sleeves_found"] = s["sleeves"] + s["inserts"]      # every tube of the sleeve sets (as before); inserts_found of them heat-set inserts
        per_part[n]["inserts_found"] = s["inserts"]
        per_part[n]["sleeves_expected_in_settings"] = exp
        for row in r["rows"]:
            if "error" in row:
                continue
            if row.get("kind") == "insert":     # heat-set insert: entry face and tip path, no exit-plane support
                ef = row["entry_face"]
                cases.append(dict(part=n, mating_part=None, pose=None, location_mm=row["centre_mm"], unit="% insert entry face flat (low is bad)",
                                  value=ef["flat_pct"], what=f"heat-set insert {row['index']} ({row['thread']}, {row['kind_source']}): entry face flat {ef['flat_pct']} %, "
                                                            f"wall {ef['wall_pct']} %, open {ef['open_pct']} % (insert OD .. tip shank); tip shank {row.get('tip_shank_dia_mm')} mm needs a "
                                                            f"{row.get('shank_clearance_mm')} mm working end; bore beyond the far end {row['exit_side']['bore_beyond_exit_mm']} mm"))
                if row["tool_tip_obstruction_mm3"] > 0:
                    cases.append(dict(part=n, mating_part=None, pose=None, location_mm=row["centre_mm"], unit="mm3 press tool path obstruction",
                                      value=row["tool_tip_obstruction_mm3"], what=f"heat-set insert {row['index']}: part inside the insert's outer diameter beyond its entry end"))
                continue
            cf = row["counter_face"]
            tool = row["tool_tip_obstruction_mm3"] + row["tool_body_obstruction_mm3"]
            # ranked by how the sleeve can actually be pressed: counter-side spacer support (flat + integral wall), the room a spacer
            # has, the entry-side tool path; the old fixed-post anvil obstruction and coverage stay in the rows only
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=row["centre_mm"], unit="% spacer support (flat + wall; low is bad)",
                              value=cf["support_pct"], what=f"sleeve {row['index']}: flat {cf['flat_pct']} %, wall {cf['wall_pct']} %, recess {cf['recess_pct']} %, "
                                                            f"open {cf['open_pct']} %; spacer free depth {cf['free_depth_mm']} mm (axis {cf['axis_free_depth_mm']} mm)"))
            if tool > 0:
                cases.append(dict(part=n, mating_part=None, pose=None, location_mm=row["centre_mm"], unit="mm3 press tool path obstruction",
                                  value=round(tool, 3), what=f"sleeve {row['index']}: entry side"))
    cases.sort(key=lambda c: (c["unit"].startswith("mm3"), -c["value"] if c["unit"].startswith("mm3") else c["value"]))
    bad = dict(sleeves_with_support_below_90pct=bad["sleeves_with_support_below_90pct"], sleeves_with_tool_obstruction=bad["sleeves_with_tool_obstruction"],
               inserts=bad["inserts"], inserts_with_face_obstruction=bad["inserts_with_face_obstruction"],
               inserts_entry_face_flat_below_90pct=bad["inserts_entry_face_flat_below_90pct"],
               tubes_matching_no_definition_item=sum(1 for r in raw.values() for row in r["rows"] if str(row.get("kind_source", "")).startswith("no definition item within")),
               old_post_model_sleeves_with_anvil_obstruction=bad["sleeves_with_anvil_obstruction"],
               old_post_model_sleeves_with_coverage_below_90pct=bad["sleeves_with_counter_face_below_90pct"],
               parts_measured_from_own_mesh=sum(1 for r in raw.values() if "own closed mesh" in r["method"]), parts_not_measured=len(skipped))
    cases[:0] = [dict(part=n, mating_part=", ".join(ctx.rel.sleeve_sets_of(n)), pose=None, location_mm=None, unit="sleeves measured", value=0.0, what=f"not measured: {why}")
                 for n, why in skipped.items()]
    return dict(per_part=per_part, worst_cases=cases[:20], bad_events=bad, not_measured=skipped, evidence=[],
                notes="; ".join(sorted({r["method"] for r in raw.values()})))
