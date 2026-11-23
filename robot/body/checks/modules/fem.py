"""fem: static FEM of every scoped part that has a per-part FEM setup (its body definition's `fem` block, else data/part_settings.json -> part -> fem). One mesher, one solver
(fem_core), run along the part's decided print direction (process.up; --fem-variant studied_axes: the stored layer_normal_axes). Per part, material and layer-normal axis: p99 / peak von Mises, p99 layer tension, strength ratios (in-plane and across layers), max displacement per load case."""
from __future__ import annotations

import json
import time

from checks import arm_loads as AL
from checks import fem_core as FC
from checks.registry import register


def ctx_variant(v):
    return v or "stored"


def _has(ctx):
    return any(FC.merged_setup(ctx, n, getattr(ctx, "fem_variant", None)) for n in ctx.scope)


@register("fem", _has)
def run(ctx):
    """Static FEM of scoped parts with a setup: p99 / peak von Mises, layer tension, strength ratio, displacement per load case."""
    variant = getattr(ctx, "fem_variant", None)
    per, cases, skipped = {}, [], {}
    for n in ctx.scope:
        s = FC.merged_setup(ctx, n, variant)
        if not s:
            continue
        mats, lns = FC.setup_materials(ctx, n, s), FC.layer_normals(ctx, n, variant)
        if "native_step" not in ctx.parts[n]:
            skipped[n] = "no native STEP in this manifest"
            continue
        if not mats or not lns:
            skipped[n] = "no material" if not mats else "no layer normal (no print direction, no layer_normal_axes)"
            continue
        ld = FC.layer_direction(ctx, n, variant)
        t0 = time.time()
        FC.progress_start(n)
        FC.say(f"setup {ctx_variant(variant)}: {len(mats)} material(s) x {len(lns)} layer normal(s) x {len(s['load_cases'])} load case(s), mesh h {float(s.get('mesh_h_mm', 2.2)):g}")
        slog = []
        mesh = FC.get_mesh(ctx, n, s)
        sup = {k: FC.patch(ctx, mesh, v) for k, v in s["supports"].items()}
        wl, linfo, _ = FC.patch(ctx, mesh, s["load"])
        where = f"{n} [{ctx_variant(variant)}]"
        for k, (w, info, _) in [("load", (wl, linfo, None))] + [(f"support {k}", v) for k, v in sup.items()]:
            if not (w > 0).any():      # an empty patch: no wrench can be applied / nothing is held; name it as fem_setup does
                raise ValueError(f"{where}: {k} patch selects no node of the part; patch: {json.dumps(s['load'] if k == 'load' else s['supports'][k[len('support '):]])}; "
                                 f"per mating part: {json.dumps(info['by_mating_part'])}" + (f"; hole selection: {json.dumps(info['hole_selection'])[:1500]}" if info.get("hole_selection") else ""))
        org = FC.origin_of(ctx, s, mesh, wl, where)
        release = FC.released_rotations(ctx, mesh, s["load"], f"{n} [{ctx_variant(variant)}]")
        FC.say(f"patches: load {linfo['nodes']} nodes, " + ", ".join(f"support {k} {i['nodes']} nodes" for k, (_, i, _) in sup.items())
               + "".join(f", released about {g['joint']} {len(g['nodes'])} nodes" for g in release or []))
        wr = {k: FC.wrench_of(ctx, c, org) for k, c in s["load_cases"].items() if "hand_envelope" not in c}
        env = {k: AL.envelope(ctx.body, c["hand_envelope"], org) for k, c in s["load_cases"].items() if "hand_envelope" in c}   # many wrenches per case
        if env:
            FC.say("hand envelopes: " + ", ".join(f"{k} {len(e[0])} wrenches" for k, e in env.items()))
        res = {}
        for ln in lns:
            res["xyz"[ln]] = {}
            for m in mats:
                FC.say(f"solve {len(slog) + 1}/{len(lns) * len(mats)}: {m}, layer normal {'xyz'[ln]}")
                res["xyz"[ln]][m] = FC.solve(mesh, [w for w, _, _ in sup.values()], wl, org, wr, m, ln, FC.slot_materials(s), rigid_load=FC.is_rigid(s["load"]), envelopes=env, release=release)
                slog.append(dict(layer_normal="xyz"[ln], material=m, **FC.SOLVE_LOG[-1]))
                lg = slog[-1]
                lg["residual_above_10x_rtol"] = bool(lg.get("true_relative_residual_max", 0.0) > 10 * FC.CG_RTOL)
                FC.say(f"solve {len(slog)}/{len(lns) * len(mats)} done: {m}, layer normal {'xyz'[ln]}")
        per[n] = dict(variant=variant, layer_direction=ld, nodes=int(len(mesh["points"])), tets=int(len(mesh["tet"])), origin_mm=org, setup_notes=FC.setup_notes(s),
                      wrenches_N_Nmm={k: v.tolist() for k, v in wr.items()}, envelope_loads={k: e[2] for k, e in env.items()}, released_rotations=[dict(joint=g["joint"], load_members=g["members"], nodes=int(len(g["nodes"]))) for g in release or []],
                      patches=dict(load=linfo, **{f"support_{k}": i for k, (_, i, _) in sup.items()}),
                      by_layer_normal=res, solver=slog, rigid_load=FC.is_rigid(s["load"]), reaction_plus_wrench_pct_max=max(x.get("reaction_plus_wrench_pct_max", 0.0) for x in slog), mesh_h_mm=float(s.get("mesh_h_mm", 2.2)), regions_elements=mesh.get("region_elements"), bore_collars=mesh.get("bore_collars"), seconds=round(time.time() - t0, 1),
                      **({"mesh_retry": mesh["mesh_retry"]} if mesh.get("mesh_retry") else {}))
        for ln, bm in res.items():
            for m, rows in bm.items():
                for c, r in rows.items():
                    cases.append(dict(part=n, mating_part=None, pose=f"{c} / {m} / layer normal {ln}", location_mm=r["peak_vm_location_mm"],
                                      unit="MPa", value=r["p99_vm_MPa"], what="p99 von Mises (volume)", strength_ratio=r["p99_strength_ratio"],
                                      p99_layer_tension_MPa=r["p99_layer_tension_MPa"], layer_strength_ratio=r["p99_layer_ratio"], layer_normal_source=per[n]["layer_direction"]["source"]))
    cases.sort(key=lambda c: -c["value"])
    bad = sum(1 for c in cases if c["strength_ratio"] is not None and c["strength_ratio"] < 1.0)
    return dict(per_part=per, worst_cases=cases, bad_events=dict(cases_with_p99_strength_ratio_below_1=bad, parts_with_setup_not_run=len(skipped),
                                                                 parts_meshed_at_retry_h=sum(1 for p in per.values() if p.get("mesh_retry")),
                                                                 collar_sleeves_material_assumed=sum(int(c.get("assumed_sleeves", 0)) for p in per.values() for c in (p.get("bore_collars") or {}).values()),
                                                                 cg_solves_with_residual_above_10x_rtol=sum(1 for p in per.values() for x in p["solver"] if x.get("residual_above_10x_rtol"))),
                evidence=[], notes=f"gmsh P1 tets, supports clamped, unit-wrench basis at the load patch; skipped: {skipped or 'none'}. "
                + " | ".join(f"{n} [{ctx_variant(variant)}]: {p['setup_notes'].get('open_difference', p['setup_notes'].get('_doc', ''))}" for n, p in per.items() if p["setup_notes"]))
