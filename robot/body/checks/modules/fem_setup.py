"""fem_setup: the FEM setups checked against the current parts (per patch: triangles, nodes, area, what each mating part contributes,
the part's own holes next to it) and coverage (printed parts in scope without a setup)."""
from __future__ import annotations

import numpy as np

from checks import fem_core as FC
from checks import process as P
from checks.core import R
from checks.registry import register


def holes(ctx, name, r_max=5.0):
    """[(radius, centre)] of the part's hole faces (fem_core.hole_faces)."""
    return [(h["r"], h["centre"]) for h in FC.hole_faces(ctx, name, r_max)]


@register("fem_setup", lambda ctx: bool(ctx.scope))
def run(ctx):
    """FEM setup vs the current part: patch selection (area, nodes, per mating part, holes) and printed parts in scope without a setup."""
    variant = getattr(ctx, "fem_variant", None)
    per, cases, uncovered = {}, [], []
    empty = overlap = 0
    hole_fallback = rounds_dropped = above_rmax = 0
    dirs = {}
    not_checked = {}
    for n in ctx.scope:
        s = FC.merged_setup(ctx, n, variant)
        if not s:
            if P.is_printed(ctx, n):           # every printed part, shells too: one without a setup is a coverage gap
                uncovered.append(n)
            continue
        if "native_step" not in ctx.parts[n]:
            not_checked[n] = "part has no native STEP in this manifest (mesh only)"
            continue
        dirs[n] = FC.layer_direction(ctx, n, variant)
        FC.progress_start(n, "fem_setup")
        mesh = FC.get_mesh(ctx, n, s)
        hs = holes(ctx, n)
        rows, masks = {}, {}
        released = {f"load_member_{i}_free_about_{x['free_rotation_about_joint']}": {k: v for k, v in x.items() if k != "free_rotation_about_joint"}
                    for i, x in enumerate(s["load"] if isinstance(s["load"], list) else [s["load"]]) if isinstance(x, dict) and x.get("free_rotation_about_joint")}
        for k, spec in {**{f"support_{a}": b for a, b in s["supports"].items()}, "load": s["load"], **released}.items():
            w, info, mask = FC.patch(ctx, mesh, spec)
            masks[k] = (w > 0, mask)
            tri_c = mesh["points"][mesh["tri"]].mean(axis=1)[mask]
            hit = sum(1 for r, c in hs if len(tri_c) and np.min(np.linalg.norm(tri_c - c, axis=1)) <= r + 1.0)
            info.update(holes_in_part=len(hs), holes_within_patch=hit,
                        mating_parts_without_contact=[m for m, v in info["by_mating_part"].items() if v["triangles"] == 0])
            rows[k] = info
            empty += info["nodes"] == 0
            cases.append(dict(part=n, mating_part=", ".join(info["mating_parts_without_contact"]) or None, pose=None, location_mm=info["centroid_mm"],
                              unit="mm2", value=info["area_mm2"], what=f"{k} patch area ({info['nodes']} nodes, {info['site_count']} site(s), {info['holes_within_patch']} of {len(hs)} holes)"))
        # a support site that is also a load site clamps its own load: shared nodes / triangles / area between each support patch and the load patch
        ln_, lt_ = masks["load"]
        cent, area = FC._cent_area(mesh)
        ov = {k[len("support_"):]: dict(shared_nodes=int((m[0] & ln_).sum()), shared_triangles=int((m[1] & lt_).sum()), shared_area_mm2=round(float(area[m[1] & lt_].sum()), 1))
              for k, m in masks.items() if k.startswith("support_")}
        n_ov = sum(1 for v in ov.values() if v["shared_nodes"])
        overlap += n_ov
        per[n] = dict(variant=variant, layer_direction=dirs[n], setup_notes=FC.setup_notes(s), patches=rows, support_load_overlap=ov, mesh_nodes=int(len(mesh["points"])), mesh_cached=mesh.get("cached"),
                      **({"mesh_retry": mesh["mesh_retry"]} if mesh.get("mesh_retry") else {}), **({"bore_collars": mesh["bore_collars"]} if mesh.get("bore_collars") else {}))
        for k, info in rows.items():
            for d in (info.get("hole_selection") or {}).values():
                hole_fallback += d["mode"].startswith("bores")
                rounds_dropped += d.get("dropped_faces", {}).get("round_or_partial_arc", 0)
                above_rmax += len(d.get("bore_above_r_max", []))
    cases.sort(key=lambda c: c["value"])
    # one layer direction for print setup and FEM: parts with a setup whose print direction is missing / seeded (unconfirmed) / not among the stored layer_normal_axes
    nodir = sorted(n for n, d in dirs.items() if d["up"] is None)
    seeded = sorted(n for n, d in dirs.items() if d["seeded"])
    disagree = {n: dict(print_direction=d["up"], stored_layer_normal_axes=d["stored"]) for n, d in sorted(dirs.items()) if d["disagrees"]}
    return dict(per_part=per, worst_cases=cases, bad_events=dict(empty_patches=empty, support_patches_sharing_nodes_with_load=overlap,
                                 hole_patch_mates_without_joining_screw=hole_fallback, round_or_partial_faces_left_out_of_hole_patches=rounds_dropped,
                                 screw_bores_wider_than_r_max_taken=above_rmax, parts_meshed_at_retry_h=sum(1 for p in per.values() if p.get("mesh_retry")), printed_parts_without_fem_setup=len(uncovered), setups_not_checked=len(not_checked),
                                 fem_parts_without_print_direction=len(nodir), fem_parts_with_seeded_print_direction=len(seeded), fem_parts_print_direction_disagrees_with_layer_normal_axes=len(disagree)),
                evidence=[], coverage=dict(printed_without_setup=uncovered, setup_not_checked=not_checked, print_direction_missing_fallback_to_stored_axes=nodir,
                                       print_direction_seeded=seeded, print_direction_disagrees=disagree),
                notes="worst-first = smallest patch; coverage.printed_without_setup lists printed parts (process.is_printed: definition kind, else role table) in scope with no fem setup. "
                + " | ".join(f"{n}: {p['setup_notes'].get('open_difference', '')}" for n, p in per.items() if p["setup_notes"].get("open_difference")))
