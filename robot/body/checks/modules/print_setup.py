"""print_setup: the resolved print setup of every scoped printed part (printer, material, nozzle, layer height, line width,
print direction) with where each value comes from, and coverage counts: printed parts the printability checks cannot run
on (no print direction, no material, material or printer not in the tables, no native STEP) and values that are seeded
(guessed from old free text or FEM axes, not yet confirmed by the part's designer). Settings only: no solids, under a second."""
from __future__ import annotations

from checks import process as P
from checks.registry import register


def _printed(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n)]


@register("print_setup", lambda ctx: bool(_printed(ctx)))
def run(ctx):
    """Print setup per printed part (printer, material, nozzle, layer height, line width, up direction; source of each value) and coverage counts."""
    per, cases = {}, []
    ev = dict(printed_parts=0, without_print_direction=0, without_material=0, material_not_in_table=0, printer_not_in_table=0,
              nozzle_not_offered_by_printer=0, without_native_step=0, direction_seeded=0, material_seeded=0, run_overrides=0)
    for n in _printed(ctx):
        p = ctx.parts[n]
        e = P.effective(ctx, n)
        ev["printed_parts"] += 1
        miss = []
        if e["up"] is None:
            ev["without_print_direction"] += 1; miss.append("print direction")
        if e["material"] is None:
            ev["without_material"] += 1; miss.append("material")
        elif not e["material_known"]:
            ev["material_not_in_table"] += 1; miss.append(f"material {e['material']!r} not in materials.json")
        if e["printer"] and not e["printer_known"]:
            ev["printer_not_in_table"] += 1; miss.append(f"printer {e['printer']!r} not in printers.json")
        if not e["nozzle_offered"]:
            ev["nozzle_not_offered_by_printer"] += 1; miss.append(f"nozzle {e['nozzle_mm']} mm not offered by {e['printer']}")
        if not p.get("native_step"):
            ev["without_native_step"] += 1; miss.append("native STEP")
        ev["direction_seeded"] += e["source"]["up"] == "seeded"
        ev["material_seeded"] += e["source"]["material"] == "seeded"
        ev["run_overrides"] += any(v == "run override" for v in e["source"].values())
        per[n] = dict(role=p.get("role"), **{k: e[k] for k in ("printer", "material", "nozzle_mm", "layer_height_mm", "line_width_mm",
                                                                  "up", "plate_yaw_deg", "material_alternatives")},
                      source=e["source"], seeded=e["seeded"], missing=miss, native_step=bool(p.get("native_step")),
                      manifest_material_text=p.get("material"), manifest_print_orientation=p.get("print_orientation"))
        if miss:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=float(len(miss)), unit="missing inputs",
                              what="printability checks cannot run: no " + ", ".join(miss)))
    cases.sort(key=lambda c: -c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes="printed = process.printed, else the kind the part's body definition declares, else its role's kind in "
                      "data/roles.json (process.is_printed). Values resolve run override > "
                      "part setting > material default (nozzle) > printer default by nozzle (layer height, line width); "
                      "'seeded' values were derived from old free text / FEM axes and await the designer's confirmation "
                      "(remove `seeded` from the part's process when confirmed).")
