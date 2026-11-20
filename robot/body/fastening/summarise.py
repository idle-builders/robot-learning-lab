"""Per-connector summary of a mount-check report, in the owner's categories (2026-11-18): alignment, surface contact
(tightness), sleeves, thread, access. Counts of screws with each kind of deviation next to the worst measured value.

  .venv/bin/python -B robot/body/fastening/summarise.py [REPORT] [--md OUT.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / ".cache/work/mount-check/assembly-parts-roll-flip/report.json"
CATEGORY = {
    "no_terminal": "alignment", "terminal_off_axis_mm": "alignment", "terminal_tilt_deg": "alignment",
    "clamped_hole_off_axis_mm": "alignment", "sleeve_off_axis_mm": "alignment", "shank_in_solid_mm": "alignment",
    "clamped_hole_too_small_mm": "alignment",
    "head_seat": "contact", "head_seat_gap_mm": "contact", "layers_gap_mm": "contact",
    "layers_interpenetrate_mm": "contact", "sleeve_not_bored_mm": "sleeve",
    "clamped_printed_hole_without_sleeve": "sleeve", "head_bearing_on_sleeve_mm": "sleeve",
    "engagement_mm": "thread", "motor_part_without_definition": "thread", "tip_past_hole_bottom_mm": "thread", "thread_family": "thread",
    "insertion_path_blocked_pct": "access", "driver_path_blocked_pct": "access",
}
ORDER = ["alignment", "contact", "sleeve", "thread", "access"]


def reclassify(r, roles):
    """A printed part overlapping its own sleeve set is a sleeve bore that was never cut, not a clamp-stack error."""
    out = []
    for f in r["findings"]:
        if f["kind"] == "layers_interpenetrate_mm":
            a, b = f["note"].split(" / ")
            servo = {"motor", "actuator", "purchased_servo_case_family_reference", "output", "output_horn", "horn",
                     "purchased_servo_output_family_reference"}
            if roles.get(a) in servo and roles.get(b) in servo:
                continue                                    # horn spline on the servo shaft, modelled as overlap
            if a + "_steel_sleeves" == b or b + "_steel_sleeves" == a:
                out.append(dict(f, kind="sleeve_not_bored_mm"))
                continue
        out.append(f)
    return out


def connector_of(r, roles):
    pr = [c for c in r.get("clamped", []) if roles.get(c, "").startswith("printed_structural") or roles.get(c) == "body_shell"]
    if pr:
        return pr[0]
    t = r.get("terminal") or {}
    if t.get("part"):
        return t["part"]
    st = [c for c in r.get("stack", []) if roles.get(c, "").startswith("printed_structural")]
    return st[0] if st else "(nothing around the screw)"


def summarise(rep, roles):
    groups = {}
    for r in rep["rows"]:
        fs = reclassify(r, roles)
        key = connector_of(r, roles)
        g = groups.setdefault(key, dict(connector=key, role=roles.get(key), screws=0, cats={c: 0 for c in ORDER}, worst={}, kinds={}, shortfall=(0.0, None), offset=(0.0, None)))
        g["screws"] += 1
        hit = set()
        for f in fs:
            cat = CATEGORY.get(f["kind"], "other")
            hit.add(cat)
            g["kinds"][f["kind"]] = g["kinds"].get(f["kind"], 0) + 1
            v = f.get("value")
            if f["kind"] == "engagement_mm" and isinstance(v, (int, float)) and isinstance(f.get("reference"), (int, float)):
                g["shortfall"] = max(g["shortfall"], (round(f["reference"] - v, 3), r["name"]), key=lambda x: x[0])
            if f["kind"] in ("terminal_off_axis_mm", "clamped_hole_off_axis_mm") and isinstance(v, (int, float)):
                g["offset"] = max(g["offset"], (v, r["name"]), key=lambda x: x[0])
            if isinstance(v, (int, float)):
                w = g["worst"].get(f["kind"])
                if w is None or abs(v) > abs(w[0]):
                    g["worst"][f["kind"]] = (v, r["name"], f.get("note"))
        for c in hit:
            if c in g["cats"]:
                g["cats"][c] += 1
    return sorted(groups.values(), key=lambda g: (-g["shortfall"][0], -g["offset"][0], -g["screws"]))


def markdown(rep, roles):
    """Per-connector table, then every motor instance's registration fit (shape RMS; hole-centre RMS for horns/idlers)."""
    lines = ["| connector | worst engagement shortfall mm | worst offset mm | screws | alignment | contact | sleeve | thread | access | worst values |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in summarise(rep, roles):
        w = "; ".join(f"{k} {v[0]}" for k, v in sorted(g["worst"].items(), key=lambda kv: -abs(kv[1][0]))[:4])
        lines.append(f"| {g['connector']} | {g['shortfall'][0]} ({g['shortfall'][1]}) | {g['offset'][0]:.2f} ({g['offset'][1]}) | {g['screws']} | " + " | ".join(str(g['cats'][c]) for c in ORDER) + f" | {w} |")
    lines += ["", "| instance | standard | member | shape fit RMS mm | hole-centre fit RMS mm | holes located/defined | turn deg |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rep.get("instances", []), key=lambda r: -((r.get("hole_snap") or {}).get("fit_rms_mm") or 0)):
        hs = r.get("hole_snap") or {}
        lines.append(f"| {r['part']} | {r['standard']} | {r['member']} | {r['fit_rms_mm']} | {hs.get('fit_rms_mm', '')} | {hs.get('holes_located', '')}/{hs.get('holes_defined', '')} | {hs.get('turn_deg', '')} |")
    return lines


def write_summary(rep_path: Path, out: Path | None = None) -> Path:
    rep = json.loads(Path(rep_path).read_text())
    man = json.loads((ROOT / rep["manifest"]).read_text())
    out = out or Path(rep_path).parent / "summary.md"
    out.write_text("\n".join(markdown(rep, {p["name"]: p["role"] for p in man["parts"]})) + "\n")
    return out


def main():
    argv = sys.argv[1:]
    md = Path(argv[argv.index("--md") + 1]) if "--md" in argv else None
    args = [a for a in argv if not a.startswith("--") and (md is None or Path(a) != md)]
    rp = Path(args[0] if args else DEFAULT)
    out = write_summary(rp, md)
    print(out.read_text())
    if "--json" in sys.argv:
        rep = json.loads(rp.read_text()); man = json.loads((ROOT / rep["manifest"]).read_text())
        print(json.dumps(summarise(rep, {p["name"]: p["role"] for p in man["parts"]}), indent=1, default=str))


if __name__ == "__main__":
    main()
