"""Coarse, datasheet-sized packaging screen for the sketch-first ape shell.

Run after sketch_shell.py. This is a static case-volume check, not a mechanism,
horn, cable, joint sweep, continuous-torque, or printability validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET

import build123d as bd

import sketch_shell as shell


@dataclass(frozen=True)
class Package:
    name: str
    kind: str
    source: str
    center_mm: tuple[float, float, float]
    size_mm: tuple[float, float, float]
    regions: tuple[str, ...]
    mass_g: float
    note: str


SOURCES = {
    "ST-3250-C001": "https://www.feetechrc.com/en/562636.html",
    "ST-3215-C018": "https://www.feetechrc.com/525603.html",
    "XC330-T288": "https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/",
    "Tattu-4S-850": "https://genstattu.com/tattu-850mah-14-8v-75c-4s1p-lipo-battery-pack-with-xt60-plug.html",
}


def packages() -> list[Package]:
    items = [Package(
        "battery_4s_850", "Tattu-4S-850", SOURCES["Tattu-4S-850"],
        (0, 0, 158), (60, 30, 30.5), ("torso_shell",), 107,
        "Physical cell pack only; 14.8 V nominal requires regulated servo buses.",
    )]
    for side, sign in (("left", 1), ("right", -1)):
        def add(joint: str, kind: str, center: tuple[float, float, float],
                size: tuple[float, float, float], regions: tuple[str, ...],
                mass: float, note: str) -> None:
            x, y, z = center
            items.append(Package(f"{side}_{joint}", kind, SOURCES[kind],
                                 (x, sign * y, z), size,
                                 tuple(f"{side}_{r}" if r != "torso_shell" else r
                                       for r in regions), mass, note))

        # Housing orientation is a first layout hypothesis. Output shafts,
        # mounts, wires, and a second bearing are deliberately not implied.
        add("hip_pitch", "ST-3250-C001", (-2, 37, 164),
            (35, 24.72, 45.22), ("thigh_shell", "torso_shell"), 74.5,
            "Higher rated effort; housing straddles pelvis/thigh boundary.")
        add("knee_pitch", "ST-3250-C001", (-1, 41, 113),
            (35, 24.72, 45.22), ("thigh_shell", "shin_shell"), 74.5,
            "Higher rated effort; needs an engineered knee transmission.")
        add("ankle_pitch", "ST-3215-C018", (-1, 41, 56),
            (35, 24.7, 45.2), ("shin_shell",), 55,
            "Housing sits in shin; the 33 mm foot cannot enclose it.")
        add("shoulder_roll", "XC330-T288", (-4, 58, 234),
            (26, 20, 34), ("torso_shell",), 23,
            "Small case in outer chest; crawl-load torque remains unproven.")
        add("shoulder_pitch", "ST-3215-C018", (-3, 88, 234),
            (35, 24.7, 45.2), ("upper_arm_shell",), 55,
            "Large case in upper arm; edge clearance is expected to be tight.")
        add("elbow_pitch", "XC330-T288", (2, 97, 188),
            (26, 20, 34), ("forearm_shell", "upper_arm_shell"), 23,
            "Small case at elbow; load capacity still needs a control study.")
    return items


def volume(shape: bd.Shape | bd.ShapeList | None) -> float:
    if shape is None:
        return 0.0
    return sum(part.volume for part in shape) if isinstance(shape, bd.ShapeList) else shape.volume


def main() -> None:
    shell.build()
    entries = []
    cases = packages()
    solids = []
    for item in cases:
        shape = bd.Pos(*item.center_mm) * bd.Box(*item.size_mm)
        solids.append(shape)
        interiors = [shell.INTERIORS[name] for name in item.regions]
        free = interiors[0]
        for region in interiors[1:]:
            free = free.fuse(region)
        contained = volume(shape.intersect(free)) / shape.volume
        shell_hits = {
            part.name: round(hit, 1)
            for part in shell.PARTS
            if (hit := volume(shape.intersect(part.shape))) > 1
        }
        hit_shell = sum(shell_hits.values())
        entries.append({
            **asdict(item),
            "inside_named_interiors_pct": round(100 * contained, 1),
            "shell_material_intersection_mm3": round(hit_shell, 1),
            "shell_hits_mm3": shell_hits,
        })
    clashes = []
    for i, a in enumerate(cases):
        for j in range(i + 1, len(cases)):
            overlap = volume(solids[i].intersect(solids[j]))
            if overlap > 0.01:
                clashes.append({"a": a.name, "b": cases[j].name,
                                "overlap_mm3": round(overlap, 1)})
    reference_model = shell.ROOT / "experiments/compact_morphology/ape_cad/ape_spaced_v8_casedynamics.xml"
    joints = [joint.get("name") for joint in ET.parse(reference_model).iter("joint")
              if joint.get("name")]
    screened = [p.name.replace("_knee_pitch", "_knee") for p in cases
                if p.kind != "Tattu-4S-850"]
    unscreened = sorted(set(joints) - set(screened))
    high_battery = bd.Pos(0, 0, 191) * bd.Box(60, 30, 30.5)
    high_battery_hits = {
        part.name: round(hit, 1) for part in shell.PARTS
        if (hit := volume(high_battery.intersect(part.shape))) > 1
    }
    retained_arm_case_trials = []
    for side, sign in (("left", 1), ("right", -1)):
        for joint, center, regions in (
            ("shoulder_roll", (-4, sign * 58, 234), ("torso_shell",)),
            ("elbow_pitch", (2, sign * 97, 188),
             (f"{side}_forearm_shell", f"{side}_upper_arm_shell")),
        ):
            large = bd.Pos(*center) * bd.Box(35, 24.7, 45.2)
            free = shell.INTERIORS[regions[0]]
            for name in regions[1:]:
                free = free.fuse(shell.INTERIORS[name])
            retained_arm_case_trials.append({
                "joint": f"{side}_{joint}",
                "candidate": "ST-3215-C018, as in the retained 22-axis physics ledger",
                "inside_named_interiors_pct": round(100 * volume(large.intersect(free)) / large.volume, 1),
                "shell_intersection_mm3": round(sum(
                    volume(large.intersect(part.shape)) for part in shell.PARTS), 1),
            })
    manifest = json.loads((shell.OUT / "manifest.json").read_text())
    report = {
        "shell_manifest": "manifest.json",
        "shell_height_mm": manifest["height_mm"],
        "design_stage": "static datasheet case-envelope screen",
        "method": "Exact BRep intersection of axis-aligned datasheet boxes with the CAD's 2 mm inset shell interiors; named neighboring interiors are fused at a joint.",
        "threshold_note": "A 100% case-volume result alone does not establish functional fit: shafts, horns, fasteners, wire bends, heat, structural members, and moving clearances are absent.",
        "packages": entries,
        "reference_model": str(reference_model.relative_to(shell.ROOT)),
        "reference_joint_count": len(joints),
        "screened_joints": screened,
        "unscreened_joints": unscreened,
        "high_battery_trial": {"center_mm": [0, 0, 191],
                               "shell_hits_mm3": high_battery_hits,
                               "note": "Same 4S case, raised 33 mm; center of mass rises with it."},
        "retained_arm_case_trials": retained_arm_case_trials,
        "case_clashes": clashes,
        "candidate_motor_mass_g": sum(p.mass_g for p in cases if p.kind != "Tattu-4S-850"),
        "candidate_battery_mass_g": 107,
        "nominal_shell_volume_cm3": round(sum(p["volume_cm3"] for p in manifest["parts"]), 2),
        "nominal_pla_shell_mass_g_at_1_24gcc": round(1.24 * sum(p["volume_cm3"] for p in manifest["parts"]), 1),
    }
    out = shell.OUT / "packaging-screen.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(cases)} static packages, {len(clashes)} case clashes -> {out}")
    for entry in entries:
        print(f"  {entry['name']}: {entry['inside_named_interiors_pct']}% in nominal interior; "
              f"{entry['shell_material_intersection_mm3']} mm³ shell interference")


if __name__ == "__main__":
    main()
