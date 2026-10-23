"""Exact shell screen for the local deep-chest relief and actuator scale ladder.

Run from the repository root with .venv/bin/python. Generates an honest
source-shell package report and a viewer payload under artifacts/.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from build123d import Compound, Plane, Solid, export_step

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "experiments" / "sketch_ape_shell"
OUT = ROOT / "artifacts" / "round2-skin-screen"
sys.path.insert(0, str(SOURCE))
from build import Study, loft  # noqa: E402

PLA_DENSITY = 1.24  # g/cm^3, same provisional appearance-shell basis as prior studies
XL_DIMS = (34.0, 26.0, 20.0)  # source-sized case orientation used for torso loom
XL_MASS_G = 18.0
XL_STALL_NM = 0.52
RSF_D_MM = 25.0
RSF_L_MM = 35.8
RSF_MASS_G = 77.0
RSF_CONT_NM = 0.44


class EnvelopeStudy(Study):
    """Retain exact un-hollowed torso/pelvis solids for containment checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_profiles = {}

    def hollow_loft(self, stations, crown=0):
        if len(stations) == 8 and stations[0][0] == 160:
            self.outer_profiles["torso"] = loft(stations).scale(self.s)
        elif len(stations) == 6 and stations[0][0] == 130:
            self.outer_profiles["pelvis"] = loft(stations).scale(self.s)
        return super().hollow_loft(stations, crown)


class DeepChestStudy(EnvelopeStudy):
    """One local front/back torso-and-pelvis depth change."""

    torso_profile = ((160, 30), (173, 30), (199, 30), (229, 26),
                     (253, 12), (264, 4), (274, 0), (278, 0))
    pelvis_profile = ((130, 0), (135, 0), (145, 8), (155, 20),
                      (164, 30), (178, 30))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._changed = False
        self.torso_outer_volume_mm3 = None
        self.pelvis_outer_volume_mm3 = None

    def hollow_loft(self, stations, crown=0):
        if len(stations) == 8 and stations[0][0] == 160:
            self._changed = True
            original = list(stations)
            zvals = np.array([p[0] for p in self.torso_profile], dtype=float)
            dvals = np.array([p[1] for p in self.torso_profile], dtype=float) / self.s
            changed = []
            for z, x, y, rx, ry in stations:
                delta = float(np.interp(z, zvals, dvals))
                changed.append((z, x, y, rx, ry + delta))
            # This solid measures the local outside-envelope volume change;
            # the actual shell still uses Study's normal interior subtraction.
            self.torso_outer_volume_mm3 = loft(changed).scale(self.s).volume
            self.base_torso_outer_volume_mm3 = loft(original).scale(self.s).volume
            self.base_torso_outer = loft(original).scale(self.s)
            return super().hollow_loft(changed, crown)
        if len(stations) == 6 and stations[0][0] == 130:
            original = list(stations)
            zvals = np.array([p[0] for p in self.pelvis_profile], dtype=float)
            dvals = np.array([p[1] for p in self.pelvis_profile], dtype=float) / self.s
            changed = []
            for z, x, y, rx, ry in stations:
                delta = float(np.interp(z, zvals, dvals))
                changed.append((z, x, y, rx, ry + delta))
            self.pelvis_outer_volume_mm3 = loft(changed).scale(self.s).volume
            self.base_pelvis_outer_volume_mm3 = loft(original).scale(self.s).volume
            self.base_pelvis_outer = loft(original).scale(self.s)
            return super().hollow_loft(changed, crown)
        return super().hollow_loft(stations, crown)


def box(center, dims):
    x, y, z = center
    dx, dy, dz = dims
    return Solid.make_box(dx, dy, dz,
        Plane(origin=(x - dx/2, y - dy/2, z - dz/2)))


def cyl_y(center, radius, length):
    x, y, z = center
    return Solid.make_cylinder(radius, length,
        Plane(origin=(x, y - length/2, z), z_dir=(0, 1, 0)))


def xl_motors(height):
    s = height / 350.0
    result = []
    for family, y0 in (("hip", -15.0), ("shoulder", 15.0)):
        for side, x0 in (("L", -18.0), ("R", 18.0)):
            for axis, z0 in zip(("yaw", "roll", "pitch"), (174.0, 196.0, 218.0)):
                result.append({"name": f"{family}_{side}_{axis}",
                    "shape": box((x0*s, y0*s, z0*s), XL_DIMS),
                    "center_mm": (x0*s, y0*s, z0*s), "dims_mm": XL_DIMS,
                    "type": "xl330"})
    return result


def rsf_motors(height):
    # The round module is an envelope comparison only. Its output/input datum,
    # mounting face and a cable route have not been mapped into this mechanism.
    s = height / 350.0
    result = []
    for family, y0 in (("hip", -15.0), ("shoulder", 15.0)):
        for side, x0 in (("L", -18.0), ("R", 18.0)):
            for axis, z0 in zip(("yaw", "roll", "pitch"), (174.0, 196.0, 218.0)):
                center = (x0*s, y0*s, z0*s)
                result.append({"name": f"{family}_{side}_{axis}",
                    "shape": cyl_y(center, RSF_D_MM/2, RSF_L_MM),
                    "center_mm": center, "dims_mm": (RSF_D_MM, RSF_L_MM),
                    "type": "rsf5b"})
    return result


def bbox(shape):
    b = shape.bounding_box()
    return np.array(tuple(b.min)), np.array(tuple(b.max))


def overlaps(a, b):
    aa, ab = bbox(a)
    ba, bb = bbox(b)
    return bool(np.all(aa < bb) and np.all(ba < ab))


def shell_index(study):
    rows = []
    for p in study.parts:
        if p["group"] not in {"torso", "pelvis"}:
            continue
        lo, hi = bbox(p["shape"])
        rows.append((p["name"], p["shape"], (lo, hi)))
    return rows


def package_checks(study, motors):
    wall = shell_index(study)
    report = []
    for m in motors:
        hits = []
        lo, hi = bbox(m["shape"])
        for name, shape, (slo, shi) in wall:
            if not (np.all(lo < shi) and np.all(slo < hi)):
                continue
            try:
                common = m["shape"] & shape
                if common is None:  # build123d returns None for a disjoint solid pair.
                    continue
                volume = common.volume
            except Exception as e:  # A boolean failure is uncertainty, not clearance.
                hits.append({"part": name, "boolean_error": str(e)[:180]})
                continue
            if volume > 0.01:
                hits.append({"part": name, "overlap_mm3": round(volume, 3)})
        report.append({"name": m["name"], "case_bounds_mm": [lo.tolist(), hi.tolist()],
                       "shell_intersections": hits})
    return report


def outer_envelope_checks(study, motors):
    envelope = study.outer_profiles["torso"].fuse(study.outer_profiles["pelvis"])
    rows = []
    for m in motors:
        try:
            outside = m["shape"] - envelope
            outside_volume = outside.volume if outside is not None else 0.0
            rows.append({"name": m["name"], "case_volume_outside_envelope_mm3": round(outside_volume, 3),
                         "contained_by_torso_pelvis_outer_envelope": outside_volume <= 0.01})
        except Exception as e:
            rows.append({"name": m["name"], "boolean_error": str(e)[:180]})
    return rows


def pair_collisions(motors):
    rows = []
    for i, a in enumerate(motors):
        for b in motors[i+1:]:
            if not overlaps(a["shape"], b["shape"]):
                continue
            common = a["shape"] & b["shape"]
            if common is None:
                continue
            v = common.volume
            if v > 0.01:
                rows.append({"a": a["name"], "b": b["name"], "overlap_mm3": round(v, 3)})
    return rows


def mesh_part(part):
    vertices, faces = part["shape"].tessellate(0.25, 0.2)
    return {"name": part["name"], "color": part["color"],
            "positions": np.asarray([tuple(v) for v in vertices], dtype=np.float32).ravel().tolist(),
            "indices": np.asarray(faces, dtype=np.uint32).ravel().tolist()}


def run_study(height, wall, deep=False):
    study = DeepChestStudy(height, wall) if deep else EnvelopeStudy(height, wall)
    with redirect_stdout(StringIO()):
        study.build()
    return study


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scales = {}
    studies350 = {}
    for h in (350, 400, 450, 500):
        study = run_study(h, 1.0)
        motors_xl = xl_motors(h)
        motors_rsf = rsf_motors(h)
        scales[str(h)] = {
            "height_mm": h,
            "shell_volume_mm3": round(sum(p["shape"].volume for p in study.parts), 2),
            "shell_mass_g_at_1_24": round(sum(p["shape"].volume for p in study.parts)/1000 * PLA_DENSITY, 2),
            "torso_bounds_mm": {p["name"]: [bbox(p["shape"])[0].tolist(), bbox(p["shape"])[1].tolist()]
                                for p in study.parts if p["name"] in ("torso_front", "torso_back")},
            "xl330_case_hits": package_checks(study, motors_xl),
            "rsf5b_envelope_hits": package_checks(study, motors_rsf),
            "xl330_outer_enclosure": outer_envelope_checks(study, motors_xl),
            "xl330_motor_motor_overlaps": pair_collisions(motors_xl),
            "rsf5b_motor_motor_overlaps": pair_collisions(motors_rsf),
        }
        if h == 350:
            studies350[1] = study
    # Nominal 2 mm comparison is limited to the target size and its local relief.
    base350_2 = run_study(350, 2.0)
    deep350_1 = run_study(350, 1.0, deep=True)
    deep350_2 = run_study(350, 2.0, deep=True)
    base_outer_union = deep350_1.base_torso_outer.fuse(deep350_1.base_pelvis_outer)
    deep_outer_union = deep350_1.outer_profiles["torso"].fuse(deep350_1.outer_profiles["pelvis"])
    export_step(Compound(label="350 mm sketch-06 with local proximal-depth relief",
                         children=[p["shape"] for p in deep350_1.parts]),
                OUT/"deep-proximal-shell-350mm-1mm.step")
    xl350 = xl_motors(350)
    rsf350 = rsf_motors(350)
    scales["350"]["xl330_case_hits_2mm"] = package_checks(base350_2, xl350)
    scales["350"]["rsf5b_envelope_hits_2mm"] = package_checks(base350_2, rsf350)
    scales["350"]["xl330_outer_enclosure_2mm"] = outer_envelope_checks(base350_2, xl350)
    scales["350"]["local_depth_change_1mm"] = {
        "profile_add_to_torso_y_radius_mm_at_350": list(DeepChestStudy.torso_profile),
        "profile_add_to_pelvis_y_radius_mm_at_350": list(DeepChestStudy.pelvis_profile),
        "base_shell_volume_mm3": round(sum(p["shape"].volume for p in studies350[1].parts), 2),
        "deep_shell_volume_mm3": round(sum(p["shape"].volume for p in deep350_1.parts), 2),
        "added_shell_material_mm3": round(sum(p["shape"].volume for p in deep350_1.parts)-sum(p["shape"].volume for p in studies350[1].parts), 2),
        "added_shell_mass_g_at_1_24": round((sum(p["shape"].volume for p in deep350_1.parts)-sum(p["shape"].volume for p in studies350[1].parts))/1000*PLA_DENSITY, 2),
        "outer_torso_envelope_added_mm3": round(deep350_1.torso_outer_volume_mm3-deep350_1.base_torso_outer_volume_mm3, 2),
        "outer_pelvis_envelope_added_mm3": round(deep350_1.pelvis_outer_volume_mm3-deep350_1.base_pelvis_outer_volume_mm3, 2),
        "outer_envelope_added_mm3": round(
            deep_outer_union.volume-base_outer_union.volume, 2),
        "base_outer_torso_pelvis_union_mm3": round(base_outer_union.volume, 2),
        "deep_outer_torso_pelvis_union_mm3": round(deep_outer_union.volume, 2),
        "base_shell_mass_g_at_1_24": scales["350"]["shell_mass_g_at_1_24"],
        "deep_shell_mass_g_at_1_24": round(sum(p["shape"].volume for p in deep350_1.parts)/1000*PLA_DENSITY, 2),
        "base_torso_bounds_mm": scales["350"]["torso_bounds_mm"],
        "deep_torso_bounds_mm": {p["name"]: [bbox(p["shape"])[0].tolist(), bbox(p["shape"])[1].tolist()]
                                 for p in deep350_1.parts if p["name"] in ("torso_front", "torso_back")},
        "xl330_case_hits": package_checks(deep350_1, xl350),
        "rsf5b_envelope_hits": package_checks(deep350_1, rsf350),
        "xl330_outer_enclosure": outer_envelope_checks(deep350_1, xl350),
        "xl330_case_hits_2mm": package_checks(deep350_2, xl350),
        "xl330_outer_enclosure_2mm": outer_envelope_checks(deep350_2, xl350),
        "nominal_wall_mm": 1.0,
        "wall_note": "The source hollow loft subtracts the same nominal radial wall at the changed stations; actual local normal thickness/minimum manufacturing wall is not certified.",
    }
    scales["350"]["local_depth_change_2mm"] = {
        "nominal_wall_mm": 2.0,
        "base_shell_mass_g_at_1_24": round(sum(p["shape"].volume for p in base350_2.parts)/1000*PLA_DENSITY, 2),
        "deep_shell_mass_g_at_1_24": round(sum(p["shape"].volume for p in deep350_2.parts)/1000*PLA_DENSITY, 2),
        "added_shell_material_mm3": round(sum(p["shape"].volume for p in deep350_2.parts)-sum(p["shape"].volume for p in base350_2.parts), 2),
        "added_shell_mass_g_at_1_24": round((sum(p["shape"].volume for p in deep350_2.parts)-sum(p["shape"].volume for p in base350_2.parts))/1000*PLA_DENSITY,2),
        "xl330_case_hits": package_checks(deep350_2, xl350),
        "rsf5b_envelope_hits": package_checks(deep350_2, rsf350),
    }
    scales["component_basis"] = {
        "xl330": {"official_source": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
                   "envelope_mm": [20,34,26], "mass_each_g": XL_MASS_G,
                   "torque_Nm": XL_STALL_NM, "torque_class": "stall at 5 V; continuous value not published"},
        "rsf5b": {"official_source": "https://www.harmonicdrive.net/_hd/content/documents1/RSA-8_RSF-5-IDT_Brochure.pdf",
                   "envelope": "Ø25 x 35.8 mm axial body", "mass_each_g": RSF_MASS_G,
                   "torque_Nm": RSF_CONT_NM, "torque_class": "allowable continuous; manufacturer operating curve uses 150 x 150 x 3 mm heatsink"},
        "counts": 12,
        "motor_mass_g": {"xl330_12": 12*XL_MASS_G, "rsf5b_12": 12*RSF_MASS_G},
        "torque_note": "No output bearing, transmission, motor/output datum or thermal design is modeled. XL330 torque remains stall only. RSF-5B provides manufacturer allowable-continuous evidence only under the specified heat-sink condition.",
    }
    (OUT/"report.json").write_text(json.dumps(scales, indent=2))
    source_mesh = [mesh_part(p) for p in studies350[1].parts]
    (OUT/"source-shell-mesh.json").write_text(json.dumps(
        {"height":350,"wall":1,"parts":source_mesh},separators=(",",":")))
    # The only mesh variant needed is 350 mm with unchanged outer shell vs
    # the local chest-depth proposal; other heights in the viewer scale the
    # exact source 350 mesh while the report retains exact B-rep tests per size.
    variant = [mesh_part(p) for p in deep350_1.parts if p["name"] in
               ("torso_front", "torso_back", "pelvis_front", "pelvis_back")]
    (OUT/"deep-proximal-shell-mesh.json").write_text(json.dumps({"height":350,"parts":variant},separators=(",",":")))
    shutil.copy2(Path(__file__).with_name("viewer.html"), OUT/"index.html")
    shutil.copy2(Path(__file__).with_name("compare.html"), OUT/"compare.html")
    vendor = ROOT/"artifacts"/"ape-wildcard-tendon-loom"/"vendor"
    if vendor.exists():
        shutil.copytree(vendor, OUT/"vendor", dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Three.js viewer vendor assets not found: {vendor}")
    print(f"Wrote {OUT/'report.json'}")
    for h in (350,400,450,500):
        row = scales[str(h)]
        print(h, "XL330", sum(bool(x["shell_intersections"]) for x in row["xl330_case_hits"]), "/12",
              "RSF-5B", sum(bool(x["shell_intersections"]) for x in row["rsf5b_envelope_hits"]), "/12",
              "case overlaps", len(row["xl330_motor_motor_overlaps"]), len(row["rsf5b_motor_motor_overlaps"]))
    local = scales["350"]["local_depth_change_1mm"]
    print("deep-350 XL330", sum(bool(x["shell_intersections"]) for x in local["xl330_case_hits"]), "/12",
          "RSF-5B", sum(bool(x["shell_intersections"]) for x in local["rsf5b_envelope_hits"]), "/12",
          "added shell mass g", local["added_shell_mass_g_at_1_24"])


if __name__ == "__main__":
    main()
