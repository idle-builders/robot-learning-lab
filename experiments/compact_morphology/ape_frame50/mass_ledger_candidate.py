"""Gross-stock mass hypothesis for the exposed 50 cm assembly.

Printed stock uses published Bambu PETG HF density. Shafts and unselected
bearing envelopes use a 7.85 g/cm3 steel screening density; belt volumes use
1.2 g/cm3. The selected JTEKT F606 ZZ hip-yaw bearings, cases, pack and cameras
use source catalog masses. Other stock is not a measured print or purchase.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import candidate
from experiments.compact_morphology.ape_frame50.shin_ankle_package import package, foot
from experiments.compact_morphology.ape_frame50.upper_frame_candidate import build

OUT = Path(__file__).with_name("mass_ledger_candidate.json")
DENSITY_KG_PER_MM3 = {"PETG_HF_solid": 1.28e-6,
                      "steel_envelope": 7.85e-6, "elastomer_belt_envelope": 1.2e-6,
                      "aluminum_pulley_envelope": 2.7e-6}


def material(name):
    if "belt" in name.lower(): return "elastomer_belt_envelope"
    if name in ("roll_drive_20T", "roll_driven_20T"):
        return "aluminum_pulley_envelope"
    if name == "yaw_motor_output" or any(v in name.lower() for v in (
            "shaft", "spindle", "bearing", "spacer", "washer", "screw")):
        return "steel_envelope"
    return "PETG_HF_solid"


def source_case(name):
    if "ST3250_case" in name: return ("Feetech ST3250", .0745)
    if "ST3215_case" in name or name == "motor_case": return ("Feetech ST3215", .055)
    if "XL330_case" in name: return ("ROBOTIS XL330", .018)
    if "battery_4s850" in name: return ("Tattu 4S850", .107)
    if "camera_module" in name: return ("Raspberry Pi Camera Module 3 Wide", .004)
    if "imu_reserve" in name: return ("Pololu MinIMU-9 v6: 0.7 g catalog board plus 4.3 g provisional mount/header/harness", .005)
    return None


def sourced_stock(name):
    if name in ("yaw_bearing_low_F606_ZZ", "yaw_bearing_high_F606_ZZ",
                "yaw_motor_bearing_low_F606_ZZ", "yaw_motor_bearing_high_F606_ZZ"):
        return ("JTEKT F606 ZZ catalog mass", .0063)
    if name in ("yaw_axial_DIN471_6_ring", "yaw_motor_axial_DIN471_6_ring"):
        return ("Beneri DIN 471-6 catalog mass", .000125)
    if name in ("roll_clamping_coupler_4001", "yaw_motor_coupler_4001"):
        return ("goBILDA 4001-0025-0006 catalog mass; ST3250 spline mating unverified", .006)
    return None


def transform(v): return [-v[1]/1000,v[0]/1000,v[2]/1000]


def stage(group):
    rows=[]
    for name, shape in group.items():
        case=source_case(name)
        part=sourced_stock(name)
        material_name=material(name) if case is None and part is None else None
        mass=(float(shape.volume)*DENSITY_KG_PER_MM3[material_name]
              if material_name else (case or part)[1])
        c=shape.center()
        rows.append({"name":name,"source_or_material":(case or part)[0] if case or part else material_name,
                     "mass_kg":round(mass,6),"center_left_m":transform((c.X,c.Y,c.Z)),
                     "volume_mm3":round(float(shape.volume),2),
                     "source_mass":bool(part or case and case[0]!="IMU unselected reserve")})
    frame=[r for r in rows if not source_case(r["name"])]
    total=sum(r["mass_kg"] for r in frame)
    centroid=(np.average([r["center_left_m"] for r in frame],axis=0,
                         weights=[r["mass_kg"] for r in frame]).tolist() if total else [0.,0.,0.])
    return {"frame_mass_kg":round(total,6),"frame_center_left_m":centroid,
            "source_components_kg":round(sum(r["mass_kg"] for r in rows if source_case(r["name"])),6),
            "parts":rows}


def main():
    hw,shared=candidate()
    thigh,shin,ankle=package()
    upper=build()
    groups={"pelvis_shared":shared,
            **{f"leg_hip_{key}":value for key,value in hw.items()},
            "leg_thigh":thigh,"leg_shin":shin,"leg_ankle_fixed":ankle,
            "leg_foot":foot(0),
            **{f"upper_{key}":value for key,value in upper.items()}}
    report={"stage":"gross-stock mass and source-component candidate, not measured body",
            "densities_kg_per_mm3":DENSITY_KG_PER_MM3,
            "PETG_density_source":"https://us.store.bambulab.com/collections/bambu-lab-3d-printer-filament/products/petg-hf?from=home_web",
            "JTEKT_F606_ZZ_mass_source":"https://koyo.jtekt.co.jp/en/products/detail/print.php?pno=F606+ZZ",
            "DIN471_6_mass_source":"https://www.beneri.com/en/prodotto/din-471",
            "roll_coupler_source":"https://www.gobilda.com/4001-series-clamping-servo-to-shaft-coupler-25-tooth-spline-to-6mm-round-bore/",
            "roll_pulley_source":"https://uk.misumi-ec.com/pdf/fa/2014/P1_1383-1384_F19_EN.pdf",
            "source_component_ledger":"experiments/compact_morphology/ape_frame50/frame_contract.yaml",
            "caveats":["Overlapping stock volumes are summed; no print infill or wall plan",
                       "Unselected bearing rings, shafts and belt envelopes use generic material densities, not purchased part weights",
                       "Roll coupler uses 6 g source mass; actual spline fit to ST3250 is untested",
                       "MISUMI pulley mass is a gross 2.7 g/cm3 aluminum envelope from catalog dimensions, not a purchased-part weight",
                       "No wiring, compute, regulators, fasteners, connectors, adapters or cable-service mass",
                       "No compliance, thermal, speed or output-duty measurement"],
            "stages":{name:stage(value) for name,value in groups.items()}}
    # The torso and fixed shoulder are one body; the two left/right limb
    # assemblies contribute symmetrically except source camera placement.
    m={name:value["frame_mass_kg"] for name,value in report["stages"].items()}
    frame=(m["pelvis_shared"]+2*sum(m[f"leg_hip_{s}"] for s in ("fixed","yaw","roll","thigh"))+
           2*sum(m[s] for s in ("leg_thigh","leg_shin","leg_ankle_fixed","leg_foot"))+
           m["upper_torso"]+m["upper_head"]+
           2*sum(m[f"upper_{s}"] for s in ("fixed","yaw","pitch","arm","forearm")))
    # Selected cases: eight ST3250, ten ST3215 (two ankles and eight arms),
    # three XL330; 4S pack; two cameras; one selected IMU board with a
    # provisional mount/header/harness allowance inside its 5 g budget.
    catalog=8*.0745+10*.055+3*.018+.107+2*.004+.005
    report["summary"]={"gross_frame_and_drive_stock_kg":round(frame,4),
                       "selected_motor_battery_camera_and_imu_reserve_kg":round(catalog,4),
                       "gross_total_before_unselected_electronics_and_fasteners_kg":round(frame+catalog,4),
                       "gap_to_old_3p16kg_allocation_kg":round(3.16-frame-catalog,4)}
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["summary"],indent=2))


if __name__=="__main__": main()
