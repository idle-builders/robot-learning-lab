"""Old vs new FEM numbers for the migration record: python3 fem_compare.py NEW_REPORT.json PART OLD_JSON_PATH_SPEC
Prints p99 vM / peak vM / max displacement per load case. Old specs: 'rollflip:KEY' (artifacts/robot/body/leg/roll-flip/report.json fem.KEY)
or 'refit:KEY:r1|refit' (KEY shin, ankle_coupler, thigh, hip_pitch_to_roll, hip_yaw_to_pitch, pelvis_cradle) (motor-refit remote-evidence fem-PART.json)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
new = json.load(open(sys.argv[1]))["checks"]["fem"]["per_part"][sys.argv[2]]
spec = sys.argv[3].split(":")
if spec[0] == "rollflip":
    old = json.load(open(ROOT / "artifacts/robot/body/leg/roll-flip/report.json"))["fem"][spec[1]]["by_layer_normal"]
    old = {ln: {m: r for m, r in d.items()} for ln, d in old.items()}
else:
    d = json.load(open(ROOT / f"artifacts/robot/body/motor-refit/remote-evidence/remote-out/fem-{spec[1]}.json"))["parts"][spec[1]][spec[2]]
    old = {"xyz"[d["layer_normal_axis"]]: d["materials"]}
MAP = {"PAHT_CF_dry": "legacy_fdm_PAHT_CF_dry", "PAHT_CF_wet": "legacy_fdm_PAHT_CF_wet", "PETG_CF": "legacy_fdm_PETG_CF"}
print(f"nodes new {new['nodes']}  patches " + ", ".join(f"{k} {v['nodes']}n {v['area_mm2']}mm2" for k, v in new["patches"].items()))
worst = {}
for ln, bm in old.items():
    for om, rows in bm.items():
        for case, o in rows.items():
            n = new["by_layer_normal"][ln][MAP[om]][case]
            rel = lambda a, b: 100 * (b - a) / a if a else 0.0  # noqa: E731
            print(f"{ln} {om:11s} {case:18s} p99 {o['p99_vm_MPa']:8.3f} -> {n['p99_vm_MPa']:8.3f} ({rel(o['p99_vm_MPa'], n['p99_vm_MPa']):+5.1f}%)  "
                  f"peak {o['peak_vm_MPa']:8.2f} -> {n['peak_vm_MPa']:8.2f} ({rel(o['peak_vm_MPa'], n['peak_vm_MPa']):+5.1f}%)  "
                  f"u {o['max_displacement_mm']:.4f} -> {n['max_displacement_mm']:.4f} ({rel(o['max_displacement_mm'], n['max_displacement_mm']):+5.1f}%)  "
                  f"ratio {o['p99_strength_ratio']} -> {n['p99_strength_ratio']}")
