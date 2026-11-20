"""Old vs new FEM numbers for the stage-based studies (neck-head / arm / torso): python3 fem_compare_stages.py NEW_REPORT.json OLD_JSON
OLD_JSON = {stage: evaluate_stage_solid result of the old script rerun on today's STEP} (scratch; neck-head layout). Percent = new vs old,
legacy PAHT-CF dry / wet / PETG-CF, all load cases; the range is printed per part."""
import json
import sys

new = json.load(open(sys.argv[1]))["checks"]["fem"]["per_part"]
old = json.load(open(sys.argv[2]))
PART = {"stage1_neck_root_cradle": "combined__neck_root_cradle", "stage2_neck_yaw_to_pitch_link": "combined__neck_yaw_to_pitch_link",
        "stage3_neck_pitch_to_head_cradle": "combined__neck_pitch_to_head_cradle", "stage4_head_sensor_compute_carrier": "combined__head_sensor_compute_carrier"}
MAP = {"PAHT_CF_dry": "legacy_fdm_PAHT_CF_dry", "PAHT_CF_wet": "legacy_fdm_PAHT_CF_wet", "PETG_CF": "legacy_fdm_PETG_CF"}
rel = lambda a, b: 100 * (b - a) / a if a else 0.0  # noqa: E731
for st, o in old.items():
    n = new.get(PART[st])
    if not n:
        continue
    d = {"p99": [], "peak": [], "disp": []}
    ln = next(iter(n["by_layer_normal"]))
    for om, mo in o["materials"].items():
        for case, oc in mo["load_cases"].items():
            nc = n["by_layer_normal"][ln].get(MAP[om], {}).get(case)
            if nc:
                d["p99"].append(rel(oc["p99_von_mises_MPa"], nc["p99_vm_MPa"])); d["peak"].append(rel(oc["max_von_mises_MPa"], nc["peak_vm_MPa"]))
                d["disp"].append(rel(oc["max_displacement_mm"], nc["max_displacement_mm"]))
    rg = lambda v: f"{min(v):+.1f}..{max(v):+.1f} %" if v else "n/a"  # noqa: E731
    print(f"{PART[st]}: nodes new {n['nodes']} (old {o['mesh_nodes']}); patch nodes " + ", ".join(f"{k} {v['nodes']}" for k, v in n["patches"].items())
          + f" (old support {o['fixed_patch_nodes']}, load {o['load_patch_nodes']}); p99 {rg(d['p99'])}, peak {rg(d['peak'])}, disp {rg(d['disp'])}")
