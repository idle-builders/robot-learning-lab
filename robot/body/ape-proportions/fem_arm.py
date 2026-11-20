"""Static FEM of the lengthened arm connectors against the same connectors before (T-gqwdrn).

Reuses the arm study's solver (robot/body/arm/fem_evaluate.py): P1 tetrahedra, orthotropic PAHT-CF dry and PETG-CF,
minimum-weight distribution of a 6-component wrench over the load patch, the support patch clamped, and its four arm
load cases (manipulation reach, 4 kg support, 4 kg at 1.5 g recovery, 6 kg). Patches come from the parts that actually
mate in the assembly manifest: connector surface within GAP_MM of the mating horn or case surface.

  Stage 3 (upper arm): clamped at the shoulder-yaw horn, loaded at the elbow horn.
  Stage 4 (forearm):   clamped at the elbow servo case, loaded at the wrist (palm).

The longer forearm also moves the hand load further from the elbow, so for stage 3 'after' the wrench gains the moment
of its force about the extra forearm length (r = (0, 0, -forearm growth)). Steel sleeves are not modelled in either
variant (same simplification both ways); the comparison is before against after, not an absolute rating.
Continuous values only.

Static FEM may run locally under infra/tools/memguard.py, one job at a time (owner, 2026-11-12).

  python3 infra/tools/memguard.py -- .venv/bin/python -B robot/body/ape-proportions/fem_arm.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "robot/body/arm"))
import fem_evaluate as FE  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("refit_fit_check", HERE.parent / "motor-refit" / "fit_check.py")
F = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F)  # the arm study has its own fit_check

REFIT = ROOT / "artifacts/robot/body/motor-refit/assembly-parts.json"
APE = ROOT / "artifacts/robot/body/ape-proportions/assembly-parts.json"
GAP_MM = 0.8
STAGES = {
    "stage3_upper_arm_yaw_to_elbow": dict(part="R_upper_arm_yaw_to_elbow_combined", support=["upper__R_yaw_C047_main_horn"],
                                          load=["upper__R_elbow_C047_main_horn"]),
    "stage4_forearm_elbow_to_wrist": dict(part="R_forearm_elbow_to_wrist_combined", support=["upper__R_elbow_C047_case"],
                                          load=["R_palm_upper", "R_palm_lower"]),
}


def tet_mesh(step, h_max=2.0):
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("s"); gmsh.model.occ.importShapes(str(step), highestDimOnly=True); gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_max / 5); gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10); gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        idx = np.zeros(int(tags.max()) + 1, np.int64); idx[tags.astype(np.int64)] = np.arange(len(tags))
        pts = coords.reshape(-1, 3)
        tet = np.vstack([idx[n.astype(np.int64)].reshape(-1, 4) for t, _, n in zip(*gmsh.model.mesh.getElements(3)) if t == 4])
    finally:
        gmsh.finalize()
    v = pts[tet]; vol6 = np.abs(np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0]))
    edge = np.max([np.linalg.norm(v[:, a] - v[:, b], axis=1) for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], axis=0)
    tet = tet[vol6 / 6 > 1e-3 * edge ** 3]  # drop slivers (quality below 1e-3 of the edge cube)
    used = np.unique(tet); remap = -np.ones(len(pts), np.int64); remap[used] = np.arange(len(used))
    pts, tet = pts[used], remap[tet]  # and the nodes only they used
    faces = np.sort(np.vstack([tet[:, [0, 1, 2]], tet[:, [0, 1, 3]], tet[:, [0, 2, 3]], tet[:, [1, 2, 3]]]), axis=1)
    uniq, cnt = np.unique(faces, axis=0, return_counts=True)
    return pts, tet, uniq[cnt == 1]


def patch_weights(pts, tri, man, names):
    P = {p["name"]: p for p in man["parts"]}
    targets = []
    for n in names:
        m = F.load_mesh(P[n]["mesh"])
        if P[n].get("placement_offset_mm"):
            m.apply_translation(P[n]["placement_offset_mm"])
        targets.append(np.vstack([m.vertices, m.sample(20000)]))
    tree = cKDTree(np.vstack(targets))
    cent = pts[tri].mean(1)
    gap = GAP_MM
    while True:
        mask = tree.query(cent, k=1)[0] <= gap
        if mask.sum() >= 30 or gap > 3.0:
            break
        gap += 0.4
    v = pts[tri]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    w = np.bincount(tri[mask].ravel(), weights=np.repeat(area[mask] / 3, 3), minlength=len(pts))
    return w, float(gap), float(area[mask].sum())


def stage_spec(stage, man):
    """Patches follow the elbow servo part on the connector's own link (horn before the flip, case after)."""
    link = {p["name"]: p["owning_link"] for p in man["parts"]}
    elbow = [n for n in ("upper__R_elbow_C047_main_horn", "upper__R_elbow_C047_case", "upper__R_elbow_C047_rear_idler") if n in link]
    spec = dict(STAGES[stage])
    if stage.startswith("stage3"):
        spec["load"] = [n for n in elbow if link[n] == "R_upper_arm"]
    else:
        spec["support"] = [n for n in elbow if link[n] == "R_forearm"]
    return spec


def run(stage, man, step, extra_lever, materials=("PAHT_CF_dry", "PETG_CF")):
    spec = stage_spec(stage, man)
    pts, tet, tri = tet_mesh(step)
    w_sup, g_sup, a_sup = patch_weights(pts, tri, man, spec["support"])
    w_load, g_load, a_load = patch_weights(pts, tri, man, spec["load"])
    origin = np.average(pts, axis=0, weights=w_load)
    unit = np.column_stack([FE.ELAST.minimum_weight_wrench(pts, w_load, np.eye(6)[i], origin)[0].ravel() for i in range(6)])
    fixed = (3 * np.flatnonzero(w_sup > 0)[:, None] + np.arange(3)).ravel()
    free = np.setdiff1d(np.arange(3 * len(pts)), fixed)
    rows = []
    for mat in materials:
        D = FE.build_material_hooke(mat, 0)
        K, vol, B, _ = FE.ELAST.assemble_materials(pts, tet, [D, D, D], np.ones(len(tet), int))
        ub = np.zeros_like(unit); ub[free] = spsolve(K[free][:, free].tocsc(), unit[free])
        for case, w in FE.ARM_LOAD_WRENCHES[stage].items():
            co = np.asarray(w, float).copy()
            co[3:] += np.cross(np.array([0.0, 0.0, -extra_lever]), co[:3])
            u = (ub @ co).reshape(-1, 3)
            st = np.zeros((len(tet), 6))
            for i in range(6):
                st += co[i] * np.einsum("eab,eb->ea", np.broadcast_to(D, (len(tet), 6, 6)),
                                        np.einsum("eij,ej->ei", B, ub[:, i].reshape(-1, 3)[tet].reshape(-1, 12)))
            vm = FE.VON_MISES(st)
            q = FE.volume_quantiles(vm, vol)
            lt = FE.volume_quantiles(np.maximum(st[:, 0], 0), vol)
            info = FE.MATERIALS[mat]
            ref_s = info.get("reference_strength_MPa", info.get("tensile_XY_MPa", 92.0))
            ref_l = info.get("tensile_Z_layer_MPa", ref_s)
            rows.append(dict(material=mat, case=case, wrench=co.round(2).tolist(), max_displacement_mm=round(float(np.linalg.norm(u, axis=1).max()), 4),
                             von_mises_p99_MPa=round(q["p99"], 2), layer_tension_p99_MPa=round(lt["p99"], 2),
                             p99_strength_ratio=round(ref_s / max(q["p99"], 1e-6), 2), p99_layer_ratio=round(ref_l / max(lt["p99"], 1e-6), 2)))
    return dict(nodes=len(pts), tets=len(tet), volume_mm3=round(float(vol.sum()), 1), support_patch_mm2=round(a_sup, 1),
                support_gap_mm=g_sup, load_patch_mm2=round(a_load, 1), load_gap_mm=g_load, extra_lever_mm=extra_lever, rows=rows)


def main():
    out = Path(sys.argv[1])
    refit, ape = json.load(open(F.src(REFIT))), json.load(open(APE))
    df = (ape["ape_proportions"]["geometry"]["R_arm"]["forearm_elbow_to_hand_tip_mm"][1]
          - ape["ape_proportions"]["geometry"]["R_arm"]["forearm_elbow_to_hand_tip_mm"][0])
    report = dict(schema="cove50.ape-arm-fem.v1", method=__doc__.split("\n\n")[1].replace("\n", " "), stages={})
    for stage, spec in STAGES.items():
        t0 = time.time()
        res = {}
        for label, man in (("before", refit), ("after", ape)):
            p = next(x for x in man["parts"] if x["name"] == spec["part"])
            step = F.src(p["native_step"])
            if p.get("native_step_offset_mm"):
                raise SystemExit(f"{p['name']}: STEP offset not handled here")
            lever = df if (label == "after" and stage.startswith("stage3")) else 0.0
            res[label] = run(stage, man, step, lever)
        report["stages"][stage] = res
        out.parent.mkdir(parents=True, exist_ok=True); json.dump(report, open(out, "w"), indent=1)
        for case in ("support_4kg_1g", "recovery_4kg_1p5g"):
            b = next(r for r in res["before"]["rows"] if r["case"] == case and r["material"] == "PAHT_CF_dry")
            a = next(r for r in res["after"]["rows"] if r["case"] == case and r["material"] == "PAHT_CF_dry")
            print(f"{stage} {case}: displacement {b['max_displacement_mm']} -> {a['max_displacement_mm']} mm, "
                  f"p99 {b['von_mises_p99_MPa']} -> {a['von_mises_p99_MPa']} MPa, "
                  f"strength ratio {b['p99_strength_ratio']} -> {a['p99_strength_ratio']}, "
                  f"layer ratio {b['p99_layer_ratio']} -> {a['p99_layer_ratio']}  ({time.time() - t0:.0f} s)", flush=True)
        print(f"   patch parts after: {stage_spec(stage, ape)}", flush=True)
        print(f"   patches before sup/load {res['before']['support_patch_mm2']}/{res['before']['load_patch_mm2']} mm2, "
              f"after {res['after']['support_patch_mm2']}/{res['after']['load_patch_mm2']} mm2", flush=True)


if __name__ == "__main__":
    main()
