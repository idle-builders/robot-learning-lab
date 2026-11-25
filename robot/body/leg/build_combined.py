"""Assemble and package the Combined Lower-Leg FDM Connector Architecture (`cove50-lower-leg-fdm-combined`).

Stage selection (the selected body's choice, fixed here; it was made from the per-study FEM comparison of 2026-11-02, whose
evaluator is retired: robot/body/checks/run.py --checks fem runs the per-part setups instead; the old numbers are in git history):
- Stage 1 (Shin): Family 1 (`truss_box`)
- Stage 2 (Ankle Coupler): Family 1 (`truss_box`)
- Stage 3 (Foot Monocoque): Family 2 (`topopt`)
"""
from pathlib import Path
import copy
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STUDY, ART, read_json, write_json, rel, sha
from package_and_render import build_family_packages


def build_combined():
    tb_manifest = read_json(ART / "truss_box" / "cad-manifest.json")
    to_manifest = read_json(ART / "topopt" / "cad-manifest.json")

    comb_dir = ART / "combined"
    parts_dir = comb_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    tb_by_stage = {(p["stage"], p["side"]): p for p in tb_manifest["parts"]}
    to_by_stage = {(p["stage"], p["side"]): p for p in to_manifest["parts"]}

    selection = [
        ("stage1_shin", "R", tb_by_stage[("stage1_shin", "R")], "R_shin_combined"),
        ("stage1_shin", "L", tb_by_stage[("stage1_shin", "L")], "L_shin_combined"),
        ("stage2_ankle_coupler", "R", tb_by_stage[("stage2_ankle_coupler", "R")], "R_ankle_coupler_combined"),
        ("stage2_ankle_coupler", "L", tb_by_stage[("stage2_ankle_coupler", "L")], "L_ankle_coupler_combined"),
        ("stage3_foot_monocoque", "R", to_by_stage[("stage3_foot_monocoque", "R")], "R_foot_monocoque_combined"),
        ("stage3_foot_monocoque", "L", to_by_stage[("stage3_foot_monocoque", "L")], "L_foot_monocoque_combined"),
    ]

    combined_parts = []
    for stage, side, src_entry, new_name in selection:
        entry = copy.deepcopy(src_entry)
        src_step = ROOT / src_entry["native_step"]
        src_stl = ROOT / src_entry["mesh"]
        dst_step = parts_dir / f"{new_name}.step"
        dst_stl = parts_dir / f"{new_name}.stl"
        shutil.copy2(src_step, dst_step)
        shutil.copy2(src_stl, dst_stl)

        entry["name"] = new_name
        entry["native_step"] = rel(dst_step)
        entry["native_step_sha256"] = sha(dst_step)
        entry["mesh"] = rel(dst_stl)
        entry["mesh_sha256"] = sha(dst_stl)
        entry["selected_from_family"] = "truss_box" if "truss_box" in src_entry["name"] else "topopt"

        # Copy sleeves
        sl = entry["sleeves"]
        sl_new_name = f"{new_name}_steel_sleeves"
        sl_src_step = ROOT / sl["native_step"]
        sl_src_stl = ROOT / sl["mesh"]
        sl_dst_step = parts_dir / f"{sl_new_name}.step"
        sl_dst_stl = parts_dir / f"{sl_new_name}.stl"
        shutil.copy2(sl_src_step, sl_dst_step)
        shutil.copy2(sl_src_stl, sl_dst_stl)
        sl["name"] = sl_new_name
        sl["native_step"] = rel(sl_dst_step)
        sl["native_step_sha256"] = sha(sl_dst_step)
        sl["mesh"] = rel(sl_dst_stl)
        sl["mesh_sha256"] = sha(sl_dst_stl)

        combined_parts.append(entry)

    manifest = {
        "family": "cove50-lower-leg-fdm-combined",
        "version": "r1",
        "description": (
            "Combined Optimal Lower-Leg FDM Connector Architecture: Stage 1 Shin (Truss-Box) + "
            "Stage 2 Ankle Coupler (Truss-Box) + Stage 3 Foot Monocoque (3D TopOpt) with 304 SS Compression Sleeves"
        ),
        "parts": combined_parts,
        "source_code": {
            "path": rel( Path(__file__).resolve() ),
            "sha256": sha(Path(__file__).resolve()),
        },
    }
    write_json(comb_dir / "cad-manifest.json", manifest)

    # The combined family's exact fit is measured by robot/body/checks/run.py (--checks fit), not gated here.

    # Package GLBs, STEP, parts.json, bom.json for combined
    build_family_packages()
    print("Packaged combined lower-leg family successfully.")


if __name__ == "__main__":
    build_combined()
