"""Make held-out Cove sensitivity bodies from the selected provisional dynamics.

These are evaluation perturbations, not alternative mechanical designs.  The
base body and the exact scale factors are kept in the artifact ledger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-transfer-20261008"
BASE = OUT / "cove50-direct-training.xml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = mujoco.MjModel.from_xml_path(str(BASE))
    cases = {
        "light-lowgrip": (0.85, 0.55),
        "heavy-lowgrip": (1.15, 0.55),
        "heavy-highgrip": (1.15, 1.25),
    }
    rows = []
    for name, (mass_scale, sole_friction) in cases.items():
        tree = ET.parse(BASE)
        root = tree.getroot()
        root.set("model", f"ape_cove50_direct_frame_{name}")
        count = 0
        for inertial in root.findall(".//inertial"):
            inertial.set("mass", f"{float(inertial.get('mass')) * mass_scale:.10g}")
            inertial.set("fullinertia", " ".join(
                f"{float(x) * mass_scale:.10g}"
                for x in inertial.get("fullinertia").split()))
            count += 1
        for side in ("left", "right"):
            geom = root.find(f".//geom[@name='{side}_sole']")
            if geom is None:
                raise ValueError(f"Absent {side} sole")
            geom.set("friction", f"{sole_friction} 0.01 0.001")
        path = OUT / f"cove50-direct-{name}.xml"
        ET.indent(tree, space="  ")
        tree.write(path, encoding="unicode", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(path))
        if (model.nu, model.nq, model.npair) != (source.nu, source.nq, source.npair):
            raise ValueError(f"Policy or contact interface changed for {name}")
        expected_mass = source.body_mass.sum() * mass_scale
        if abs(model.body_mass.sum() - expected_mass) > 1e-6:
            raise ValueError(f"Mass scaling failed for {name}")
        for side in ("left", "right"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
            if abs(model.geom_friction[geom_id, 0] - sole_friction) > 1e-9:
                raise ValueError(f"Friction scaling failed for {name}")
        rows.append({"name": name, "path": str(path.relative_to(ROOT)),
                     "sha256": sha256(path), "mass_scale": mass_scale,
                     "sole_slide_friction": sole_friction,
                     "compiled_mass_kg": round(float(model.body_mass.sum()), 6),
                     "body_inertials_scaled": count})
    (OUT / "eval-variants.json").write_text(json.dumps({
        "base_body": str(BASE.relative_to(ROOT)),
        "base_sha256": sha256(BASE),
        "purpose": "held-out sensitivity, not calibrated hardware uncertainty",
        "variants": rows,
    }, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
