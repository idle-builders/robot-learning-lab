"""Display GLB of a body manifest: one named node per manifest part (the part's name), metres, the part's manifest colour,
meshes over 30k faces decimated (a display copy only: the native STEP and the npz meshes stay exact). No part knowledge:
everything comes from the manifest rows (mesh, placement_offset_mm, color, role).

    import display_glb; display_glb.scene(manifest).export("assembly.glb")       # used by publish_body.py package
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks import bodylib as F  # noqa: E402  (load_mesh: restores a missing mesh from the blob store by its path)

# manifest roles that carry no display geometry (allocations of material or adhesive, a drawn harness)
NOT_GEOMETRY = {"cut_material", "adhesive_allocation", "authored_harness"}
MAX_FACES = 30000


def scene(man) -> trimesh.Scene:
    sc = trimesh.Scene()
    for p in man["parts"]:
        if not p.get("mesh") or p.get("role") in NOT_GEOMETRY:
            continue
        m = F.load_mesh(p["mesh"])
        if p.get("placement_offset_mm"):
            m.apply_translation(p["placement_offset_mm"])
        if len(m.faces) > MAX_FACES:
            m = m.simplify_quadric_decimation(face_count=MAX_FACES)
        m.apply_scale(0.001)
        m.visual.face_colors = np.array(p.get("color") or [180, 180, 180, 255], np.uint8)
        sc.add_geometry(m, node_name=p["name"], geom_name=p["name"])
    return sc
