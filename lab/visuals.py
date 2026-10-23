"""Report which robot surfaces a MuJoCo camera actually renders."""

import re
import mujoco


def visual_geometry(model):
    expected = []
    for index in range(model.ngeom):
        geom = model.geom(index)
        body = model.body(int(geom.bodyid[0])).name
        if re.fullmatch(re.escape(body) + r"_\d+", geom.name):
            expected.append(geom.name)
    names = {
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    }
    present = [name for name in expected if name + "_cad" in names]
    mode = (
        "cad_meshes"
        if expected and len(present) == len(expected)
        else "mixed"
        if present
        else "primitive_envelopes"
    )
    return {
        "mode": mode,
        "robot_mesh_count": len(present),
        "expected_robot_mesh_count": len(expected),
        "contact_geometry": "simplified primitives; detailed mechanism fit is checked separately",
    }
