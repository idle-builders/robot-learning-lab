"""Checked cap STL export with explicit removal of zero-area export triangles."""

import trimesh
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.StlAPI import StlAPI_Writer


def clean_export_mesh(mesh):
    original_volume = mesh.volume
    clean = mesh.copy()
    keep = clean.nondegenerate_faces(height=1e-8)
    removed = int((~keep).sum())
    clean.update_faces(keep)
    clean.remove_unreferenced_vertices()
    delta = abs(clean.volume - original_volume)
    if delta > max(abs(original_volume) * 1e-7, 1e-8):
        raise ValueError("Degenerate-face cleanup changed volume")
    if not clean.is_watertight or not clean.is_winding_consistent:
        raise ValueError("STL retains an open seam or inconsistent winding")
    return clean, dict(
        degenerate_faces_removed=removed,
        cleanup_volume_change_mm3=delta,
        vertex_positions_changed=False,
    )


def export_checked_stl(shape, target):
    raw = target.with_name(target.stem + ".raw.stl")
    BRepTools.Clean_s(shape.wrapped)
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, 0.03, False, 0.1, True)
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    if not mesher.IsDone() or not writer.Write(shape.wrapped, str(raw)):
        raise ValueError("Cap STL export failed")
    mesh, cleanup = clean_export_mesh(trimesh.load_mesh(raw))
    mesh.export(target)
    mesh = trimesh.load_mesh(target)
    error = abs(mesh.volume - shape.volume) / shape.volume
    if not mesh.is_watertight or not mesh.is_winding_consistent or error > 0.01:
        raise ValueError("Serialized cap STL identity failed")
    return mesh, dict(
        raw_stl=raw.name, volume_error_fraction=error, watertight=True, **cleanup
    )
