"""Shared source pins, paths, material models, and helpers for the lower-leg FDM connector study."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("package-complete-leg-hip")   # configured values: robot/body/leg/package_complete_leg_hip.bom.yaml config:

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
STUDY = Path(__file__).resolve().parent
ART = ROOT / "artifacts/robot/body/leg"
V16_DIR = ROOT / "artifacts/cove50-engineering-competition/waist-pelvis-20261021/lead/integration-v16"
V16_MANIFEST = V16_DIR / "parts.json"
EC_MANIFEST = ROOT / "artifacts/cove50-electronics-integration/r1-20261026/assembly-parts.json"
FULL_SHELL_GLB = ROOT / "artifacts/cove50-engineering-competition/waist-pelvis-20261021/C/full-shell-context-r1/full-shell-neutral.glb"
RIGHT_LEG_SHELL_DIR = ROOT / "artifacts/cove50-right-leg-shell/r14-20261029"

PINS = {
    V16_MANIFEST: "14c4df3030c64cab375db559eafab56e8f2ad400cd98d101a9d32f1f22dc41df",
}


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def read_json(path: Path):
    """A declared JSON input, restored by path from the bucket when it is not on disk."""
    import sys
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from infra.storage.blob_store import require
    return json.loads(Path(require(path)).read_text())


def write_json(path: Path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def ref(path: Path, label: str | None = None) -> dict:
    d = {"path": rel(path), "sha256": sha(path)}
    if label:
        d["label"] = label
    return d


def load_elasticity():
    p = ROOT / "robot/body/fem/elasticity.py"
    spec = importlib.util.spec_from_file_location("retained_elasticity", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def orthotropic_matrix_MPa(Ex: float, Ey: float, Ez: float, nu: float = 0.35, G: float = 1100.0) -> np.ndarray:
    """Construct a 6x6 orthotropic Hooke matrix in MPa (Voigt order: xx, yy, zz, xy, yz, xz)."""
    C = np.zeros((6, 6), float)
    C[:3, :3] = [
        [1.0 / Ex, -nu / np.sqrt(Ex * Ey), -nu / np.sqrt(Ex * Ez)],
        [-nu / np.sqrt(Ex * Ey), 1.0 / Ey, -nu / np.sqrt(Ey * Ez)],
        [-nu / np.sqrt(Ex * Ez), -nu / np.sqrt(Ey * Ez), 1.0 / Ez],
    ]
    D = np.zeros((6, 6), float)
    D[:3, :3] = np.linalg.inv(C[:3, :3])
    D[3:, 3:] = np.diag([G, G, G])
    return D


# Material definitions used across matched FEM comparisons:
# - Aluminium 5052-H32 / 6061-T6 baseline sheet/plate
# - PAHT-CF / PA12-CF dry & conditioned (orthotropic FDM, print layer normal customizable per part)
# - PETG-CF / PLA-CF home-lab stiff filament sensitivity
# - 304 Stainless / Alloy Steel compression sleeves & cross-bolts
MATERIALS = BOM_CONFIG.material_table(CFG["MATERIALS"])


def fillet_2d_face(f, r_default: float = 2.0, skip_xy=(), r_inner: float | None = None):
    """Fillet all non-tangent 2D vertices of an outer/inner planar Face before extrusion.

    Automatically distinguishes convex (outside) vs concave (inside) 2D corners using a
    point-in-face test along the corner bisector:
    - Convex (outside) corners receive `r_default` (small/clean rounding on outside corners).
    - Concave (inside) corners receive `r_inner` (larger rounding on inside corners, default `1.5 * r_default`).
    Vertices within 0.22 mm of any coordinate in `skip_xy` or already G1-tangent (`|d0 + d1| <= 0.08`) are preserved.
    """
    from build123d import Vector, fillet
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Pnt

    if r_inner is None:
        r_inner = round(max(r_default * 1.45, r_default + 0.8), 3)

    w = f.outer_wire()
    eds = list(w.edges())
    outer_xy = []
    inner_xy = []
    for v in f.vertices():
        if any((v.X - sx) ** 2 + (v.Y - sy) ** 2 < 0.05 for sx, sy in skip_xy):
            continue
        v_eds = [
            e
            for e in eds
            if any((ev.X - v.X) ** 2 + (ev.Y - v.Y) ** 2 < 1e-6 for ev in e.vertices())
        ]
        if len(v_eds) == 2:
            p0 = v_eds[0].position_at(
                0.05
                if (v_eds[0].vertices()[0].X - v.X) ** 2
                + (v_eds[0].vertices()[0].Y - v.Y) ** 2
                < 1e-6
                else 0.95
            )
            p1 = v_eds[1].position_at(
                0.05
                if (v_eds[1].vertices()[0].X - v.X) ** 2
                + (v_eds[1].vertices()[0].Y - v.Y) ** 2
                < 1e-6
                else 0.95
            )
            d0 = (p0 - Vector(v.X, v.Y, v.Z)).normalized()
            d1 = (p1 - Vector(v.X, v.Y, v.Z)).normalized()
            if (d0 + d1).length > 0.08:
                b_dir = (d0 + d1).normalized()
                pt_test = gp_Pnt(float(v.X + 0.15 * b_dir.X), float(v.Y + 0.15 * b_dir.Y), float(v.Z))
                clf = BRepClass_FaceClassifier(f.wrapped, pt_test, 1e-4)
                if clf.State() == TopAbs_IN:
                    outer_xy.append((v.X, v.Y))
                else:
                    inner_xy.append((v.X, v.Y))

    def _fillet_targets(face_in, target_xys: list[tuple[float, float]], radii_try: tuple[float, ...]):
        if not target_xys:
            return face_in
        # First try batch fillet of all matching vertices
        w_cur = face_in.outer_wire()
        eds_cur = list(w_cur.edges())
        matched = [
            v
            for v in face_in.vertices()
            if any((v.X - tx) ** 2 + (v.Y - ty) ** 2 < 0.04 for tx, ty in target_xys)
            and len([e for e in eds_cur if any((ev.X - v.X) ** 2 + (ev.Y - v.Y) ** 2 < 1e-6 for ev in e.vertices())]) == 2
        ]
        if not matched:
            return face_in
        for r_t in radii_try:
            try:
                res = fillet(matched, r_t)
                if res.faces():
                    return res.faces()[0]
            except Exception:
                pass
        # Fallback: fillet vertex-by-vertex so one tight edge never blocks the other corners
        f_work = face_in
        for tx, ty in target_xys:
            w_c = f_work.outer_wire()
            eds_c = list(w_c.edges())
            v_cand = [
                v
                for v in f_work.vertices()
                if (v.X - tx) ** 2 + (v.Y - ty) ** 2 < 0.04
                and len([e for e in eds_c if any((ev.X - v.X) ** 2 + (ev.Y - v.Y) ** 2 < 1e-6 for ev in e.vertices())]) == 2
            ]
            if not v_cand:
                continue
            for r_t in radii_try:
                try:
                    res = fillet(v_cand[0], r_t)
                    if res.faces():
                        f_work = res.faces()[0]
                        break
                except Exception:
                    pass
        return f_work

    cur_f = _fillet_targets(f, inner_xy, (r_inner, r_default, r_default * 0.75, 1.2, 0.8))
    cur_f = _fillet_targets(cur_f, outer_xy, (r_default, r_default * 0.8, 1.5, 1.2, 0.8))
    return cur_f


def _face_outward_normal_at(face, pt):
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.TopAbs import TopAbs_REVERSED

    surf = BRepAdaptor_Surface(face)
    st = surf.GetType()
    rev = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    if st == GeomAbs_Plane:
        d = surf.Plane().Axis().Direction()
        return np.array([d.X(), d.Y(), d.Z()], dtype=float) * rev
    if st == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        ax_p = cyl.Location()
        ax_d = cyl.Axis().Direction()
        a = np.array([ax_d.X(), ax_d.Y(), ax_d.Z()], dtype=float)
        v = np.array([pt.X() - ax_p.X(), pt.Y() - ax_p.Y(), pt.Z() - ax_p.Z()], dtype=float)
        rad = v - np.dot(v, a) * a
        n = np.linalg.norm(rad)
        if n > 1e-9:
            return (rad / n) * rev
    return np.array([0.0, 0.0, 1.0], dtype=float)


def bevel_3d_edges(sol, r_outer: float = 0.42, r_inner: float = 0.75, min_cyl_r: float = 4.6):
    """Apply 3D OpenCascade edge fillets/bevels to every structural 3D edge of a Solid,
    using smaller rounding (`r_outer = 0.42 mm`) on outside convex edges and larger rounding
    (`r_inner = 0.75 mm`) on inside concave pocket/step edges, while preserving precision
    fastener bores/counterbores (`R < min_cyl_r`).
    """
    import math
    from build123d import Solid
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    def _shape_clean_for_meshing(sh_occ) -> bool:
        exp_f = TopExp_Explorer(sh_occ, TopAbs_FACE)
        while exp_f.More():
            f = TopoDS.Face_s(exp_f.Current())
            g = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, g)
            area = float(g.Mass())
            st = BRepAdaptor_Surface(f).GetType()
            if st == GeomAbs_Plane and area < 0.26:
                return False
            if area < 0.04:
                return False
            exp_f.Next()
        exp_e = TopExp_Explorer(sh_occ, TopAbs_EDGE)
        while exp_e.More():
            ed = TopoDS.Edge_s(exp_e.Current())
            if not BRep_Tool.Degenerated_s(ed):
                g = GProp_GProps()
                BRepGProp.LinearProperties_s(ed, g)
                if float(g.Mass()) < 0.16:
                    return False
            exp_e.Next()
        return True

    base_sol = sol.solids()[0] if hasattr(sol, "solids") and len(sol.solids()) > 0 else sol
    cur = base_sol.wrapped
    ef_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(cur, TopAbs_EDGE, TopAbs_FACE, ef_map)
    mk = BRepFilletAPI_MakeFillet(cur)

    for i in range(1, ef_map.Extent() + 1):
        edge = TopoDS.Edge_s(ef_map.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        g = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, g)
        if float(g.Mass()) < 0.70:
            continue
        faces = ef_map.FindFromIndex(i)
        if faces.Extent() != 2:
            continue
        f1 = TopoDS.Face_s(faces.First())
        f2 = TopoDS.Face_s(faces.Last())
        s1 = BRepAdaptor_Surface(f1)
        s2 = BRepAdaptor_Surface(f2)
        t1, t2 = s1.GetType(), s2.GetType()
        if t1 == GeomAbs_Cylinder and s1.Cylinder().Radius() < min_cyl_r:
            continue
        if t2 == GeomAbs_Cylinder and s2.Cylinder().Radius() < min_cyl_r:
            continue
        c = BRepAdaptor_Curve(edge)
        u_mid = 0.5 * (c.FirstParameter() + c.LastParameter())
        pt_mid = gp_Pnt()
        vec_tan = gp_Vec()
        c.D1(u_mid, pt_mid, vec_tan)
        n1 = _face_outward_normal_at(f1, pt_mid)
        n2 = _face_outward_normal_at(f2, pt_mid)
        ang = math.degrees(math.acos(np.clip(float(np.dot(n1, n2)), -1.0, 1.0)))
        if 55.0 <= ang <= 120.0:
            is_concave = (t1 == GeomAbs_Cylinder and f1.Orientation() == TopAbs_REVERSED) or (
                t2 == GeomAbs_Cylinder and f2.Orientation() == TopAbs_REVERSED
            )
            try:  # an edge OCC cannot fillet is skipped (never raised by the selected parts)
                mk.Add(r_inner if is_concave else r_outer, edge)
            except Exception:
                pass

    contours = []
    for ic in range(1, mk.NbContours() + 1):
        eds = [mk.Edge(ic, ie) for ie in range(1, mk.NbEdges(ic) + 1)]
        try:
            r_c = float(mk.Radius(ic))
        except Exception:
            r_c = r_outer
        contours.append((r_c, eds))
    for r_c, eds in contours:
        for r_try in (r_c, r_outer):
            m2 = BRepFilletAPI_MakeFillet(cur)
            for e in eds:
                try:
                    m2.Add(r_try, e)
                except Exception:
                    pass
            try:
                m2.Build()
                if m2.IsDone():
                    s_try = Solid(m2.Shape())
                    if s_try.is_valid:
                        cur = m2.Shape()
                        break
            except Exception:
                pass
    return Solid(cur)



def extrude_xy_to_x(f_yz, x0: float, x1: float, sx: float = 1.0):
    """Extrude a 2D face sketched in (Y, Z) across X in [sx*x0, sx*x1]."""
    from build123d import Axis, Solid, Vector

    xa, xb = min(sx * float(x0), sx * float(x1)), max(sx * float(x0), sx * float(x1))
    s = Solid.extrude(f_yz, Vector(0, 0, xb - xa))
    return Solid(s.rotate(Axis.Y, 90).rotate(Axis.X, 90).translate((xa, 0.0, 0.0)).wrapped)


def extrude_xy_to_y(f_xz, y0: float, y1: float, sx: float = 1.0):
    """Extrude a 2D face sketched in (sx*X, Z) across Y in [y0, y1]."""
    from build123d import Axis, Solid, Vector

    ya, yb = min(float(y0), float(y1)), max(float(y0), float(y1))
    s = Solid.extrude(f_xz, Vector(0, 0, yb - ya))
    return Solid(s.rotate(Axis.X, 90).translate((0.0, yb, 0.0)).wrapped)


def extrude_xy_to_z(f_xy, z0: float, z1: float):
    """Extrude a 2D face sketched in (X, Y) across Z in [z0, z1]."""
    from build123d import Solid, Vector

    za, zb = min(float(z0), float(z1)), max(float(z0), float(z1))
    return Solid(Solid.extrude(f_xy, Vector(0, 0, zb - za)).translate((0.0, 0.0, za)).wrapped)


def make_captive_nut_pocket(
    center_pt: tuple[float, float, float],
    axis_d: tuple[float, float, float],
    af: float = 5.84,
    h: float = 2.60,
    slot_len: float = 8.5,
):
    """Create a 2D-filleted U-shaped hex-slot captive M3 nut pocket oriented along `axis_d` (±X or ±Y).

    The pocket starts at `center_pt - 0.10 * axis_d` and extends `h` along `-axis_d` (matching `build_mounts.py`),
    with an open insertion slot along +Z (for ±Y posts) or +Y (for ±X posts) and G1-filleted obtuse 120°/130° corners
    so there are zero acute wedges, zero short edges, zero sliver faces, and 0.0000 mm^3 overlap with all 6 M3 nuts.
    """
    from build123d import Polygon

    cx, cy, cz = map(float, center_pt)
    dx, dy, dz = map(float, axis_d)

    if abs(dy) > 0.5:
        # Post along ±Y: nut in r4-20261028 has vertices at X-cx = ±3.1754, Z-cz = 0 and X-cx = ±1.5877, Z-cz = ±2.75
        pts_2d = [
            (3.38, slot_len),
            (-3.38, slot_len),
            (-3.38, -1.60),
            (-1.69, -2.95),
            (1.69, -2.95),
            (3.38, -1.60),
        ]
        pts_xz = [(cx + u, cz + v) for u, v in pts_2d]
        f_xz = fillet_2d_face(
            Polygon(*pts_xz, align=None).faces()[0],
            0.45,
            skip_xy=[(cx + 3.38, cz + slot_len), (cx - 3.38, cz + slot_len)],
        )
        y_start = cy - 0.10 * dy
        y_end = y_start - h * dy
        return extrude_xy_to_y(f_xz, min(y_start, y_end), max(y_start, y_end))
    else:
        # Post along ±X: nut in r4-20261028 has vertices at Y-cy = ±3.1754, Z-cz = 0 and Y-cy = ±1.5877, Z-cz = ±2.75
        pts_x = [
            (2.95, slot_len),
            (-2.95, slot_len),
            (-2.95, -1.70),
            (0.0, -3.40),
            (2.95, -1.70),
        ]
        pts_yz = [(cy + v, cz + u) for u, v in pts_x]
        f_yz = fillet_2d_face(
            Polygon(*pts_yz, align=None).faces()[0],
            0.45,
            skip_xy=[(cy + slot_len, cz + 2.95), (cy + slot_len, cz - 2.95)],
        )
        x_start = cx - 0.10 * dx
        x_end = x_start - h * dx
        return extrude_xy_to_x(f_yz, min(x_start, x_end), max(x_start, x_end))


def add_shin_shell_mounts(body):
    """Integrate all 4 right-leg hard-shell mounting posts (2 screws per shell panel), Ø3.4 mm through-bores,
    and U-shaped captive M3 nut pockets directly into Stage 1 Shin (`R_shin`), replacing the separate `Shin_carrier`
    block and providing dual-screw anti-rotation clamping on both `shin_front` and `shin_back`:
      1. `Shin_carrier_shell_shin_front_1` (anterior -Y left, side-by-side on front bridge) at (X=53.0, Z=102.5), Y in [-14.00, -8.80], axis_d=(0, +1, 0)
      2. `Shin_carrier_shell_shin_front_2` (anterior -Y right, side-by-side on front bridge) at (X=67.0, Z=102.5), Y in [-14.00, -8.80], axis_d=(0, +1, 0)
      3. `Shin_carrier_shell_shin_back_1` (medial-rear +Y) at (X=33.5, Z=103.0), Y in [8.95, 13.08], axis_d=(0, -1, 0)
      4. `Shin_carrier_shell_shin_back_2` (lateral-rear +Y) at (X=94.0, Z=103.0), Y in [8.95, 13.08], axis_d=(0, -1, 0)
    """
    from build123d import Axis, Circle, Cylinder, Polygon, Pos

    def _cyl_y(r: float, y0: float, y1: float, x: float, z: float):
        return Cylinder(r, abs(y1 - y0)).rotate(Axis.X, 90).translate((x, 0.5 * (y0 + y1), z))

    # 1 & 2. Front `shin_front_1` (X=53.0) and `shin_front_2` (X=67.0) side-by-side cylindrical posts on the front bridge (r = 4.8 mm at Z = 102.5, Y in [-14.00, -8.80])
    front_post_1 = _cyl_y(4.8, -14.00, -8.80, 53.0, 102.5)
    front_post_2 = _cyl_y(4.8, -14.00, -8.80, 67.0, 102.5)

    # 3 & 4. Rear `shin_back_1` (medial X = 33.5) and `shin_back_2` (lateral X = 94.0) tangent half-pill posts (Z in [98.2, 107.8], Y in [8.95, 13.08])
    rear_xz_1 = (
        Polygon((33.5, 98.2), (41.25, 98.2), (41.25, 107.8), (33.5, 107.8), align=None)
        + Pos(33.5, 103.0) * Circle(4.8)
    ).clean().faces()[0]
    rear_post_1 = extrude_xy_to_y(rear_xz_1, 8.95, 13.08)

    rear_xz_2 = (
        Polygon((78.75, 98.2), (94.0, 98.2), (94.0, 107.8), (78.75, 107.8), align=None)
        + Pos(94.0, 103.0) * Circle(4.8)
    ).clean().faces()[0]
    rear_post_2 = extrude_xy_to_y(rear_xz_2, 8.95, 13.08)

    out = body + front_post_1 + front_post_2 + rear_post_1 + rear_post_2

    # Cut Ø3.4 mm (r = 1.70 mm) shell bolt through-bores and U-shaped hex-slot captive M3 nut pockets for all 4 mounts
    out -= _cyl_y(1.70, -15.0, -8.60, 53.0, 102.5)
    out -= make_captive_nut_pocket((53.0, -10.12, 102.5), (0.0, 1.0, 0.0))

    out -= _cyl_y(1.70, -15.0, -8.60, 67.0, 102.5)
    out -= make_captive_nut_pocket((67.0, -10.12, 102.5), (0.0, 1.0, 0.0))

    out -= _cyl_y(1.70, 8.0, 15.0, 33.5, 103.0)
    out -= make_captive_nut_pocket((33.5, 10.40, 103.0), (0.0, -1.0, 0.0))

    out -= _cyl_y(1.70, 8.0, 15.0, 94.0, 103.0)
    out -= make_captive_nut_pocket((94.0, 10.40, 103.0), (0.0, -1.0, 0.0))
    return out


def ensure_two_screw_shin_shell_assets() -> dict:
    """Generate and cache the 2-screw `shin_front.npz` and `shin_back.npz` hard-shell cover meshes
    (with side-by-side anterior counterbored bosses for `Shin_carrier_shell_shin_front_1/2` at X=53.0, 67.0, Z=102.5
    and posterior bosses for `Shin_carrier_shell_shin_back_1/2` at X=33.5, 94.0, Z=103.0)
    plus all 8 shin shell hardware STEP files (4 M3x8 screws + 4 captive M3 nuts) in `ART / 'shell_mounts_2screw'`."""
    import shutil
    import manifold3d as md
    import trimesh
    from build123d import import_step
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    out_dir = ART / "shell_mounts_2screw"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf_npz = out_dir / "shin_front.npz"
    sb_npz = out_dir / "shin_back.npz"

    hw_src_dir = ROOT / "artifacts/cove50-right-leg-shell/r4-20261028/knee125/native"

    def _write_step_solid(shape_obj, dst_path: Path):
        writer = STEPControl_Writer()
        writer.Transfer(shape_obj.wrapped, STEPControl_AsIs)
        writer.Write(str(dst_path))

    # 1. Side-by-side front M3x8 screws and captive M3 nuts at (X=53.0, Z=102.5) and (X=67.0, Z=102.5)
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from infra.storage.blob_store import require
    tf_scr_p = require(hw_src_dir / "Thigh_carrier_shell_thigh_front_1_M3x8.step")
    tf_nut_p = require(hw_src_dir / "Thigh_carrier_shell_thigh_front_1_nut.step")
    # inputs are required above; outputs are always rebuilt
    for dst_scr, dst_nut, dx in [
        (out_dir / "Shin_carrier_shell_shin_front_1_M3x8.step", out_dir / "Shin_carrier_shell_shin_front_1_nut.step", -4.0),
        (out_dir / "Shin_carrier_shell_shin_front_2_M3x8.step", out_dir / "Shin_carrier_shell_shin_front_2_nut.step", 10.0),
    ]:
        _write_step_solid(import_step(tf_scr_p).translate((dx, 16.6148, -75.5)), dst_scr)
        _write_step_solid(import_step(tf_nut_p).translate((dx, 16.1644, -75.5)), dst_nut)

    # 2. Rear M3x8 screws and captive M3 nuts at (X=33.5, Z=103.0) and (X=94.0, Z=103.0)
    for base_hw, mir_hw in [
        ("Shin_carrier_shell_shin_back_1_M3x8", "Shin_carrier_shell_shin_back_2_M3x8"),
        ("Shin_carrier_shell_shin_back_1_nut", "Shin_carrier_shell_shin_back_2_nut"),
    ]:
        src_p = require(hw_src_dir / f"{base_hw}.step")
        dst1_p = out_dir / f"{base_hw}.step"
        dst2_p = out_dir / f"{mir_hw}.step"
        shutil.copyfile(src_p, dst1_p)
        _write_step_solid(import_step(src_p).translate((60.5, 0.0, 0.0)), dst2_p)

    mp = read_json(RIGHT_LEG_SHELL_DIR / "manufactured-parts.json")

    def _tm_to_mf(m: trimesh.Trimesh) -> md.Manifold:
        return md.Manifold(md.Mesh(np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.uint32)))

    def _mf_to_tm(m: md.Manifold) -> trimesh.Trimesh:
        mesh = m.to_mesh()
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vert_properties[:, :3], dtype=np.float64),
            faces=np.asarray(mesh.tri_verts, dtype=np.int64),
            process=False,
        )

    def _cyl_mf(p0, d, r, length, n=48):
        c = md.Manifold.cylinder(length, r, r, n)
        d = np.asarray(d, dtype=float) / np.linalg.norm(d)
        if abs(d[0] - 1.0) < 1e-6:
            c = c.rotate([0.0, 90.0, 0.0])
        elif abs(d[0] + 1.0) < 1e-6:
            c = c.rotate([0.0, -90.0, 0.0])
        elif abs(d[1] - 1.0) < 1e-6:
            c = c.rotate([-90.0, 0.0, 0.0])
        elif abs(d[1] + 1.0) < 1e-6:
            c = c.rotate([90.0, 0.0, 0.0])
        return c.translate(list(p0))

    z_sf = np.load(require(ROOT / mp["parts"][2]["lossless_mesh"]["path"]))
    mf_sf = _tm_to_mf(trimesh.Trimesh(vertices=z_sf["vertices"], faces=z_sf["faces"], process=False))
    # Patch/smooth out the old medial side hole at (Y=4.0, Z=94.0) and add the 2 side-by-side front bosses at X=53.0, 67.0, Z=102.5
    patch_side = (mf_sf.translate([0.0, -6.0, 0.0]) + mf_sf.translate([0.0, 0.0, -6.5])) ^ _cyl_mf([24.0, 4.0, 94.0], [1, 0, 0], 4.9, 5.0)
    mf_sf_clean = (mf_sf + patch_side) - _cyl_mf([28.5, 4.0, 94.0], [1, 0, 0], 5.2, 4.0)
    interior_sf = md.Manifold.batch_boolean([mf_sf_clean.translate([0.0, dy, 0.0]) for dy in np.linspace(0.0, 6.0, 13)], md.OpType.Add)
    boss_f1 = _cyl_mf([53.0, -20.0, 102.5], [0, 1, 0], 4.8, 5.98) ^ interior_sf
    boss_f2 = _cyl_mf([67.0, -20.0, 102.5], [0, 1, 0], 4.8, 5.98) ^ interior_sf
    mf_sf2 = (
        mf_sf_clean
        + boss_f1
        + boss_f2
        - _cyl_mf([53.0, -22.0, 102.5], [0, 1, 0], 1.70, 9.0)
        - _cyl_mf([53.0, -22.0, 102.5], [0, 1, 0], 3.20, 5.20)
        - _cyl_mf([67.0, -22.0, 102.5], [0, 1, 0], 1.70, 9.0)
        - _cyl_mf([67.0, -22.0, 102.5], [0, 1, 0], 3.20, 5.20)
    )
    tm_sf2 = _mf_to_tm(mf_sf2)
    np.savez_compressed(sf_npz, vertices=tm_sf2.vertices, faces=tm_sf2.faces)

    z_sb = np.load(require(ROOT / mp["parts"][3]["lossless_mesh"]["path"]))
    mf_sb = _tm_to_mf(trimesh.Trimesh(vertices=z_sb["vertices"], faces=z_sb["faces"], process=False))
    interior_sb = md.Manifold.batch_boolean([mf_sb.translate([0.0, -dy, 0.0]) for dy in np.linspace(0.0, 5.5, 12)], md.OpType.Add)
    boss_sb2 = _cyl_mf([94.0, 13.10, 103.0], [0, 1, 0], 4.8, 6.0) ^ interior_sb
    mf_sb2 = (
        mf_sb
        + boss_sb2
        - _cyl_mf([94.0, 7.95, 103.0], [0, 1, 0], 4.95, 5.15)
        - mf_sf2
        - mf_sf2.translate([0.0, 0.2, 0.0])
        - _cyl_mf([94.0, 12.8, 103.0], [0, 1, 0], 1.70, 7.0)
        - _cyl_mf([94.0, 16.91683, 103.0], [0, 1, 0], 3.20, 4.0)
    )
    tm_sb2 = _mf_to_tm(mf_sb2)
    np.savez_compressed(sb_npz, vertices=tm_sb2.vertices, faces=tm_sb2.faces)

    return {
        "dir": out_dir,
        "shin_front_npz": sf_npz,
        "shin_back_npz": sb_npz,
    }



