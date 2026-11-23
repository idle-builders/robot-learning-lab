"""Shared source pins, paths, material models, compression sleeves, 3D bevels (`r_inner > r_outer`),
and kinematic transforms for the 4-stage Neck & Head FDM connector study.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
STUDY = Path(__file__).resolve().parent
ART = ROOT / "artifacts/robot/body/neck-head"
EC_MANIFEST = ROOT / "artifacts/cove50-electronics-integration/r1-20261026/assembly-parts.json"

STAGE_PART_BASENAMES = {
    "stage1_neck_root_cradle": "neck_root_cradle",
    "stage2_neck_yaw_to_pitch_link": "neck_yaw_to_pitch_link",
    "stage3_neck_pitch_to_head_cradle": "neck_pitch_to_head_cradle",
    "stage4_head_sensor_compute_carrier": "head_sensor_compute_carrier",
}

STAGE_LINKS = {
    "stage1_neck_root_cradle": "torso",
    "stage2_neck_yaw_to_pitch_link": "neck_yaw",
    "stage3_neck_pitch_to_head_cradle": "head",
    "stage4_head_sensor_compute_carrier": "head",
}

# Exact baseline STEP paths for the 4 neck & head connector stages
BASELINE_STEPS = {
    "stage1_neck_root_cradle": ROOT / "artifacts/cove50-sensor-integration/v21/parts/fixed_neck_ribbon_anchor_v21.step",
    "stage2_neck_yaw_to_pitch_link": ROOT / "artifacts/cove50-engineering-competition/upper-body-20261023/C/final-r4-shell-fem/parts/one_piece_bent_neck_L.step",
    "stage3_neck_pitch_to_head_cradle": ROOT / "artifacts/cove50-sensor-integration/original-shell-layout-20261026/parts/compact_neck_U_strap_v23.step",
    "stage4_head_sensor_compute_carrier": ROOT / "artifacts/cove50-sensor-integration/original-shell-layout-20261026/parts/single_shared_carrier_v23.step",
}

# Legacy sheet-metal and boxy PA12 neck/head brackets & obsolete v21 ribbon clamp hardware replaced by the 4-stage FDM connectors
REPLACED_NECK_HEAD_PARTS = {
    "fixed_neck_ribbon_anchor_v21",
    "fixed_neck_ribbon_cap_v21",
    "fixed_neck_ribbon_pad_rear_v21",
    "fixed_neck_ribbon_pad_front_v21",
    "fixed_ribbon_M2x8_0_v21",
    "fixed_ribbon_M2_washer_0_v21",
    "fixed_ribbon_M2x8_1_v21",
    "fixed_ribbon_M2_washer_1_v21",
    "upper__one_piece_bent_neck_L",
    "compact_neck_U_strap_v23",
    "single_shared_carrier_v23",
    "microphone_compact_tab_v23",
    "mic_foot_M2x3_0",
    "mic_foot_M2x3_1",
}

# Exact neck joint origins and axes in world neutral coordinates (mm)
P_NECK_YAW = np.array([0.0, 5.0, 365.0], dtype=float)
AXIS_NECK_YAW = np.array([0.0, 0.0, 1.0], dtype=float)
P_NECK_PITCH = np.array([-13.5, 5.0, 414.0], dtype=float)
AXIS_NECK_PITCH = np.array([1.0, 0.0, 0.0], dtype=float)


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
    import importlib.util

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


MATERIALS = {
    "Al5052_H32": {
        "name": "Aluminium 5052-H32 (baseline sheet)",
        "density_g_cm3": 2.68,
        "E_MPa": 68300.0,
        "nu": 0.33,
        "reference_strength_MPa": 193.0,
    },
    "PAHT_CF_dry": {
        "name": "Bambu PAHT-CF / PA12-CF (dry, FDM)",
        "density_g_cm3": 1.22,
        "E_inplane_MPa": 7550.0,
        "E_layer_MPa": 3290.0,
        "G_MPa": 1100.0,
        "nu": 0.35,
        "tensile_XY_MPa": 92.0,
        "tensile_Z_layer_MPa": 43.0,
    },
    "PAHT_CF_wet": {
        "name": "Bambu PAHT-CF / PA12-CF (moisture-conditioned, FDM)",
        "density_g_cm3": 1.22,
        "E_inplane_MPa": 6570.0,
        "E_layer_MPa": 2700.0,
        "G_MPa": 950.0,
        "nu": 0.35,
        "tensile_XY_MPa": 71.0,
        "tensile_Z_layer_MPa": 37.0,
    },
    "PETG_CF": {
        "name": "PETG-CF / PLA-CF (home-lab standard CF filament)",
        "density_g_cm3": 1.25,
        "E_inplane_MPa": 5800.0,
        "E_layer_MPa": 2850.0,
        "G_MPa": 1000.0,
        "nu": 0.35,
        "tensile_XY_MPa": 68.0,
        "tensile_Z_layer_MPa": 34.0,
    },
    "Steel_Sleeve": {
        "name": "304 Stainless Steel compression sleeve",
        "density_g_cm3": 7.85,
        "E_MPa": 200000.0,
        "nu": 0.29,
    },
}


def fillet_2d_face(f, r_default: float = 2.0, skip_xy=(), r_inner: float | None = None):
    """Fillet all non-tangent 2D vertices of an outer/inner planar Face before extrusion,
    with larger rounding (`r_inner`) on concave inner vertices and smaller rounding (`r_default`)
    on convex outer vertices.
    """
    from build123d import Vector, fillet
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Pnt

    if r_inner is None:
        r_inner = round(max(r_default * 1.45, r_default + 0.8), 3)

    if hasattr(f, "faces") and len(f.faces()) > 0:
        f = f.faces()[0]
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
                if (v_eds[0].vertices()[0].X - v.X) ** 2 + (v_eds[0].vertices()[0].Y - v.Y) ** 2 < 1e-6
                else 0.95
            )
            p1 = v_eds[1].position_at(
                0.05
                if (v_eds[1].vertices()[0].X - v.X) ** 2 + (v_eds[1].vertices()[0].Y - v.Y) ** 2 < 1e-6
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
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.gp import gp_Pnt, gp_Vec

    bf = BRepGProp_Face(face)
    u0, u1, v0, v1 = bf.Bounds()
    p_mid = gp_Pnt()
    vn = gp_Vec()
    bf.Normal(0.5 * (u0 + u1), 0.5 * (v0 + v1), p_mid, vn)
    surf = BRepAdaptor_Surface(face)
    st = surf.GetType()
    if st == GeomAbs_Plane:
        arr = np.array([vn.X(), vn.Y(), vn.Z()], dtype=float)
        return arr / max(1e-9, np.linalg.norm(arr))
    if st == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        ax_p = cyl.Location()
        ax_d = cyl.Axis().Direction()
        a = np.array([ax_d.X(), ax_d.Y(), ax_d.Z()], dtype=float)
        v_m = np.array([p_mid.X() - ax_p.X(), p_mid.Y() - ax_p.Y(), p_mid.Z() - ax_p.Z()], dtype=float)
        r_m = v_m - np.dot(v_m, a) * a
        sign = 1.0 if np.dot([vn.X(), vn.Y(), vn.Z()], r_m) >= 0.0 else -1.0
        v = np.array([pt.X() - ax_p.X(), pt.Y() - ax_p.Y(), pt.Z() - ax_p.Z()], dtype=float)
        rad = v - np.dot(v, a) * a
        n = np.linalg.norm(rad)
        if n > 1e-9:
            return (rad / n) * sign
    arr = np.array([vn.X(), vn.Y(), vn.Z()], dtype=float)
    return arr / max(1e-9, np.linalg.norm(arr))


def bevel_3d_edges(sol, r_outer: float = 0.40, r_inner: float = 0.75, skip_fn=None):
    """Apply 3D OpenCascade edge fillets/bevels to every structural 3D edge of a Solid,
    using smaller rounding (`r_outer`) on outside convex edges and larger rounding
    (`r_inner > r_outer`) on inside concave pocket/step edges, while preserving precision
    fastener bores/counterbores.
    """
    from build123d import Solid
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    def _count_defects(sh_occ) -> tuple[int, int, int]:
        slivers = 0
        micro = 0
        exp_f = TopExp_Explorer(sh_occ, TopAbs_FACE)
        while exp_f.More():
            f = TopoDS.Face_s(exp_f.Current())
            g = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, g)
            area = float(g.Mass())
            st = BRepAdaptor_Surface(f).GetType()
            if st == GeomAbs_Plane and area < 0.25:
                slivers += 1
            if area < 0.02:
                micro += 1
            exp_f.Next()
        short_eds = 0
        ef = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(sh_occ, TopAbs_EDGE, TopAbs_FACE, ef)
        for idx in range(1, ef.Extent() + 1):
            ed = TopoDS.Edge_s(ef.FindKey(idx))
            if BRep_Tool.Degenerated_s(ed):
                continue
            g = GProp_GProps()
            BRepGProp.LinearProperties_s(ed, g)
            if float(g.Mass()) < 0.30:
                faces = ef.FindFromIndex(idx)
                if faces.Extent() == 2:
                    f1 = TopoDS.Face_s(faces.First())
                    f2 = TopoDS.Face_s(faces.Last())
                    if (
                        BRepAdaptor_Surface(f1).GetType() == GeomAbs_Plane
                        and BRepAdaptor_Surface(f2).GetType() == GeomAbs_Plane
                    ):
                        c = BRepAdaptor_Curve(ed)
                        if c.GetType() == GeomAbs_Line:
                            u_m = 0.5 * (c.FirstParameter() + c.LastParameter())
                            p_m = c.Value(u_m)
                            n1 = _face_outward_normal_at(f1, p_m)
                            n2 = _face_outward_normal_at(f2, p_m)
                            ang = math.degrees(math.acos(np.clip(float(np.dot(n1, n2)), -1.0, 1.0)))
                            if ang > 10.0:
                                short_eds += 1
        return slivers, micro, short_eds

    base_sol = sol.solids()[0] if hasattr(sol, "solids") and len(sol.solids()) > 0 else sol
    cur = base_sol.wrapped
    base_slivers, base_micro, base_short = _count_defects(cur)

    def _shape_clean_for_meshing(sh_occ) -> bool:
        slivers, micro, short_eds = _count_defects(sh_occ)
        return slivers <= base_slivers and micro <= base_micro and short_eds <= base_short

    ef_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(cur, TopAbs_EDGE, TopAbs_FACE, ef_map)
    mk_probe = BRepFilletAPI_MakeFillet(cur)
    mid_conc: dict[tuple[float, float, float], bool] = {}

    valid_axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    for i in range(1, ef_map.Extent() + 1):
        edge = TopoDS.Edge_s(ef_map.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        g = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, g)
        if float(g.Mass()) < 0.60:
            continue
        faces = ef_map.FindFromIndex(i)
        if faces.Extent() != 2:
            continue
        f1 = TopoDS.Face_s(faces.First())
        f2 = TopoDS.Face_s(faces.Last())
        s1 = BRepAdaptor_Surface(f1)
        s2 = BRepAdaptor_Surface(f2)
        t1, t2 = s1.GetType(), s2.GetType()
        if t1 != GeomAbs_Plane or t2 != GeomAbs_Plane:
            continue
        c = BRepAdaptor_Curve(edge)
        if c.GetType() != GeomAbs_Line:
            continue
        ld = c.Line().Direction()
        d_vec = np.array([ld.X(), ld.Y(), ld.Z()], dtype=float)
        if max(abs(float(np.dot(d_vec, ax))) for ax in valid_axes) < 0.985:
            continue
        u_mid = 0.5 * (c.FirstParameter() + c.LastParameter())
        pt_mid = gp_Pnt()
        vec_tan = gp_Vec()
        c.D1(u_mid, pt_mid, vec_tan)
        if skip_fn is not None and skip_fn(pt_mid.X(), pt_mid.Y(), pt_mid.Z()):
            continue
        n1 = _face_outward_normal_at(f1, pt_mid)
        n2 = _face_outward_normal_at(f2, pt_mid)
        ang = math.degrees(math.acos(np.clip(float(np.dot(n1, n2)), -1.0, 1.0)))
        if 22.0 <= ang <= 125.0:
            is_concave = False
            for f_chk, n_chk, n_oth in ((f1, n1, n2), (f2, n2, n1)):
                gp_f = GProp_GProps()
                BRepGProp.SurfaceProperties_s(f_chk, gp_f)
                cm = gp_f.CentreOfMass()
                v_to_cm = np.array([cm.X() - pt_mid.X(), cm.Y() - pt_mid.Y(), cm.Z() - pt_mid.Z()])
                v_in = v_to_cm - np.dot(v_to_cm, n_chk) * n_chk
                if np.linalg.norm(v_in) > 1e-6:
                    v_in /= np.linalg.norm(v_in)
                    if float(np.dot(v_in, n_oth)) > 0.05:
                        is_concave = True
                        break
            try:
                mk_probe.Add(r_outer, edge)
                mid_conc[(round(pt_mid.X(), 3), round(pt_mid.Y(), 3), round(pt_mid.Z(), 3))] = is_concave
            except Exception:
                pass

    contours = []
    for ic in range(1, mk_probe.NbContours() + 1):
        eds = [mk_probe.Edge(ic, ie) for ie in range(1, mk_probe.NbEdges(ic) + 1)]
        concs = []
        for e in eds:
            c_e = BRepAdaptor_Curve(e)
            if c_e.GetType() == GeomAbs_Line:
                p_e = c_e.Value(0.5 * (c_e.FirstParameter() + c_e.LastParameter()))
                concs.append(mid_conc.get((round(p_e.X(), 3), round(p_e.Y(), 3), round(p_e.Z(), 3)), False))
        r_c = r_inner if (concs and all(concs)) else r_outer
        contours.append((r_c, eds))

    def _try_build(c_list, check_tess: bool = False):
        if not c_list:
            return None
        m = BRepFilletAPI_MakeFillet(cur)
        for r_val, eds in c_list:
            for e in eds:
                try:
                    m.Add(r_val, e)
                except Exception:
                    pass
        try:
            m.Build()
            if m.IsDone():
                s_try = Solid(m.Shape())
                if s_try.is_valid and len(s_try.solids()) == 1 and _shape_clean_for_meshing(m.Shape()):
                    if check_tess:
                        solid_to_watertight_arrays(s_try, tol=0.05)
                    return m.Shape()
        except Exception:
            pass
        return None

    sh_all = _try_build(contours, check_tess=True)
    if sh_all is not None:
        return Solid(sh_all)

    solo_valid: list[tuple[float, list]] = []
    for r_orig, eds in contours:
        if _try_build([(r_orig, eds)], check_tess=True) is not None:
            solo_valid.append((r_orig, eds))
        elif r_outer < r_orig and _try_build([(r_outer, eds)], check_tess=True) is not None:
            solo_valid.append((r_outer, eds))

    sh_solo = _try_build(solo_valid, check_tess=True)
    if sh_solo is not None:
        return Solid(sh_solo)

    good: list[tuple[float, list]] = []
    chunk_size = 6
    for start_idx in range(0, len(solo_valid), chunk_size):
        chunk = solo_valid[start_idx : start_idx + chunk_size]
        if _try_build(good + chunk, check_tess=True) is not None:
            good.extend(chunk)
        else:
            for r_orig, eds in chunk:
                if _try_build(good + [(r_orig, eds)], check_tess=True) is not None:
                    good.append((r_orig, eds))
                elif r_outer < r_orig and _try_build(good + [(r_outer, eds)], check_tess=True) is not None:
                    good.append((r_outer, eds))

    sh_best = _try_build(good, check_tess=True)
    return Solid(sh_best if sh_best is not None else cur)


def extrude_xy_to_x(f_yz, x0: float, x1: float):
    """Extrude a 2D face sketched in (Y, Z) across X in [x0, x1]."""
    from build123d import Axis, Solid, Vector

    if hasattr(f_yz, "faces") and len(f_yz.faces()) > 0:
        f_yz = f_yz.faces()[0]
    xa, xb = min(float(x0), float(x1)), max(float(x0), float(x1))
    s = Solid.extrude(f_yz, Vector(0, 0, xb - xa))
    return Solid(s.rotate(Axis.Y, 90).rotate(Axis.X, 90).translate((xa, 0.0, 0.0)).wrapped)


def extrude_xy_to_y(f_xz, y0: float, y1: float):
    """Extrude a 2D face sketched in (X, Z) across Y in [y0, y1]."""
    from build123d import Axis, Solid, Vector

    if hasattr(f_xz, "faces") and len(f_xz.faces()) > 0:
        f_xz = f_xz.faces()[0]
    ya, yb = min(float(y0), float(y1)), max(float(y0), float(y1))
    s = Solid.extrude(f_xz, Vector(0, 0, yb - ya))
    return Solid(s.rotate(Axis.X, 90).translate((0.0, yb, 0.0)).wrapped)


def extrude_xy_to_z(f_xy, z0: float, z1: float):
    """Extrude a 2D face sketched in (X, Y) across Z in [z0, z1]."""
    from build123d import Solid, Vector

    if hasattr(f_xy, "faces") and len(f_xy.faces()) > 0:
        f_xy = f_xy.faces()[0]
    za, zb = min(float(z0), float(z1)), max(float(z0), float(z1))
    return Solid(Solid.extrude(f_xy, Vector(0, 0, zb - za)).translate((0.0, 0.0, za)).wrapped)


def cyl_along(pt, direction, r: float, length: float):
    """Create a Solid cylinder starting at `pt` and extending `length` along `direction`."""
    from build123d import Align, Cylinder, Plane, Solid

    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    p = tuple(float(x) for x in pt)
    c = Plane(origin=p, z_dir=tuple(float(x) for x in d)) * Cylinder(
        float(r), float(length), align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    return Solid(c.wrapped)


# @lore Sleeve length = plate thickness at an XM335 screw site, chosen so a standard screw length reaches the hole's reference
# engagement with the tip inside the hole (screw_plan.choose_length): the case M2 holes are 1.8 mm deep from a seat 0.2 mm
# outside the opening, so a 2.5 mm sleeve takes M2x5 (engagement 1.8, tip at the bottom); the horn M2 holes are 4.1 mm deep,
# reference 3.0, so a 3.5 mm sleeve takes M2x8 (engagement 3.8, tip 0.3 inside). Sleeves are full-thickness and flush (M-evmrv6).
# Head-skin taper cut of the microphone wing's top (x >= MIC_TAPER["x0"], z in [z0, z1]): builder and sleeve lengths share it
MIC_TAPER = {"x0": 47.8, "z0": 467.2, "z1": 472.4}


def neck_anchor() -> dict:
    """The neck anchor joint (cradle to torso chassis, T-jxshhv): robot/body/neck-head/neck_anchor.json."""
    return json.loads((Path(__file__).resolve().parent / "neck_anchor.json").read_text())


def head_std_m3_button() -> dict:
    """ISO 7380-1 M3 head (dk, k, lengths) from the shared screw standards table (robot/body/checks/data/hardware.json)."""
    st = json.loads((ROOT / "robot/body/checks/data/hardware.json").read_text())["screw_standards"]["head_types"]
    return st[neck_anchor()["head"]["standard"]][neck_anchor()["head"]["thread"]]


def stage_interface_specs() -> dict:
    """Return exact hole centers, per-hole outward normals, and the pressed-in sleeve / insert specifications
    for all 4 Neck & Head connector stages (stages 1-3: 304 SS compression sleeves; stage 4: brass heat-set inserts, plain holes elsewhere):
      - Stage 1 (`stage1_neck_root_cradle`, on `torso` — 4 M2 sleeves; the 2 M3 anchor holes are plain, T-jxshhv):
          * Support: 2x M3 torso anchor screws (neck_anchor.json sites, x -30 / -17, z 347.5, on the plate's front face y 25, normal `+Y`):
            plain clearance holes, no sleeve; ISO 7380-1 M3 into brass heat-set inserts in the chassis pad (T-jxshhv)
          * Load: 4x M2 `neck_yaw_case_XM335` rear & left-cheek seats:
            `(-6.5, 30.5, 356.75)`, `(-6.5, 30.5, 362.25)`, `(6.5, 30.5, 356.75)`, `(6.5, 30.5, 362.25)` (normal `+Y`, `h = 3.0 mm`)
      - Stage 2 (`stage2_neck_yaw_to_pitch_link`, on `neck_yaw` — 8 sleeves: 4x M2 yaw horn + 4x M2 pitch horn):
          * Support: all 4x M2 holes on `upper__neck_yaw_front_horn` (`PCD 12.0 mm` at `(0, 5, 372.5)`:
            `(0, -1, 372.5)`, `(0, 11, 372.5)`, `(-6, 5, 372.5)`, `(6, 5, 372.5)`, normal `+Z`, `h = 3.0 mm`)
          * Load: all 4x M2 holes on `upper__neck_pitch_front_horn` (`PCD 12.0 mm` at `(-21, 5, 414.0)`:
            `(-21, 5, 408)`, `(-21, 5, 420)`, `(-21, -1, 414)`, `(-21, 11, 414)`, normal `-X`, `h = 3.0 mm`)
      - Stage 3 (`stage3_neck_pitch_to_head_cradle`, on `head` — 4 sleeves + 2 inserts: 4x M2 pitch case sleeves + 2x M3 heat-set inserts):
          * Support: 4x M2 rear taps on `upper__neck_pitch_case_XM335`:
            `(-10.75, 14.5, 391.5)`, `(-10.75, 14.5, 401.0)`, `(-5.25, 14.5, 391.5)`, `(-5.25, 14.5, 401.0)` (normal `+Y`, `h = 3.0 mm`)
          * Load: 2x brass M3 x 4 heat-set inserts in the tongue (head_carrier_posts.json neck_root: x -0.85, z 413 / 419, from the
            tongue's rear face y -12.92 to its front face, normal `-Y`; T-9wjdqc)
      - Stage 4 (`stage4_head_sensor_compute_carrier`, on `head` — 9 brass M2 heat-set inserts: 8 cameras + 1 mic; T-5bqc3m):
          * Support: 2x M3 neck carrier root holes (head_carrier_posts.json neck_root sites, normal `+Y`): plain holes, no sleeve
          * Load: 4x M2.5 ROCK4D compute posts (measured, head_holes.compute_posts), normal `+Y`: plain holes, no sleeve
            + 4x M2 B0410 camera posts (`(-36.0, -12.9, 441.0)`, `(-36.0, -12.9, 471.0)`, `(-6.0, -12.9, 441.0)`, `(-6.0, -12.9, 471.0)`, normal `+Y`, `h = 1.5 mm`)
            + 4x M2 B0196 camera posts (`(7.0, -12.9, 444.1662)`, `(7.0, -12.9, 472.1662)`, `(35.0, -12.9, 444.1662)`, `(35.0, -12.9, 472.1662)`, normal `+Y`, `h = 1.5 mm`)
    """
    # Screw sites of the XM335 horns / cases come from the standard-part definitions (defined_holes.py, M-jpppfb).
    import defined_holes as DH
    sites = {c: DH.seat_sites(c) for c in DH.connectors()}

    def _pts(connector, group):
        return [np.asarray(x["centre"], float) for x in sites[connector] if x["group"] == group]

    def _norms(connector, group):
        return [np.asarray(x["axis_out"], float) for x in sites[connector] if x["group"] == group]

    # Stage 1 (`neck_root_cradle`): the 2 torso anchor screws (T-jxshhv, neck_anchor.json): sites on the spine plate's front face, which lies
    # on the chassis pad; plain clearance holes, ISO 7380-1 heads on the plate's rear face, brass inserts in the chassis
    na = neck_anchor()
    s1_sup_pts = [np.array([float(x), float(na["face_y_mm"]), float(z)]) for x, z in na["sites_xz_mm"]]
    s1_sup_norms = [np.array([0.0, 1.0, 0.0])] * len(s1_sup_pts)

    s1_load_pts = _pts("neck_root_cradle", "yaw_case_rear")
    s1_load_norms = _norms("neck_root_cradle", "yaw_case_rear")

    # Stage 2 (`neck_yaw_to_pitch_link`)
    s2_sup_pts = _pts("neck_yaw_to_pitch_link", "yaw_horn")
    s2_sup_norms = _norms("neck_yaw_to_pitch_link", "yaw_horn")

    s2_load_pts = _pts("neck_yaw_to_pitch_link", "pitch_horn")
    s2_load_norms = _norms("neck_yaw_to_pitch_link", "pitch_horn")

    # Stage 3 (`neck_pitch_to_head_cradle`)
    s3_sup_pts = _pts("neck_pitch_to_head_cradle", "pitch_case_rear")
    s3_sup_norms = _norms("neck_pitch_to_head_cradle", "pitch_case_rear")

    # neck root (T-9wjdqc): one pair of axes for the carrier holes and the cradle tongue's M3 heat-set inserts (head_carrier_posts.json
    # neck_root); the inserts sit flush in the tongue from its rear face (the carrier's seat) to its front face
    import head_holes as HH
    nr = HH.POSTS["neck_root"]
    ins3 = HH.root_insert()
    y_tongue = float(nr["tongue_rear_y_mm"])
    s3_load_pts = [np.array([float(x), y_tongue, float(z)]) for x, z in nr["sites_xz_mm"]]
    s3_load_norms = [np.array([0.0, -1.0, 0.0])] * len(s3_load_pts)

    # Stage 4 (`head_sensor_compute_carrier`)
    s4_sup_pts = [np.array([float(x), -12.9, float(z)]) for x, z in nr["sites_xz_mm"]]
    s4_sup_norms = [np.array([0.0, 1.0, 0.0])] * len(s4_sup_pts)

    # compute, camera posts: bores of the supplier spacers / nuts (head_holes.py, T-5bqc3m)
    hs = HH.sites()
    s4_load_pts = hs["compute"]
    s4_cam_pts = hs["camera"]
    s4_load_norms = [np.array([0.0, 1.0, 0.0])] * len(s4_load_pts)
    s4_cam_norms = [np.array([0.0, 1.0, 0.0])] * len(s4_cam_pts)
    cm = HH.sleeve("camera")
    comp, root = HH.POSTS["compute"], HH.POSTS["neck_root"]
    root_head = HH.head_std(root["head"])
    plate_t = HH.plate_faces()[1] - HH.plate_faces()[0]
    cam_lens, cam_starts = HH.sleeve_lens("camera"), HH.sleeve_starts("camera")
    mic_rows = HH.threaded_posts()["mic"]
    mic_pts = [np.asarray(r["plan"]["opening"], float) for r in mic_rows]

    return {
        "stage1_neck_root_cradle": {
            "support": {    # plain clearance holes, no sleeve (T-jxshhv): ISO 7380-1 M3 into brass inserts in the chassis pad
                "name": "torso_spine_anchor",
                "normal": np.array([0.0, 1.0, 0.0]),
                "normals": s1_sup_norms,
                "centres": s1_sup_pts,
                "no_sleeve": True,
                "sleeve_ro": float(na["hole_dia"]) / 2.0,
                "sleeve_ri": float(na["hole_dia"]) / 2.0,
                "sleeve_len": float(na["plate_rear_y_mm"]) - float(na["face_y_mm"]),
                "head_clear_r": head_std_m3_button()["dk"] / 2.0 + float(na["head_clear_mm"]),
                "patch_r": 2.8,
            },
            "load": {
                "name": "neck_yaw_case_rear",
                "normal": np.array([0.0, 1.0, 0.0]),
                "normals": s1_load_norms,
                "centres": s1_load_pts,
                "sleeve_ro": 1.75,
                "sleeve_ri": 1.10,
                "sleeve_len": 2.5,
                "patch_r": 2.2,
            },
            "aux_bores": [],
        },
        "stage2_neck_yaw_to_pitch_link": {
            "support": {
                "name": "neck_yaw_front_horn",
                "normal": np.array([0.0, 0.0, 1.0]),
                "normals": s2_sup_norms,
                "centres": s2_sup_pts,
                "rocker_center": np.array([0.0, 5.0, 372.5]),
                "rocker_r": 3.0,
                "sleeve_ro": 1.75,
                "sleeve_ri": 1.10,
                "sleeve_len": 3.5,
                "patch_r": 2.2,
            },
            "load": {
                "name": "neck_pitch_front_horn",
                "normal": np.array([-1.0, 0.0, 0.0]),
                "normals": s2_load_norms,
                "centres": s2_load_pts,
                "rocker_center": np.array([-21.0, 5.0, 414.0]),
                "rocker_r": 3.0,
                "sleeve_ro": 1.75,
                "sleeve_ri": 1.10,
                "sleeve_len": 3.5,
                "patch_r": 2.2,
            },
            "aux_bores": [],
        },
        "stage3_neck_pitch_to_head_cradle": {
            "support": {
                "name": "neck_pitch_case_rear",
                "normal": np.array([0.0, 1.0, 0.0]),
                "normals": s3_sup_norms,
                "centres": s3_sup_pts,
                "sleeve_ro": 1.75,
                "sleeve_ri": 1.10,
                "sleeve_len": 2.5,
                "patch_r": 2.2,
            },
            "load": {       # 2 brass M3 heat-set inserts (T-9wjdqc), pressed from the tongue's front face, flush with both faces
                "name": "head_carrier_tongue",
                "normal": np.array([0.0, -1.0, 0.0]),
                "normals": s3_load_norms,
                "centres": s3_load_pts,
                "sleeve_ro": float(ins3["ro"]),
                "sleeve_ri": float(ins3["ri"]),
                "sleeve_len": float(ins3["length"]),
                "threaded": True,
                "patch_r": 2.6,
            },
            "aux_bores": [],
        },
        "stage4_head_sensor_compute_carrier": {
            "support": {    # no sleeve (T-5bqc3m K2): plain clearance hole, ISO 7380-1 M3 head on the rear face in a clearance cut through the ribs
                "name": "carrier_neck_root",
                "normal": np.array([0.0, 1.0, 0.0]),
                "normals": s4_sup_norms,
                "centres": s4_sup_pts,
                "no_sleeve": True,
                "sleeve_ro": float(root["hole_dia"]) / 2.0,
                "sleeve_ri": float(root["hole_dia"]) / 2.0,
                "sleeve_len": plate_t,
                "head_clear_r": float(root_head["dk"]) / 2.0 + float(root["head_clear_mm"]),
                "rib_window_half": float(root["rib_window_half_mm"]),
                "patch_r": 3.6,
            },
            "load": {       # no sleeve (T-5bqc3m K2): plain clearance hole, ISO 7092 washer on the front (counterbore + printed column where needed)
                "name": "compute_and_camera_mounts",
                "normal": np.array([0.0, 1.0, 0.0]),
                "normals": s4_load_norms,
                "centres": s4_load_pts,
                "no_sleeve": True,
                "sleeve_ro": float(comp["hole_dia"]) / 2.0,
                "sleeve_ri": float(comp["hole_dia"]) / 2.0,
                "sleeve_len": plate_t,
                "posts": HH.compute_posts(),
                "counterbore_r": float(comp["washer"]["d2"]) / 2.0 + float(comp["column"]["counterbore_clear_mm"]),
                "washer_r": float(comp["washer"]["d2"]) / 2.0,
                "column": comp["column"],
                "patch_r": 3.3,
            },
            "camera": {
                "name": "camera_posts",
                "normal": np.array([0.0, 1.0, 0.0]),
                "normals": s4_cam_norms,
                "centres": s4_cam_pts,
                "sleeve_ro": cm["ro"],
                "sleeve_ri": cm["ri"],
                "sleeve_len": max(cam_lens),
                "sleeve_lens": cam_lens,
                "sleeve_starts": cam_starts,
                "threaded": True,
                "column": HH.POSTS["camera"]["column"],
                "press_clear_mm": float(HH.POSTS["camera"]["press_clear_mm"]),
                "plate_rear": float(HH.plate_faces()[1]),
                "patch_r": 2.3,
            },
            "mic": {        # one insert in the wing's boss, pushed from the boss end (+x) up to the floor; the screw comes from the board side
                "name": "microphone_post",
                "normal": np.array([1.0, 0.0, 0.0]),
                "normals": [np.array([1.0, 0.0, 0.0])] * len(mic_pts),
                "centres": mic_pts,
                "sleeve_ro": HH.sleeve("mic")["ro"],
                "sleeve_ri": HH.sleeve("mic")["ri"],
                "sleeve_len": max(HH.sleeve_lens("mic")),
                "sleeve_lens": HH.sleeve_lens("mic"),
                "sleeve_starts": HH.sleeve_starts("mic"),
                "threaded": True,
                "wing_face": float(mic_rows[0]["face"]),
                "wing": HH.POSTS["mic"]["wing"],
                "boss": HH.POSTS["mic"]["boss"],
                "pin": dict(HH.POSTS["mic"]["pin"], centre=HH.pin_site()),
                "port": HH.POSTS["mic"]["port"],
                "patch_r": 2.3,
            },
            "aux_bores": [
                # 2x M2 radial head skin screw bores at X = ±44.0..45.5, Y = -14.9, Z = 451.0
                {"pt": np.array([45.5, -14.9, 451.0]), "normal": np.array([-1.0, 0.0, 0.0]), "ri": 1.05, "length": 1.8, "driver_dir": np.array([1.0, 0.0, 0.0])},
                {"pt": np.array([-45.5, -14.9, 451.0]), "normal": np.array([1.0, 0.0, 0.0]), "ri": 1.05, "length": 1.8, "driver_dir": np.array([-1.0, 0.0, 0.0])},
            ],
        },
    }


def build_stage_sleeves(stage_key: str) -> list[dict]:
    """Build the pressed-in parts of `stage_key`: 304 stainless steel compression sleeves (stages 1-3) and the brass heat-set inserts
    (stage 4 camera / mic posts M2, T-5bqc3m; stage 3 carrier tongue M3, T-9wjdqc); groups marked no_sleeve (stage 4 neck root, compute posts)
    have none."""
    from build123d import Plane, Solid

    specs = stage_interface_specs()[stage_key]
    out = []
    for role_key in [r for r in ("support", "load", "camera", "mic") if r in specs and not specs[r].get("no_sleeve")]:
        grp = specs[role_key]
        norms = grp.get("normals", [grp["normal"]] * len(grp["centres"]))
        ro = float(grp["sleeve_ro"])
        ri = float(grp["sleeve_ri"])
        for idx, (pt, d) in enumerate(zip(grp["centres"], norms)):
            h = float(grp["sleeve_lens"][idx]) if "sleeve_lens" in grp else float(grp["sleeve_len"])
            if "sleeve_starts" in grp:      # an insert that starts away from the site's plane (camera: inside the printed column)
                pt = pt + (float(grp["sleeve_starts"][idx]) - float(np.dot(pt, d))) * np.asarray(d, float)
            outer = cyl_along(pt, d, ro, h)
            inner = cyl_along(pt - 0.05 * d, d, ri, h + 0.10)
            slv = Solid((outer - inner).wrapped)
            if grp.get("countersink_dia"):      # 90 deg countersink for the countersunk head, from the sleeve's head end
                R0 = float(grp["countersink_dia"]) / 2.0
                depth = min(h, R0 - ri)
                cone = Solid.make_cone(R0 + 0.05, ri, depth + 0.05, Plane(origin=tuple(float(x) for x in pt - 0.05 * d), z_dir=tuple(float(x) for x in d)))
                slv = Solid((slv - cone).wrapped)
            name = f"{STAGE_PART_BASENAMES[stage_key]}__{grp['name']}_sleeve_{idx + 1}"
            vol = float(slv.volume)
            rho = 8.5 if grp.get("threaded") else MATERIALS["Steel_Sleeve"]["density_g_cm3"]     # brass heat-set insert / 304 sleeve
            mass_g = vol * (rho * 1e-3)
            out.append({
                "name": name,
                "stage": stage_key,
                "interface": grp["name"],
                "owning_link": STAGE_LINKS[stage_key],
                "solid": slv,
                "od_mm": round(ro * 2.0, 2),
                "id_mm": round(ri * 2.0, 2),
                "length_mm": round(h, 2),
                "volume_mm3": round(vol, 3),
                "mass_g": round(mass_g, 4),
            })
    return out


def solid_to_watertight_arrays(sol, tol: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate an OpenCascade Solid via `BRepMesh_IncrementalMesh`, weld shared vertices across faces,
    and verify watertight `manifold3d.Manifold` status (`Error.NoError`)."""
    import manifold3d as md
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    s0 = sol.solids()[0] if hasattr(sol, "solids") and len(sol.solids()) > 0 else sol
    for t_try, dig in [(tol, 5), (tol, 4), (0.025, 4), (0.065, 4), (0.03, 3)]:
        BRepTools.Clean_s(s0.wrapped)
        BRepMesh_IncrementalMesh(s0.wrapped, float(t_try), False, 0.12, True).Perform()
        vertices, faces = [], []
        for f in s0.faces():
            loc = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(f.wrapped, loc)
            if poly is None:
                continue
            off = len(vertices)
            tr = loc.Transformation()
            vertices.extend(
                [[p.X(), p.Y(), p.Z()] for p in [poly.Node(i).Transformed(tr) for i in range(1, poly.NbNodes() + 1)]]
            )
            for i in range(1, poly.NbTriangles() + 1):
                a, b, c = poly.Triangle(i).Get()
                faces.append(
                    [off + a - 1, off + c - 1, off + b - 1]
                    if f.wrapped.Orientation() == TopAbs_REVERSED
                    else [off + a - 1, off + b - 1, off + c - 1]
                )
        m = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=True)
        m.merge_vertices(digits_vertex=dig)
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        if not m.is_watertight:
            trimesh.repair.fill_holes(m)
            trimesh.repair.fix_normals(m)
        if m.is_watertight and m.volume < 0:
            m.invert()
        v_arr = np.asarray(m.vertices, dtype=np.float64)
        f_arr = np.asarray(m.faces, dtype=np.int64)
        q = md.Manifold(md.Mesh64(v_arr, f_arr.astype(np.uint64)))
        if q.status() == md.Error.NoError and q.volume() > 0.0:
            return v_arr, f_arr
    assert q.status() == md.Error.NoError and q.volume() > 0.0, f"Non-manifold tessellation: {q.status()} vol={q.volume()}"
    return v_arr, f_arr


def save_solid_npz(sol, npz_path: Path, tol: float = 0.05):
    """Save a verified watertight `manifold3d`-ready `.npz` mesh (`vertices`, `faces`) for `sol`."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    v_arr, f_arr = solid_to_watertight_arrays(sol, tol=tol)
    np.savez_compressed(npz_path, vertices=v_arr, faces=f_arr)


def rodrigues_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    th = math.radians(float(angle_deg))
    c, s = math.cos(th), math.sin(th)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def Homogeneous_about(origin: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    R = rodrigues_matrix(axis, angle_deg)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin - R @ origin
    return T


def neck_link_transform(link_name: str, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Return the 4x4 homogeneous world transform for `torso`, `neck_yaw`, or `head`."""
    if link_name == "torso":
        return np.eye(4)
    T_yaw = Homogeneous_about(P_NECK_YAW, AXIS_NECK_YAW, yaw_deg)
    if link_name == "neck_yaw":
        return T_yaw
    T_pitch = Homogeneous_about(P_NECK_PITCH, AXIS_NECK_PITCH, pitch_deg)
    if link_name == "head":
        return T_yaw @ T_pitch
    return np.eye(4)


NECK_YAW_PARTS = {
    "upper__neck_yaw_front_horn",
    "upper__neck_yaw_horn_M2x5_0",
    "upper__neck_yaw_horn_M2x5_1",
    "upper__one_piece_bent_neck_L",
    "upper__neck_pitch_front_horn",
    "upper__neck_pitch_horn_M2x5_0",
    "upper__neck_pitch_horn_M2x5_1",
}

TORSO_HIGH_PARTS = {
    "Common_power_carrier_mounting_r33",
    "ape_body_shell__torso_front_relief4_r33",
    "ape_body_shell__torso_back_relief4_r33",
    "upper__simple_continuous_rear_neck_cowl",
    "upper__cowl_M2_nut_0",
    "upper__cowl_M2_nut_1",
    "upper__cowl_M2x10_countersunk_0",
    "upper__cowl_M2x10_countersunk_1",
    "fixed_neck_ribbon_anchor_v21",
    "fixed_neck_ribbon_cap_v21",
    "fixed_neck_ribbon_pad_rear_v21",
    "fixed_neck_ribbon_pad_front_v21",
    "fixed_ribbon_M2x8_0_v21",
    "fixed_ribbon_M2_washer_0_v21",
    "fixed_ribbon_M2x8_1_v21",
    "fixed_ribbon_M2_washer_1_v21",
    "upper__neck_yaw_case_XM335",
    "upper__neck_root_M2_washer_0",
    "upper__neck_root_M2_washer_1",
    "upper__neck_root_M2_washer_2",
    "upper__neck_root_M2_washer_3",
    "upper__neck_root_M2x6_0",
    "upper__neck_root_M2x6_1",
    "upper__neck_root_M2x6_2",
    "upper__neck_root_M2x6_3",
    "neck_anchor_M3x8_0",           # T-jxshhv (neck_anchor.json; the length follows screw_plan.choose_length)
    "neck_anchor_M3x8_1",
    "IMU_ADC_carrier_v17",
    "Power_supervisor_PA12_open_tray",
    "Power_supervisor_source_board",
}


def classify_neck_head_link(part_dict: dict) -> str | None:
    """Classify a part from `assembly-parts.json` into `'torso'`, `'neck_yaw'`, `'head'`, or `None` (distant limb)."""
    nm = part_dict.get("name", "")
    b = part_dict.get("bounds_mm")
    if not b:
        return None
    if nm in NECK_YAW_PARTS:
        return "neck_yaw"
    if nm in TORSO_HIGH_PARTS:
        return "torso"
    # Exclude left/right arm parts (shoulder pitch/roll/yaw/elbow)
    if nm.startswith(("upper__L_", "upper__R_", "upper__A_L_", "upper__A_R_")):
        return "torso" if "pitch_C047_case" in nm or "clocked_pitch" in nm else None
    if b[1][2] < 325.0:
        return None
    if b[0][2] >= 387.0 and abs(0.5 * (b[0][0] + b[1][0])) <= 60.0:
        return "head"
    return "torso"


# ---- study load cases (moved from the retired per-study fem_evaluate.py); run_neck_head_topopt.py loads its topology optimisation with them
# Optimal FDM build-plate normal axis for each stage so primary bending & loop tension lie in-plane (XY)
PRINT_AXES = {
    "stage1_neck_root_cradle": "Y",            # XZ cradle & spine frame in XY plane
    "stage2_neck_yaw_to_pitch_link": "Y",      # XZ 90-degree L-bend & side webs in XY plane
    "stage3_neck_pitch_to_head_cradle": "Z",   # XY U-cradle loop around +X in XY plane
    "stage4_head_sensor_compute_carrier": "Y", # XZ carrier plate & ribs flat on build plate
}

HEAD_CG_MM = np.array([-4.0, -2.0, 445.0], dtype=float)

# 6-DOF wrenches defined at HEAD_CG_MM (`F_N` in N, `M_Nm` in N*m)
LOAD_CASES = {
    "upright_gravity_1g": {
        "title": "1g Upright Head Hold + Gaze Offset",
        "F_N": np.array([0.5, -1.8, -3.5]),
        "M_Nm": np.array([0.14, 0.06, 0.05]),
    },
    "dynamic_nod_turn_3g": {
        "title": "3g Dynamic Nod + Fast Yaw Turn",
        "F_N": np.array([4.5, -7.5, -11.0]),
        "M_Nm": np.array([0.48, 0.32, 0.42]),
    },
    "fall_recovery_head_6g": {
        "title": "6g Stumble / Fall Arrest Head Whip",
        "F_N": np.array([10.5, -16.0, -21.0]),
        "M_Nm": np.array([0.95, 0.68, 0.62]),
    },
    "adverse_neck_stall": {
        "title": "XM335 Dual-Axis Stall + Blocked Forehead Contact",
        "F_N": np.array([13.0, 14.5, -12.0]),
        "M_Nm": np.array([1.25, 0.85, 1.15]),
    },
}


def stage_case_wrench(stage_key: str, case_key: str, load_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transfer the 6-DOF head wrench at `HEAD_CG_MM` to `load_center` of `stage_key` (in N and N*mm)."""
    case = LOAD_CASES[case_key]
    F = np.asarray(case["F_N"], dtype=float)
    M_Nmm = np.asarray(case["M_Nm"], dtype=float) * 1000.0
    # Stage 1 shares torso neck root load with the right-hand (+X) 4x M2 flange of Common_power_carrier_mounting_r33
    scale = 0.55 if stage_key == "stage1_neck_root_cradle" else (0.75 if stage_key == "stage4_head_sensor_compute_carrier" else 1.0)
    F_s = F * scale
    r_arm_mm = (HEAD_CG_MM - load_center) * 0.45
    M_total_Nmm = M_Nmm * scale + np.cross(r_arm_mm, F_s)
    return F_s, M_total_Nmm
