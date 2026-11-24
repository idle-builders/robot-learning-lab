"""Shared source pins, paths, material models, compression sleeves, 3D bevels, and integrated hard-shell mounts for the 4-stage Arm FDM connector study."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import manifold3d as md

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("build-arm-connectors")   # configured values: robot/body/arm/build_arm_connectors.bom.yaml config:

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
STUDY = Path(__file__).resolve().parent
ART = ROOT / "artifacts/robot/body/arm"
EC_MANIFEST = ROOT / "artifacts/cove50-electronics-integration/r1-20261026/assembly-parts.json"
ARM_SHELL_DIR = ROOT / "artifacts/cove50-arm-shell/r5-surface-20261029"
HAND_DIR = ROOT / "artifacts/cove50-engineering-competition/hand-radical-20261026/A/twin-crawl-wrist-seats-r1"

# Exact baseline STEP paths for the 4 arm connector stages (Right arm)
BASELINE_STEPS = {
    "stage1_shoulder_pitch_to_roll": ROOT / "artifacts/cove50-engineering-competition/upper-body-20261023/B/shell-fit-r4/stage-frames9/R_pitch_to_roll_one_piece.step",
    "stage2_shoulder_roll_to_yaw": ROOT / "artifacts/cove50-engineering-competition/upper-body-20261023/B/shell-fit-r4/stage-frames10/R_roll_to_yaw_one_piece.step",
    "stage3_upper_arm_yaw_to_elbow": ROOT / "artifacts/cove50-arm-shell/r4-roll70-onepiece-20261029/native/R_S_yaw_to_elbow.step",
    "stage4_forearm_elbow_to_wrist": ROOT / "artifacts/cove50-engineering-competition/upper-body-20261023/B/shell-fit-r4/stage-frames11/R_forearm_wrist_one_piece.step",
}

STAGE_LINKS = CFG["STAGE_LINKS"]

STAGE_PART_BASENAMES = CFG["STAGE_PART_BASENAMES"]

# Complete set of legacy sheet-metal frames, plates, tubes, carriers, and carrier screws replaced by the 4-stage FDM arm connectors
REPLACED_ARM_PARTS = set(CFG["REPLACED_ARM_PARTS"])

# Exact kinematic frames and axes on the Right arm
N_YAW = np.array([-0.1643989873053573, 0.0, 0.9863939238321437], dtype=float)
V_YAW = np.array([0.9863939238321437, 0.0, 0.1643989873053573], dtype=float)
W_YAW = np.array([0.0, 1.0, 0.0], dtype=float)
A_YAW = np.column_stack([N_YAW, V_YAW, W_YAW])
P_PITCH = np.array([52.0, 9.0, 352.0], dtype=float)
P_ROLL = np.array([112.0, 3.0, 360.0], dtype=float)
P_YAW = np.array([134.0, -10.0, 306.0], dtype=float)
O_YAW_CASE = np.array([134.49319696191608, -10.0, 303.04081822850355], dtype=float)
P_ELBOW = np.array([145.0, -3.0, 252.0], dtype=float)
P_WRIST_ORIGIN = np.array([138.5, 1.0, 172.5], dtype=float)

_A20 = math.radians(20.0)
ROT_ROLL = np.column_stack([
    [0.0, 1.0, 0.0],
    [math.cos(_A20), 0.0, math.sin(_A20)],
    [math.sin(_A20), 0.0, -math.cos(_A20)],
])


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


MATERIALS = CFG["MATERIALS"]


def fillet_2d_face(f, r_default: float = 2.0, skip_xy=(), r_inner: float | None = None):
    """Fillet all non-tangent 2D vertices of an outer/inner planar Face before extrusion."""
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


def bevel_3d_edges(sol, r_outer: float = 0.42, r_inner: float = 0.75, min_cyl_r: float = 0.95, skip_fn=None):
    """Apply 3D OpenCascade edge fillets/bevels to every structural 3D edge of a Solid,
    using smaller rounding (`r_outer`) on outside convex edges and larger rounding
    (`r_inner`) on inside concave pocket/step edges, while preserving precision
    fastener bores/counterbores.
    """
    from build123d import Solid
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
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
        np.asarray(V_YAW, dtype=float),
        np.asarray(N_YAW, dtype=float),
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
                        s_try.tessellate(0.05)
                    return m.Shape()
        except Exception:
            pass
        return None

    # Fast path: try all contours together in 1 pass
    sh_all = _try_build(contours, check_tess=True)
    if sh_all is not None:
        return Solid(sh_all)

    # Pre-filter contours individually so any contour that fails alone is excluded up-front
    solo_valid: list[tuple[float, list]] = []
    for r_orig, eds in contours:
        if _try_build([(r_orig, eds)], check_tess=False) is not None:
            solo_valid.append((r_orig, eds))
        elif r_outer < r_orig and _try_build([(r_outer, eds)], check_tess=False) is not None:
            solo_valid.append((r_outer, eds))

    sh_solo = _try_build(solo_valid, check_tess=True)
    if sh_solo is not None:
        return Solid(sh_solo)

    # Chunked greedy accumulation (chunk_size = 6) over pre-validated contours
    good: list[tuple[float, list]] = []
    chunk_size = 6
    for start_idx in range(0, len(solo_valid), chunk_size):
        chunk = solo_valid[start_idx : start_idx + chunk_size]
        if _try_build(good + chunk, check_tess=False) is not None:
            good.extend(chunk)
        else:
            for r_orig, eds in chunk:
                if _try_build(good + [(r_orig, eds)], check_tess=False) is not None:
                    good.append((r_orig, eds))
                elif r_outer < r_orig and _try_build(good + [(r_outer, eds)], check_tess=False) is not None:
                    good.append((r_outer, eds))

    sh_best = _try_build(good, check_tess=True)
    return Solid(sh_best if sh_best is not None else cur)


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


def stage_interface_specs() -> dict:
    """Return exact hole centers, per-hole outward normals, and sleeve specifications for all 4 Right-arm stages,
    exposing all dual-sided / dual-cheek C047 motor case PA2 mounting bosses in addition to the main horns and wrist pad."""
    # Stage 1: R_shoulder_pitch_to_roll (4x M3 Pitch Horn + 6x PA2 Dual-Cheek Roll Case = 10 sleeves)
    s1_horn_pts = [np.array([70.625, 9.0 + u, 352.0 + v]) for u, v in [(-7.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0)]]
    s1_horn_norms = [np.array([1.0, 0.0, 0.0]) for _ in s1_horn_pts]

    s1_case_rear_pts = [
        P_ROLL + ROT_ROLL @ np.array([17.575, u, v])
        for u in [-10.25, 10.25]
        for v in [8.3, 32.75]
    ]
    s1_case_pts = s1_case_rear_pts
    s1_case_norms = [np.array([0.0, 1.0, 0.0])] * 4

    # Stage 2: R_shoulder_roll_to_yaw (4x M3 Roll Horn + 4x PA2 Yaw Case = 8 sleeves)
    s2_horn_pts = [
        P_ROLL + ROT_ROLL @ np.array([-18.625, u, v])
        for u, v in [(-7.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0)]
    ]
    s2_horn_norms = [np.array([0.0, -1.0, 0.0]) for _ in s2_horn_pts]

    s2_case_pts = [
        P_YAW + A_YAW @ np.array([17.575, u, v])
        for u in [-10.25, 10.25]
        for v in [8.3, 32.75]
    ]
    s2_case_norms = [N_YAW.copy()] * 4

    # Stage 3: R_upper_arm_yaw_to_elbow (4x M3 Yaw Horn + 4x M3 Elbow Horn = 8 sleeves)
    s3_yaw_origin_inner = P_YAW - 18.625 * N_YAW
    s3_yaw_pts = [
        s3_yaw_origin_inner + u * V_YAW + v * W_YAW
        for u, v in [(-7.0, 0.0), (0.0, -7.0), (0.0, 7.0), (7.0, 0.0)]
    ]
    s3_yaw_norms = [-N_YAW.copy() for _ in s3_yaw_pts]
    s3_elbow_pts = [
        np.array([163.625, -3.0 + u, 252.0 + v])
        for u, v in [(-7.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0)]
    ]
    s3_elbow_norms = [np.array([1.0, 0.0, 0.0]) for _ in s3_elbow_pts]

    # Stage 4: R_forearm_elbow_to_wrist (6x PA2 Dual-Cheek Elbow Case + 4x M3 Wrist Pad = 10 sleeves)
    s4_case_medial_pts = [
        np.array([127.425, y, z])
        for y in [-13.25, 7.25]
        for z in [219.25, 243.70]
    ]
    s4_case_lateral_pts = [
        np.array([160.425, y, 219.25])
        for y in [-13.25, 7.25]
    ]
    s4_case_pts = s4_case_medial_pts + s4_case_lateral_pts
    s4_case_norms = [np.array([-1.0, 0.0, 0.0])] * 4 + [np.array([1.0, 0.0, 0.0])] * 2

    s4_wrist_pts = [
        np.array([x, y, 171.0])
        for x in [135.0, 142.0]
        for y in [-3.0, 5.0]
    ]
    s4_wrist_norms = [np.array([0.0, 0.0, 1.0]) for _ in s4_wrist_pts]

    return {
        "stage1_shoulder_pitch_to_roll": {
            "support": {
                "name": "pitch_main_horn",
                "normal": np.array([1.0, 0.0, 0.0]),
                "normals": s1_horn_norms,
                "plane_normal_outward": np.array([-1.0, 0.0, 0.0]),
                "centres": s1_horn_pts,
                "rocker_center": np.array([70.625, 9.0, 352.0]),
                "rocker_r": 3.0,
                "sleeve_ro": 2.25,
                "sleeve_ri": 1.65,
                "sleeve_len": 3.0,
                "patch_r": 2.8,
            },
            "load": {
                "name": "roll_motor_case",
                "normal": np.array([0.0, 1.0, 0.0]),
                "normals": s1_case_norms,
                "plane_normal_outward": np.array([0.0, -1.0, 0.0]),
                "centres": s1_case_pts,
                "sleeve_ro": 1.90,
                "sleeve_ri": 1.20,
                "sleeve_len": 3.0,
                "patch_r": 2.1,
            },
        },
        "stage2_shoulder_roll_to_yaw": {
            "support": {
                "name": "roll_main_horn",
                "normal": np.array([0.0, -1.0, 0.0]),
                "normals": s2_horn_norms,
                "plane_normal_outward": np.array([0.0, 1.0, 0.0]),
                "centres": s2_horn_pts,
                "rocker_center": np.array([112.0, -15.625, 360.0]),
                "rocker_r": 3.0,
                "sleeve_ro": 2.25,
                "sleeve_ri": 1.65,
                "sleeve_len": 3.0,
                "patch_r": 2.8,
            },
            "load": {
                "name": "yaw_motor_case",
                "normal": N_YAW.copy(),
                "normals": s2_case_norms,
                "plane_normal_outward": -N_YAW.copy(),
                "centres": s2_case_pts,
                "sleeve_ro": 1.75,
                "sleeve_ri": 1.20,
                "sleeve_len": 3.0,
                "patch_r": 2.0,
            },
        },
        "stage3_upper_arm_yaw_to_elbow": {
            "support": {
                "name": "yaw_main_horn",
                "normal": -N_YAW.copy(),
                "normals": s3_yaw_norms,
                "plane_normal_outward": N_YAW.copy(),
                "centres": s3_yaw_pts,
                "rocker_center": s3_yaw_origin_inner.copy(),
                "rocker_r": 3.0,
                "sleeve_ro": 2.25,
                "sleeve_ri": 1.65,
                "sleeve_len": 3.0,
                "patch_r": 2.8,
            },
            "load": {
                "name": "elbow_main_horn",
                "normal": np.array([1.0, 0.0, 0.0]),
                "normals": s3_elbow_norms,
                "plane_normal_outward": np.array([-1.0, 0.0, 0.0]),
                "centres": s3_elbow_pts,
                "rocker_center": np.array([163.625, -3.0, 252.0]),
                "rocker_r": 3.0,
                "sleeve_ro": 2.25,
                "sleeve_ri": 1.65,
                "sleeve_len": 3.0,
                "patch_r": 2.8,
            },
        },
        "stage4_forearm_elbow_to_wrist": {
            "support": {
                "name": "elbow_motor_case",
                "normal": np.array([-1.0, 0.0, 0.0]),
                "normals": s4_case_norms,
                "plane_normal_outward": np.array([1.0, 0.0, 0.0]),
                "centres": s4_case_pts,
                "sleeve_ro": 1.75,
                "sleeve_ri": 1.20,
                "sleeve_len": 3.0,
                "patch_r": 2.0,
            },
            "load": {
                "name": "wrist_landing_pad",
                "normal": np.array([0.0, 0.0, 1.0]),
                "normals": s4_wrist_norms,
                "plane_normal_outward": np.array([0.0, 0.0, -1.0]),
                "centres": s4_wrist_pts,
                "sleeve_ro": 2.25,
                "sleeve_ri": 1.65,
                "sleeve_len": 3.0,
                "patch_r": 2.8,
            },
        },
    }


def build_stage_sleeves(stage_key: str, side: str = "R") -> list[dict]:
    """Build all pressed-in 304 stainless steel compression sleeves for `stage_key` on `side` ('R' or 'L')."""
    from build123d import Plane, Solid

    specs = stage_interface_specs()[stage_key]
    out = []
    for role_key in ("support", "load"):
        grp = specs[role_key]
        norms = grp.get("normals", [grp["normal"]] * len(grp["centres"]))
        ro = float(grp["sleeve_ro"])
        ri = float(grp["sleeve_ri"])
        h = float(grp["sleeve_len"])
        for idx, (pt, d) in enumerate(zip(grp["centres"], norms)):
            outer = cyl_along(pt, d, ro, h)
            inner = cyl_along(pt - 0.05 * d, d, ri, h + 0.10)
            slv = Solid((outer - inner).wrapped)
            if side == "L":
                slv = Solid(slv.mirror(Plane.YZ).wrapped)
            name = f"{side}_{STAGE_PART_BASENAMES[stage_key]}__{grp['name']}_sleeve_{idx + 1}"
            vol = float(slv.volume)
            mass_g = vol * (MATERIALS["Steel_Sleeve"]["density_g_cm3"] * 1e-3)
            out.append({
                "name": name,
                "stage": stage_key,
                "side": side,
                "interface": grp["name"],
                "owning_link": f"{side}_{STAGE_LINKS[stage_key]}",
                "solid": slv,
                "od_mm": round(ro * 2.0, 2),
                "id_mm": round(ri * 2.0, 2),
                "length_mm": round(h, 2),
                "volume_mm3": round(vol, 3),
                "mass_g": round(mass_g, 4),
            })
    return out


def cut_stage_sleeve_bores(body_r, stage_key: str):
    """Cut all sleeve outer-diameter receiving bores and central rocker clearance bores from a Right-arm stage body."""
    from build123d import Solid

    specs = stage_interface_specs()[stage_key]
    out = body_r
    for role_key in ("support", "load"):
        grp = specs[role_key]
        norms = grp.get("normals", [grp["normal"]] * len(grp["centres"]))
        ro = float(grp["sleeve_ro"])
        ri = float(grp["sleeve_ri"])
        h = float(grp["sleeve_len"])
        for pt, d in zip(grp["centres"], norms):
            # Cut exact sleeve seat (length h) at outer radius ro, and through-clearance bore at ri beyond h
            out = out - cyl_along(pt - 0.02 * d, d, ro, h + 0.04)
            out = out - cyl_along(pt - 2.0 * d, d, ri, h + 12.0)
        if "rocker_center" in grp:
            rc = grp["rocker_center"]
            rr = float(grp["rocker_r"])
            d0 = grp["normal"]
            out = out - cyl_along(rc - 2.0 * d0, d0, rr, h + 10.0)
    return Solid(out.solids()[0].wrapped)


# Exact Right-arm integrated hard-shell mount specifications (4 forearm mounts on R arm = 8 bilateral)
RIGHT_ARM_SHELL_MOUNTS = [
    {
        "id": "fore_front_187",
        "stage": "stage4_forearm_elbow_to_wrist",
        "shell": "fore_front",
        "axis": (0.0, -1.0, 0.0),
        "center": (145.0, -19.40, 187.0),
        "post_span": (-23.00, -13.50),
        "post_r": 4.0,
        "slot_dir": (0.0, 0.0, 1.0),
        "screw_src": "fore_front_M3x8_187",
        "screw_dx": 0.0,
    },
    {
        "id": "fore_front_178",
        "stage": "stage4_forearm_elbow_to_wrist",
        "shell": "fore_front",
        "axis": (0.0, -1.0, 0.0),
        "center": (145.0, -15.90, 178.0),
        "post_span": (-18.80, -11.50),
        "post_r": 4.0,
        "slot_dir": (0.0, 0.0, -1.0),
        "screw_src": "fore_front_M3x8_178",
        "screw_dx": 0.0,
    },
    {
        "id": "fore_back_187",
        "stage": "stage4_forearm_elbow_to_wrist",
        "shell": "fore_back",
        "axis": (0.0, 1.0, 0.0),
        "center": (145.0, 13.40, 187.0),
        "post_span": (7.50, 17.00),
        "post_r": 4.0,
        "slot_dir": (0.0, 0.0, 1.0),
        "screw_src": "fore_back_M3x8_187",
        "screw_dx": 0.0,
    },
    {
        "id": "fore_back_178",
        "stage": "stage4_forearm_elbow_to_wrist",
        "shell": "fore_back",
        "axis": (0.0, 1.0, 0.0),
        "center": (145.0, 9.90, 178.0),
        "post_span": (5.50, 12.30),
        "post_r": 4.0,
        "slot_dir": (0.0, 0.0, -1.0),
        "screw_src": "fore_back_M3x8_178",
        "screw_dx": 0.0,
    },
]


def make_captive_nut_pocket(
    center_pt: tuple[float, float, float],
    axis_d: tuple[float, float, float],
    slot_dir_z: float = 1.0,
    h: float = 2.65,
    slot_len: float = 8.2,
):
    """Create a 2D-filleted U-shaped hex-slot captive M3 nut pocket centered at `center_pt` along `axis_d` (±X or ±Y).
    The M3 nut has across-flats 5.5 mm (circumradius 3.175 mm) and thickness 2.40 mm centered at `center_pt`.
    The pocket has width 5.84 mm (half-width 2.92 mm), thickness `h = 2.65 mm`, and opens along `slot_dir_z * Z`.
    """
    from build123d import Polygon

    cx, cy, cz = map(float, center_pt)
    dx, dy, _ = map(float, axis_d)
    sz = 1.0 if slot_dir_z >= 0.0 else -1.0

    # U-shaped hex pocket in (u, z_rel): flat sides at u = ±2.82, hex V-bottom at z_rel = -3.25 * sz, open slot to +slot_len * sz
    pts_2d = [
        (2.82, sz * slot_len),
        (-2.82, sz * slot_len),
        (-2.82, -sz * 1.63),
        (0.0, -sz * 3.25),
        (2.82, -sz * 1.63),
    ]
    if abs(dy) > 0.5:
        pts_xz = [(cx + u, cz + v) for u, v in pts_2d]
        f_xz = fillet_2d_face(
            Polygon(*pts_xz, align=None).faces()[0],
            0.55,
            skip_xy=[(cx + 2.82, cz + sz * slot_len), (cx - 2.82, cz + sz * slot_len)],
            r_inner=0.55,
        )
        return extrude_xy_to_y(f_xz, cy - 0.5 * h, cy + 0.5 * h)
    else:
        pts_yz = [(cy + u, cz + v) for u, v in pts_2d]
        f_yz = fillet_2d_face(
            Polygon(*pts_yz, align=None).faces()[0],
            0.55,
            skip_xy=[(cy + 2.82, cz + sz * slot_len), (cy - 2.82, cz + sz * slot_len)],
            r_inner=0.55,
        )
        return extrude_xy_to_x(f_yz, cx - 0.5 * h, cx + 0.5 * h)


def make_m3_hex_nut_solid(center_pt: tuple[float, float, float], axis_d: tuple[float, float, float], h: float = 2.40):
    """Build a standard ISO 4032 M3 hex nut Solid (5.5 mm across flats, Ø3.0 mm bore, thickness 2.40 mm) centered at `center_pt`."""
    from build123d import Circle, Polygon, Pos, Solid

    cx, cy, cz = map(float, center_pt)
    dx, dy, _ = map(float, axis_d)
    r_circ = 5.5 / math.sqrt(3.0)  # 3.1754 mm
    # Vertices oriented so flat sides are at u = ±2.75 mm and points are at z = ±3.1754 mm
    hex_uv = [
        (r_circ * math.cos(math.radians(30 + 60 * k)), r_circ * math.sin(math.radians(30 + 60 * k)))
        for k in range(6)
    ]
    if abs(dy) > 0.5:
        pts_xz = [(cx + u, cz + v) for u, v in hex_uv]
        f_xz = (Polygon(*pts_xz, align=None) - Pos(cx, cz) * Circle(1.50)).faces()[0]
        return Solid(extrude_xy_to_y(f_xz, cy - 0.5 * h, cy + 0.5 * h).wrapped)
    else:
        pts_yz = [(cy + u, cz + v) for u, v in hex_uv]
        f_yz = (Polygon(*pts_yz, align=None) - Pos(cy, cz) * Circle(1.50)).faces()[0]
        return Solid(extrude_xy_to_x(f_yz, cx - 0.5 * h, cx + 0.5 * h).wrapped)


def add_shoulder_shell_mounts(body_r):
    """Integrate all 4 Right-shoulder hard-shell mounting posts (2 screws on `R_shoulder_front`, 2 screws on `R_shoulder_back`),
    Ø3.4 mm through-bores, and U-shaped captive M3 nut slots directly into Stage 1 (`R_shoulder_pitch_to_roll`),
    replacing the separate `R_shoulder_carrier` block and its 4 extra PA2x10 frame fasteners:
      1. `shoulder_front_339` at (Y = -7.0, Z = 339.0), X in [84.80, 91.35], nut center X = 88.40
      2. `shoulder_back_339` at (Y = 15.0, Z = 339.0), X in [84.80, 91.30], nut center X = 88.40
      3. `shoulder_front_351` at (Y = -10.0, Z = 351.0), X in [133.20, 139.25], nut center X = 136.95
      4. `shoulder_back_353` at (Y = 19.0, Z = 353.0), X in [131.80, 137.75], nut center X = 135.55
    """
    from build123d import Circle, Polygon, Pos, Solid

    # 1. Proximal bridge & posts at X in [84.80, 91.35] spanning Y in [-7.0, 15.0] at Z = 339.0
    # Connects directly into the Stage 1 bridge web (`X in [75, 100], Y in [0, 20], Z in [333, 358]`)
    f_prox_front_yz = (
        Polygon((-7.0, 335.0), (5.0, 335.0), (5.0, 343.0), (-7.0, 343.0), align=None)
        + Pos(-7.0, 339.0) * Circle(4.0)
    ).clean().faces()[0]
    post_f339 = extrude_xy_to_x(f_prox_front_yz, 84.80, 91.35)

    f_prox_back_yz = (
        Polygon((6.0, 335.0), (15.0, 335.0), (15.0, 343.0), (6.0, 343.0), align=None)
        + Pos(15.0, 339.0) * Circle(4.0)
    ).clean().faces()[0]
    post_b339 = extrude_xy_to_x(f_prox_back_yz, 84.80, 91.30)

    # 2. Lateral roll-case outrigger arm & posts for `shoulder_front_351` (Y=-10, Z=351) and `shoulder_back_353` (Y=19, Z=353)
    # The roll motor case occupies X in [98.0, 133.12], Y in [-14.125, 22.175], Z in [325.0, 358.5].
    # We wrap around the +X end of the roll case (`X >= 133.20`) from the Stage 1 roll-case flange (`Y in [20.575, 23.575]`).
    # First: a structural collar on the +Y roll-case flange face (`Y in [20.575, 23.575]`) extending from the top-right PA2 boss
    # `(124.47, 355.71)` & bottom-right PA2 boss `(132.83, 332.73)` out to `X = 137.75, Z in [331.0, 357.0]`:
    flange_ext_xz = fillet_2d_face(
        Polygon(
            (122.0, 357.0),
            (137.75, 357.0),
            (137.75, 348.5),
            (135.2, 331.0),
            (128.0, 331.0),
            align=None,
        ).faces()[0],
        1.5,
        skip_xy=[(137.75, 357.0), (137.75, 348.5)],
    )
    flange_ext = extrude_xy_to_y(flange_ext_xz, 20.575, 23.575)

    # Second: `shoulder_back_353` post at (Y = 19.0, Z = 353.0), X in [133.20, 137.75], fused to `flange_ext` (Y in [20.575, 23.575])
    f_b353_yz = (
        Polygon((19.0, 349.0), (23.575, 349.0), (23.575, 357.0), (19.0, 357.0), align=None)
        + Pos(19.0, 353.0) * Circle(4.0)
    ).clean().faces()[0]
    post_b353 = extrude_xy_to_x(f_b353_yz, 133.20, 137.75)

    # Third: C-strap along +X end of roll case (`X in [133.30, 136.80]`, clear of roll case X_max = 133.12)
    # bridging from `Y = 23.0, Z in [349.0, 356.5]` across to `Y = -10.0, Z in [347.0, 355.0]`, plus `shoulder_front_351` post (`X in [133.30, 139.25]`)
    f_strap_yz = (
        Polygon((-10.0, 347.0), (23.0, 349.0), (23.0, 356.5), (-10.0, 355.0), align=None)
        + Pos(-10.0, 351.0) * Circle(4.0)
        + Pos(19.0, 353.0) * Circle(4.0)
    ).clean().faces()[0]
    strap_x = extrude_xy_to_x(f_strap_yz, 133.30, 136.80)

    f_f351_yz = (
        Polygon((-10.0, 347.0), (-5.5, 347.0), (-5.5, 355.0), (-10.0, 355.0), align=None)
        + Pos(-10.0, 351.0) * Circle(4.0)
    ).clean().faces()[0]
    post_f351 = extrude_xy_to_x(f_f351_yz, 133.30, 139.25)

    out = body_r + post_f339 + post_b339 + flange_ext + post_b353 + strap_x + post_f351

    # Cut Ø3.4 mm (r = 1.70 mm) through-bores and U-shaped captive M3 nut slots for all 4 shoulder mounts
    out = Solid(out.solids()[0].wrapped)
    for m in RIGHT_ARM_SHELL_MOUNTS:
        if m["stage"] != "stage1_shoulder_pitch_to_roll":
            continue
        cx, cy, cz = m["center"]
        x0, x1 = m["post_span"]
        out = Solid((out - cyl_along((x0 - 1.5, cy, cz), (1.0, 0.0, 0.0), 1.70, (x1 - x0) + 3.0)).solids()[0].wrapped)
        out = Solid((out - make_captive_nut_pocket((cx, cy, cz), m["axis"], slot_dir_z=m["slot_dir"][2])).solids()[0].wrapped)

    return out


def add_forearm_shell_mounts(body_r):
    """Integrate all 4 Right-forearm hard-shell mounting posts (2 screws on `R_fore_front`, 2 screws on `R_fore_back`),
    Ø3.4 mm through-bores, and U-shaped captive M3 nut slots directly into Stage 4 (`R_forearm_elbow_to_wrist`),
    replacing the separate `R_fore_carrier` block and its 2 extra M3x10 frame fasteners:
      1. `fore_front_187` at (X = 145.0, Z = 187.0), Y in [-23.00, -13.50], nut center Y = -19.40
      2. `fore_front_178` at (X = 145.0, Z = 178.0), Y in [-18.80, -11.50], nut center Y = -15.90
      3. `fore_back_187` at (X = 145.0, Z = 187.0), Y in [7.50, 17.00], nut center Y = 13.40
      4. `fore_back_178` at (X = 145.0, Z = 178.0), Y in [5.50, 12.30], nut center Y = 9.90
    """
    from build123d import Circle, Polygon, Pos, Solid

    # Lateral mounting keel at X in [136.5, 147.2], Z in [174.6, 190.6] connecting the Stage 4 forearm spine
    # (`X in [127.425, 144.0], Y in [-14.0, 8.0], Z in [171.0, 195.0]`) to the 4 anterior/posterior shell posts at X = 145.0:
    f_keel_xz = fillet_2d_face(
        Polygon((136.5, 174.6), (147.2, 174.6), (147.2, 190.6), (134.5, 190.6), align=None).faces()[0],
        1.4,
    )
    keel_core = extrude_xy_to_y(f_keel_xz, -14.20, 8.20)

    # Individual posts extending to their exact zero-overlap planar faces against `R_fore_front` and `R_fore_back`
    f_p187_xz = fillet_2d_face(
        Polygon((141.0, 183.2), (148.6, 183.2), (148.6, 190.8), (141.0, 190.8), align=None).faces()[0],
        1.4,
    )
    post_ff187 = extrude_xy_to_y(f_p187_xz, -23.00, -13.50)
    post_fb187 = extrude_xy_to_y(f_p187_xz, 7.50, 17.00)

    f_p178_xz = fillet_2d_face(
        Polygon((141.0, 174.6), (148.6, 174.6), (148.6, 181.6), (141.0, 181.6), align=None).faces()[0],
        1.4,
    )
    post_ff178 = extrude_xy_to_y(f_p178_xz, -18.80, -11.50)
    post_fb178 = extrude_xy_to_y(f_p178_xz, 5.50, 12.30)

    out = Solid((body_r + keel_core + post_ff187 + post_fb187 + post_ff178 + post_fb178).solids()[0].wrapped)

    # Cut Ø3.4 mm (r = 1.70 mm) through-bores and U-shaped captive M3 nut slots for all 4 forearm mounts
    for m in RIGHT_ARM_SHELL_MOUNTS:
        if m["stage"] != "stage4_forearm_elbow_to_wrist":
            continue
        cx, cy, cz = m["center"]
        y0, y1 = m["post_span"]
        out = Solid((out - cyl_along((cx, y0 - 1.5, cz), (0.0, 1.0, 0.0), 1.70, (y1 - y0) + 3.0)).solids()[0].wrapped)
        out = Solid((out - make_captive_nut_pocket((cx, cy, cz), m["axis"], slot_dir_z=m["slot_dir"][2])).solids()[0].wrapped)

    return out


def make_m3x8_screw_solid(tip_pt: tuple[float, float, float], axis_to_head: tuple[float, float, float]):
    """Build an analytical ISO 7380 M3x8 button-head screw Solid (shank Ø3.0 x 8.0 mm, head Ø5.7 x 1.65 mm with 2.0 mm hex socket)."""
    from build123d import Solid

    p0 = np.asarray(tip_pt, dtype=float)
    d = np.asarray(axis_to_head, dtype=float)
    d = d / np.linalg.norm(d)
    shank = cyl_along(p0, d, 1.50, 8.05)
    head = cyl_along(p0 + 8.00 * d, d, 2.85, 1.65)
    socket = cyl_along(p0 + 8.65 * d, d, 1.05, 1.10)
    return Solid((shank + head - socket).wrapped)


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


def ensure_arm_shell_nut_assets() -> dict:
    """Generate and cache the 16 bilateral captive M3 nut STEP/mesh files (replacing the 16 brass heat-set inserts)
    and the 4 bilateral `shoulder_front_M3x8_351` / `shoulder_back_M3x8_353` screw STEP/mesh files in `ART / 'shell_hardware'`."""
    from build123d import Plane, Solid, export_step
    from infra.storage.blob_store import ensure

    mp = read_json(ARM_SHELL_DIR / "manufactured-parts.json")
    for p in mp["parts"]:
        for k in ("lossless_mesh", "native_step"):
            if k in p and isinstance(p[k], dict) and "path" in p[k]:
                ensure(ROOT / p[k]["path"])

    out_dir = ART / "shell_hardware"
    out_dir.mkdir(parents=True, exist_ok=True)
    nuts = []
    screws = []

    for m in RIGHT_ARM_SHELL_MOUNTS:
        nut_r = make_m3_hex_nut_solid(m["center"], m["axis"])
        nut_l = Solid(nut_r.mirror(Plane.YZ).wrapped)
        link_base = STAGE_LINKS[m["stage"]]
        for side, sol in (("R", nut_r), ("L", nut_l)):
            name = f"{side}_{m['id']}_M3_nut"
            step_p = out_dir / f"{name}.step"
            npz_p = out_dir / f"{name}.npz"
            export_step(sol, str(step_p))
            save_solid_npz(sol, npz_p, tol=0.04)
            nuts.append({
                "name": name,
                "side": side,
                "mount_id": m["id"],
                "stage": m["stage"],
                "owning_link": f"{side}_{link_base}",
                "step_path": step_p,
                "npz_path": npz_p,
                "solid": sol,
            })

        # For the two lateral shoulder mounts (`351` and `353`), provide M3x8 screw STEPs and NPZs seated at the exact shell counterbore
        if "m3x8_template" in m:
            _, cy, cz = m["center"]
            tip_x = 135.384 if m["id"] == "shoulder_front_351" else 134.076
            scr_r = make_m3x8_screw_solid((tip_x, cy, cz), (1.0, 0.0, 0.0))
            scr_l = Solid(scr_r.mirror(Plane.YZ).wrapped)
            new_base = m["screw_src"].replace("M3x6", "M3x8")
            for side, sol in (("R", scr_r), ("L", scr_l)):
                sname = f"{side}_{new_base}"
                s_step = out_dir / f"{sname}.step"
                s_npz = out_dir / f"{sname}.npz"
                export_step(sol, str(s_step))
                save_solid_npz(sol, s_npz, tol=0.04)
                screws.append({
                    "name": sname,
                    "replaces": f"{side}_{m['screw_src']}",
                    "side": side,
                    "mount_id": m["id"],
                    "owning_link": f"{side}_{link_base}",
                    "step_path": s_step,
                    "npz_path": s_npz,
                    "solid": sol,
                })

    return {"dir": out_dir, "nuts": nuts, "screws": screws}


# ---- helpers the arm study builders use (moved from the retired per-study fit_check.py / fem_evaluate.py)
def rot_mat(axis, deg: float) -> np.ndarray:
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def fk(joints: list[dict], q: dict[str, float]) -> dict[str, np.ndarray]:
    T = {"torso": np.eye(4)}
    pending = list(joints)
    while pending:
        for j in list(pending):
            if j["parent"] in T:
                R = rot_mat(j["axis_world_neutral"], q.get(j["name"], 0.0))
                p = np.asarray(j["origin_world_mm"], dtype=float)
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = p - R @ p
                T[j["child"]] = T[j["parent"]] @ M
                pending.remove(j)
    return T


def npz_to_mf(npz_path: Path) -> md.Manifold:
    z = np.load(npz_path)
    q = md.Manifold(md.Mesh64(np.asarray(z["vertices"], dtype=np.float64), np.asarray(z["faces"], dtype=np.uint64)))
    assert q.status() == md.Error.NoError and q.volume() > 0.0, f"Invalid manifold {npz_path}: {q.status()} vol={q.volume()}"
    return q


def cyl_mf(p0, d, r: float, length: float, n: int = 36) -> md.Manifold:
    d_arr = np.asarray(d, dtype=float)
    d_arr = d_arr / np.linalg.norm(d_arr)
    # Build cylinder along +Z then rotate/translate to p0 + t*d_arr
    c = md.Manifold.cylinder(float(length), float(r), float(r), n)
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(d_arr, z_axis):
        R = np.eye(3)
    elif np.allclose(d_arr, -z_axis):
        R = rot_mat([1.0, 0.0, 0.0], 180.0)
    else:
        v = np.cross(z_axis, d_arr)
        ang = math.degrees(math.acos(np.clip(float(np.dot(z_axis, d_arr)), -1.0, 1.0)))
        R = rot_mat(v, ang)
    M = np.zeros((3, 4), dtype=float)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(p0, dtype=float)
    return c.transform(M)


def build_poses() -> list[tuple[str, dict[str, float]]]:
    poses = [("neutral", {})]
    for a in (-60.0, -30.0, 45.0, 90.0, 150.0):
        poses.append((f"shoulder_pitch_{int(a)}", {"R_shoulder_pitch": a}))
    for a in (-15.0, 15.0, 35.0, 55.0, 70.0):
        poses.append((f"shoulder_roll_{int(a)}", {"R_shoulder_roll": a}))
    for a in (-60.0, -30.0, 30.0, 60.0):
        poses.append((f"shoulder_yaw_{int(a)}", {"R_shoulder_yaw": a}))
    for a in (25.0, 55.0, 90.0, 110.0):
        poses.append((f"elbow_{int(a)}", {"R_elbow": a}))
    for yaw in (-60.0, -30.0, 30.0, 60.0):
        for elb in (55.0, 90.0, 110.0):
            poses.append((f"yaw_{int(yaw)}_elbow_{int(elb)}", {"R_shoulder_yaw": yaw, "R_elbow": elb}))
    for p, r, y, e in [
        (-20.0, 10.0, -45.0, 60.0),
        (-20.0, 70.0, 45.0, 110.0),
        (15.0, 35.0, -60.0, 90.0),
        (15.0, 70.0, 60.0, 110.0),
        (75.0, 10.0, -45.0, 110.0),
        (75.0, 35.0, 45.0, 90.0),
        (75.0, 70.0, -45.0, 60.0),
        (75.0, 70.0, 45.0, 110.0),
        (120.0, 70.0, 0.0, 110.0),
        (-45.0, -15.0, 60.0, 90.0),
    ]:
        poses.append((f"coupled_p{int(p)}_r{int(r)}_y{int(y)}_e{int(e)}", {
            "R_shoulder_pitch": p,
            "R_shoulder_roll": r,
            "R_shoulder_yaw": y,
            "R_elbow": e,
        }))
    return poses


ELAST = load_elasticity()
VON_MISES = ELAST.reference_namespace()["von_mises"]


def build_material_hooke(mat_key: str, layer_normal_axis: int = 0) -> np.ndarray:
    m = MATERIALS[mat_key]
    if "E_MPa" in m:
        return ELAST.isotropic(m["E_MPa"], m["nu"])
    E_in = m["E_inplane_MPa"]
    E_lay = m["E_layer_MPa"]
    E_vec = [E_in, E_in, E_in]
    E_vec[layer_normal_axis] = E_lay
    return orthotropic_matrix_MPa(E_vec[0], E_vec[1], E_vec[2], nu=m["nu"], G=m["G_MPa"])


# Study load cases (N, N mm at the stage load patch); run_arm_topopt.py loads its topology optimisation with them.
ARM_LOAD_WRENCHES = BOM_CONFIG.unfrozen(CFG["ARM_LOAD_WRENCHES"])
