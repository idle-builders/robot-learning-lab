"""Body model helpers shared by the checks and the body builders: manifest meshes with placement, forward kinematics from
joints[] (Body.link_T), part preparation for overlap / distance (Prepped, measure), fastener detection, per-joint pose sets.
The body model every check and builder imports as `from checks import bodylib`."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from infra.storage.blob_store import ensure  # noqa: E402


def src(path) -> Path:
    """An input file, restored from the bucket by its recorded (or pre-restructure) path when not on disk."""
    rc = sys.modules.get("checks.resultcache")     # recording a check's inputs (checks/resultcache.py), when loaded
    if rc is not None:
        rc.note(ROOT / path)                # also the STEPs OCC / gmsh read in C++
    return ensure(ROOT / path)


RELATIONS = None      # the run's checks.relations.Relations (set by checks.ctx.Ctx): what a part is, from its definition


def load_mesh(path: str) -> trimesh.Trimesh:
    if str(path).endswith(".npz"):
        z = np.load(src(path))
        return trimesh.Trimesh(np.asarray(z["vertices"], float), np.asarray(z["faces"]), process=True)
    return trimesh.load(src(path), force="mesh", process=True)


def rot_about(origin, axis, deg) -> np.ndarray:
    T = trimesh.transformations.rotation_matrix(math.radians(deg), np.asarray(axis, float), np.asarray(origin, float))
    return T


class Body:
    def __init__(self, manifest: dict):
        self.parts = {p["name"]: p for p in manifest["parts"] if p.get("mesh")}
        self.joints = {j["name"]: j for j in manifest["joints"]}
        self.child_joint = {j["child"]: j for j in manifest["joints"]}
        # the root link: the manifest's root_link, else the one link that is no joint's child (the joint tree, never a name)
        roots = sorted({j["parent"] for j in manifest["joints"]} - {j["child"] for j in manifest["joints"]})
        self.root = manifest.get("root_link") or (roots[0] if len(roots) == 1 else None)
        self._mesh = {}

    def mesh(self, name):
        if name not in self._mesh:
            m = load_mesh(self.parts[name]["mesh"])
            off = self.parts[name].get("placement_offset_mm")
            if off:
                m.apply_translation(off)
            self._mesh[name] = m
        return self._mesh[name]

    def limb_of(self) -> dict:
        """{link: limb root joint}: the subtree under the first mirrored joint on the way out from the root (the two legs and
        two arms; pelvis, torso and head are on no limb), from the joint table (checks.relations.limbs), not joint names.
        Parts on two different limbs can be posed apart."""
        if getattr(self, "_limb", None) is None:
            from checks.relations import limbs
            self._limb = limbs(self.joints, {p["owning_link"] for p in self.parts.values()})
        return self._limb

    def link_T(self, link, q: dict) -> np.ndarray:
        T = np.eye(4)
        chain = []
        while link in self.child_joint:
            j = self.child_joint[link]
            chain.append(j)
            link = j["parent"]
        for j in reversed(chain):
            T = T @ rot_about(j["origin_world_mm"], j["axis_world_neutral"], q.get(j["name"], 0.0))
        return T


def to_mf(m: trimesh.Trimesh):
    import manifold3d as mf
    M = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    return M if M.status() == mf.Error.NoError else None


def samples(m: trimesh.Trimesh, spacing: float, cap: int = 60000) -> np.ndarray:
    n = int(min(cap, max(1500, m.area / spacing ** 2)))
    return np.vstack([trimesh.sample.sample_surface(m, n, seed=1)[0], m.vertices[:: max(1, len(m.vertices) // 5000)]])


class Prepped:
    """A part in its neutral world frame (= its link frame at q = 0): manifold, surface tree, bounds."""
    def __init__(self, m: trimesh.Trimesh):
        self.mesh = m
        self.mf = to_mf(m)
        self.tree = cKDTree(samples(m, 0.7))
        self.bounds = m.bounds


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def is_fastener(name, p):
    """Hardware that sits in or passes through a part by design (screws, nuts, washers, spacers, sleeve sets): the part's body
    definition (kind), else the role table (checks/data/roles.json). Never the name (it called 16 parts hardware on a regex)."""
    if RELATIONS is not None and name in RELATIONS.raw:
        return RELATIONS.is_fastener(name)
    from checks.relations import ROLES
    return ROLES["kind"].get(p.get("role")) == "hardware" and p.get("role") not in ROLES["classes"]["solid_hardware"]["roles"]


def measure(new_mf, new_pts, new_bounds, T_rel, part: Prepped, margin):
    """Overlap volume (mm^3) and minimum surface distance (mm) of the moved new body against a fixed part."""
    c = np.array([[new_bounds[i][0], new_bounds[j][1], new_bounds[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    cb = apply(T_rel, c)
    if np.any(cb.min(0) > part.bounds[1] + margin) or np.any(cb.max(0) < part.bounds[0] - margin):
        return None
    pts = apply(T_rel, new_pts)
    clear = float(part.tree.query(pts, k=1)[0].min())
    vol, method = 0.0, "manifold"
    if clear < 1.5:
        if part.mf is not None and new_mf is not None:
            vol = float((new_mf.transform(T_rel[:3, :4]) ^ part.mf).volume())
        else:
            method = "points-in-closed-mesh"
            try:
                inside = part.mesh.contains(pts[:: max(1, len(pts) // 3000)]) if part.mesh.is_watertight else np.zeros(1, bool)
                vol = float("nan") if inside.any() else 0.0
            except Exception:
                vol = float("nan")
    if vol > 0.01 or (vol != vol):
        clear = 0.0
    return vol, clear, method


def pose_set(body, jname, step):
    """([{joint: deg}], adjacent joints) of joint jname: its range_deg in steps, each neighbouring joint over its range at jname's ends
    and middle, and the pairwise range corners. The checks' Body carries the declared ranges (checks.core.declare_ranges). A joint
    without a range_deg is never posed over an invented one: jname without one gives no poses, a neighbour without one is left out;
    both are added to body.unranged_in_poses."""
    j = body.joints[jname]
    parent, child = j["parent"], j["child"]
    unranged = body.__dict__.setdefault("unranged_in_poses", set())
    rng = j.get("range_deg")
    if not rng:
        unranged.add(jname)
        return [], []
    own = list(np.arange(rng[0], rng[1] + 1e-6, step))
    if rng[1] not in own:
        own.append(rng[1])
    poses = [{jname: float(a)} for a in own]
    adj = [jj for jj in body.joints.values() if jj.get("active") is not False and jj["name"] != jname and
           (jj["child"] in (parent, child) or jj["parent"] in (parent, child))]
    unranged |= {jj["name"] for jj in adj if not jj.get("range_deg")}
    adj = [jj for jj in adj if jj.get("range_deg")]
    for jj in adj:
        r2 = jj["range_deg"]
        for a2 in list(np.arange(r2[0], r2[1] + 1e-6, step * 2)) + [r2[1]]:
            for a1 in (rng[0], 0.5 * (rng[0] + rng[1]), rng[1]):
                poses.append({jname: float(a1), jj["name"]: float(a2)})
    # pairwise extremes of adjacent joints together (flexion corners)
    for x in range(len(adj)):
        for y in range(x + 1, len(adj)):
            ra, rb = adj[x]["range_deg"], adj[y]["range_deg"]
            for a in ra:
                for b in rb:
                    for a1 in (rng[0], rng[1]):
                        poses.append({jname: float(a1), adj[x]["name"]: float(a), adj[y]["name"]: float(b)})
    return poses, adj
