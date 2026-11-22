"""One run's shared state: the manifest loaded once, the scope, neighbours, solids, poses, settings."""
from __future__ import annotations

import json
import time

import numpy as np

from checks import bodylib as BL
from checks.core import DATA, ROOT, R, kit, load_body, pose_set, screen_solid, sha, tick

FASTENER_KEEP = "compression_limit_sleeves"
SPARSE = 0.25        # solid volume / bounds box volume under this: a shell or cover, whose box is mostly what it covers
SAMPLE_MM = 1.5      # surface sample spacing for the neighbour test of such parts


def trimesh_sample(m, k):
    """k area-weighted surface points of mesh m, the same every run."""
    rng = np.random.default_rng(0)
    area = m.area_faces
    i = rng.choice(len(m.faces), k, p=area / area.sum())
    u, v = rng.random(k), rng.random(k)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    t = m.triangles[i]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
# Body shells are voxel-filled closed meshes of several 100k faces: too heavy to screen as neighbours by default
# (fast_check assumption A-01 left them out too). --include-shells brings them in; a shell in the scope is still checked.
SHELL_ROLE = "body_shell"


SETTING_BLOCKS = ("process", "expected_hardware", "fit_classes", "covers", "exact_fit", "loadpath_slices", "geometry_thresholds", "fem", "edges", "heat_source")


def overlay_definitions(part_settings, manifest):
    """Per-part settings for the checks: the part's body definition (robot/**/<builder>.bom.yaml, its parts entry blocks
    process / expected_hardware / fit_classes / covers / exact_fit / loadpath_slices / geometry_thresholds / fem / edges)
    first, data/part_settings.json for what the definition does not say yet (migration, phase 2 of T-ghvncz). Returns
    (settings, source {part: {block: 'definition <file> parts[i]' | 'part_settings.json'}}, definition_of {part: (file, entry)})."""
    import sys as _sys
    from checks.core import ROOT as _ROOT
    out = json.loads(json.dumps(part_settings))
    src, dof = {}, {}
    try:
        _sys.path.insert(0, str(_ROOT / "robot/body/bom"))
        import definitions as DEF
        man = json.loads(open(manifest).read())
        names = [p["name"] for p in man["parts"]] if isinstance(man.get("parts"), list) else list(man.get("parts", {}))
        claims = DEF.load().claims(names)
    except Exception:  # noqa: BLE001  (no definitions readable: part_settings.json alone)
        claims = {}
    for n, e in claims.items():
        tag = f"{e['file']} parts[{e['index']}] line {e['line']}"
        dof[n] = (e["file"], tag)
        for k in SETTING_BLOCKS:
            if k in e:
                out.setdefault(n, {})[k] = e[k]
                src.setdefault(n, {})[k] = f"definition {tag}"
    for n, v in part_settings.items():
        if n.startswith("_"):
            continue
        for k in v:
            src.setdefault(n, {}).setdefault(k, "part_settings.json")
    return out, src, dof


class Ctx:
    def __init__(self, manifest, scope, poses="key", search_mm=5.0, exact_press=False, render_timeout=240.0, out_dir=None, include_shells=False):
        self.include_shells = include_shells
        self.manifest = manifest
        self.body = load_body(manifest)
        self.scope = list(scope)
        self.poses_name = poses
        self.search_mm = search_mm
        self.exact_press = exact_press
        self.render_timeout = render_timeout
        self.out_dir = out_dir
        self.sol, self.mf = {}, {}
        self._poses = {}
        self._nb = None
        S = json.loads((DATA / "part_settings.json").read_text())
        self._settings, self.settings_source, self.definition_of = overlay_definitions(S, self.manifest)
        self.process_override = {}      # run.py --process KEY=VALUE: applied to the process of every printed part (cross-evaluation)

    def definition(self, n):
        """(definition file, entry) that claims part n, or (None, None)."""
        return self.definition_of.get(n, (None, None))

    @property
    def parts(self):
        return self.body.parts

    def link(self, n):
        return self.body.parts[n]["owning_link"]

    def settings(self, n):
        s = self._settings
        out = json.loads(json.dumps(s.get("_defaults", {})))
        for k, v in s.get(n, {}).items():
            out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) else v
        if self.process_override and n in self.body.parts:
            from checks.process import is_printed
            if is_printed(self, n, out):
                out["process"] = {**out.get("process", {}), **self.process_override}
        return out

    @property
    def poses(self):
        return self.poses_for(self.scope)

    def poses_for(self, scope):
        """Pose set for these scoped parts: the arm / neck / torso region poses for parts on those links (core.pose_set), cached per link set."""
        links = tuple(sorted({self.body.parts[n]["owning_link"] for n in scope if n in self.body.parts}))
        if links not in self._poses:
            self._poses[links] = pose_set(self.body, self.poses_name, links)
        return self._poses[links]

    def solid(self, n):
        """(trimesh, how) of part n, built once; parts in the scope get the fine STEP tessellation."""
        if n not in self.sol:
            t0 = time.time()
            self.sol[n] = screen_solid(self.body, n, set(self.scope)); self.mf[n] = kit.mf(self.sol[n][0])
            tick("solids (scope)" if n in self.scope else "solids (others)", t0)
        return self.sol[n]

    def neighbours(self):
        """Neighbours of the scope (see neighbours_of), computed once."""
        if self._nb is None:
            t0 = time.time()
            self._nb = self.neighbours_of(self.scope)
            tick("neighbours", t0)
        return self._nb

    def neighbours_of(self, scope):
        """Parts that come within search_mm of a part of scope (bounds boxes, over the pose set) or touch it at standing.
        Fasteners are not neighbours (their sleeves are). Conservative: oriented bounds corners -> world AABB."""
        body = self.body
        scope = list(scope)
        cand = {n: p for n, p in body.parts.items() if n not in scope
                and (self.include_shells or p.get("role") != SHELL_ROLE)
                and not (BL.is_fastener(n, p) and p.get("role") != FASTENER_KEEP)}

        def corners(p):
            b = np.array(p["bounds_mm"], float)
            return np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        C = {n: corners(p) for n, p in {**cand, **{s: body.parts[s] for s in scope}}.items()}
        # a sparse scoped part (a shell or cover: its solid fills under SPARSE of its box) has a box that holds most of what it
        # covers; its box pulled the whole torso in as neighbours of the torso shells and a 3-check run passed 1800 s
        # (T-fdczae). For such a part a box hit is kept only if the candidate's box comes within search_mm (+ the sample
        # spacing) of the part's surface, sampled every SAMPLE_MM, at some pose.
        sparse = {}
        for s_ in scope:
            p = body.parts[s_]
            b = np.array(p["bounds_mm"], float); bv = float(np.prod(np.maximum(b[1] - b[0], 1e-3)))
            if p.get("volume_mm3") and p["volume_mm3"] < SPARSE * bv and bv > 5e4:
                try:
                    m = body.mesh(s_)
                    k = int(min(80000, max(2000, m.area / SAMPLE_MM ** 2)))
                    sparse[s_] = np.vstack([np.asarray(m.vertices, float), trimesh_sample(m, k)])
                except Exception:  # noqa: BLE001  without its mesh the part keeps the box test
                    pass
        near, box_hits = set(), {}
        for _, _, q in self.poses_for(scope):
            Tl = {}

            def T_of(n):
                lk = body.parts[n]["owning_link"]
                if lk not in Tl:
                    Tl[lk] = body.link_T(lk, q)
                return Tl[lk]

            def box(n):
                T = T_of(n)
                c = C[n] @ T[:3, :3].T + T[:3, 3]
                return c.min(0), c.max(0)
            sb = [(s_, box(s_)) for s_ in scope]
            for n in cand:
                if n in near:
                    continue
                lo, hi = box(n)
                for s_, (slo, shi) in sb:
                    if np.linalg.norm(np.maximum(0.0, np.maximum(slo - hi, lo - shi))) > self.search_mm:
                        continue
                    box_hits[n] = box_hits.get(n, 0) + 1
                    if s_ in sparse:
                        T = T_of(s_)
                        P = sparse[s_] @ T[:3, :3].T + T[:3, 3]
                        d = np.linalg.norm(np.maximum(0.0, np.maximum(lo - P, P - hi)), axis=1).min()
                        if d > self.search_mm + SAMPLE_MM * 2:
                            continue
                    near.add(n); break
        self.neighbour_stats = dict(sparse_scoped_parts=sorted(sparse), box_candidates=len(box_hits), neighbours=len(near),
                                    left_out_by_surface=len(set(box_hits) - near))
        return {n: cand[n] for n in sorted(near)}

    def locate(self, a, b, pose_name):
        """World point (mm) of the contact of a and b at a named pose: centroid of their overlap, else the midpoint of the closest
        vertex pair. None when the pose is not in the set."""
        q = next((q for nm, _, q in self.poses if nm == pose_name), None)
        if q is None:
            return None
        m = {}
        for n in (a, b):
            g = self.solid(n)[0].copy()
            g.apply_transform(self.body.link_T(self.link(n), q))
            m[n] = g
        try:
            inter = (kit.mf(m[a]) ^ kit.mf(m[b]))
            if inter.volume() > 1e-3:
                return np.round(kit.tm(inter).center_mass, 2).tolist()
            from scipy.spatial import cKDTree
            d, j = cKDTree(m[b].vertices).query(m[a].vertices)   # closest vertex pair (mesh vertices are < 1 mm apart)
            i = int(np.argmin(d))
            return np.round((m[a].vertices[i] + m[b].vertices[j[i]]) / 2, 2).tolist()
        except Exception:  # noqa: BLE001
            return None
