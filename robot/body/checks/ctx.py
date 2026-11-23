"""One run's shared state: the manifest loaded once, the scope, neighbours, solids, poses, settings."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from checks import bodylib as BL
from checks.core import DATA, ROOT, R, kit, load_body, pose_set, screen_solid, sha, tick

# Body shells are voxel-filled closed meshes of several 100k faces: too heavy to screen as neighbours by default
# (fast_check assumption A-01 left them out too). --include-shells brings them in; a shell in the scope is still checked.


SETTING_BLOCKS = ("process", "expected_hardware", "fit_classes", "covers", "exact_fit", "loadpath_slices", "geometry_thresholds", "fem", "edges", "heat_source")


def part_settings_file():
    """data/part_settings.json, or LAB_PART_SETTINGS=<file> (with LAB_DEFINITIONS_ROOT: the settings of an alternate body)."""
    import os
    alt = os.environ.get("LAB_PART_SETTINGS")
    return Path(alt) if alt else DATA / "part_settings.json"


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
        from checks import bodylib as _BL
        from checks.relations import Relations
        self.rel = Relations(self.body, manifest)      # what each part is: definition, role table, geometry (never names)
        _BL.RELATIONS = self.rel
        self.scope = list(scope)
        self.poses_name = poses
        self.search_mm = search_mm
        self.exact_press = exact_press
        self.render_timeout = render_timeout
        self.out_dir = out_dir
        self.sol, self.mf = {}, {}
        self._poses = {}
        self._nb = None
        S = json.loads(part_settings_file().read_text())
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
        rel = self.rel
        cand = {n: p for n, p in body.parts.items() if n not in scope
                and (self.include_shells or not rel.is_shell(n))
                and not (rel.is_fastener(n) and not rel.is_sleeve_set(n))}

        def corners(p):
            b = np.array(p["bounds_mm"], float)
            return np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        C = {n: corners(p) for n, p in {**cand, **{s: body.parts[s] for s in scope}}.items()}
        near = set()
        for _, _, q in self.poses_for(scope):
            Tl = {}

            def box(n):
                lk = body.parts[n]["owning_link"]
                if lk not in Tl:
                    Tl[lk] = body.link_T(lk, q)
                c = C[n] @ Tl[lk][:3, :3].T + Tl[lk][:3, 3]
                return c.min(0), c.max(0)
            sb = [box(s) for s in scope]
            for n in cand:
                if n in near:
                    continue
                lo, hi = box(n)
                for slo, shi in sb:
                    if np.linalg.norm(np.maximum(0.0, np.maximum(slo - hi, lo - shi))) <= self.search_mm:
                        near.add(n); break
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
