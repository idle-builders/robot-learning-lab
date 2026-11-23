"""What the checks know about the parts of a body, without knowing any design (owner 2026-11-22: "the tests shouldn't know
about the design, that gets fed in via the config"; relations derived from the manifest and the CAD, a definition may
declare any of them).

Every answer comes, in this order, from:
  1. the part's body definition (robot/**/<builder>.bom.yaml, robot/body/bom/definitions.py): kind, spec.kind, joined_to
     (how: pressed = this sleeve set is pressed into that part), mirror_of;
  2. the role table data/roles.json (the manifest's role vocabulary: kind, classes);
  3. geometry and the joint tree of the manifest: the part a sleeve set's sleeves touch, the part whose x-mirrored box
     matches, the joint whose origin and axis mirror a joint, limbs as the subtrees under the first mirrored joint.
Never from a part, link or joint name. `source(n, what)` says where an answer came from; a derived answer can be fixed by
declaring it in the definition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROLES = json.loads((HERE / "data/roles.json").read_text())
GEO = ROLES["geometry"]


def _sample(m, k, seed=0):
    """k area-weighted surface points of mesh m, the same every run."""
    rng = np.random.default_rng(seed)
    area = m.area_faces
    i = rng.choice(len(m.faces), k, p=area / area.sum())
    u, v = rng.random(k), rng.random(k)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    t = m.triangles[i]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


class Relations:
    def __init__(self, body, manifest_path):
        self.body = body
        raw = json.loads(Path(manifest_path).read_text())
        self.raw = {p["name"]: p for p in raw["parts"]} if isinstance(raw.get("parts"), list) else dict(raw.get("parts", {}))
        self.joints = {j["name"]: j for j in raw.get("joints", [])}
        try:
            sys.path.insert(0, str(ROOT / "robot/body/bom"))
            import definitions as DEF
            self.claims = DEF.load().claims(list(self.raw))
        except Exception:  # noqa: BLE001  without readable definitions: role table and geometry
            self.claims = {}
        self._src, self._owner, self._mirror, self._jm = {}, {}, None, None

    # ------------------------------------------------------------------------------------------ what a part is
    def role(self, n):
        return (self.raw.get(n) or {}).get("role")

    def kind(self, n):
        """printed | purchased | hardware | included | made | other | None."""
        e = self.claims.get(n) or {}
        if e.get("kind"):
            self._src[(n, "kind")] = "definition"
            return e["kind"]
        k = ROLES["kind"].get(self.role(n))
        self._src[(n, "kind")] = "role table" if k else "unknown"
        return k

    def hardware_kind(self, n):
        """screw | nut | washer | spacer | threaded_sleeve | ... from the definition's spec.kind; None when it does not say."""
        k = ((self.claims.get(n) or {}).get("spec") or {}).get("kind")
        self._src[(n, "hardware_kind")] = "definition" if k else "unknown"
        return k

    def in_class(self, n, cls):
        return self.role(n) in ROLES["classes"][cls]["roles"]

    def is_sleeve_set(self, n):
        return self.in_class(n, "sleeve_set")

    def is_hardware(self, n):
        """Standard hardware (screws, nuts, washers, spacers, threaded sleeves, bearings ...). A sleeve set is hardware too;
        the checks measure it as part of the part it is pressed into (is_sleeve_set)."""
        return self.kind(n) == "hardware"

    def is_fastener(self, n):
        """Hardware that passes through or sits in a part by design (screws, nuts, washers, spacers, sleeve sets ...): never
        scoped, not a neighbour (a sleeve set is kept as one where a check says so); solid hardware such as bearings is not."""
        return self.is_hardware(n) and not self.in_class(n, "solid_hardware")

    def is_printed(self, n):
        return self.kind(n) == "printed"

    def is_shell(self, n):
        """A body shell / cover: the role class, or a definition that says what it covers."""
        e = self.claims.get(n) or {}
        return self.kind(n) == "printed" and (bool(e.get("covers")) or self.in_class(n, "shell"))

    # ------------------------------------------------------------------------------------------ sleeves
    def sleeve_owner(self, s):
        """The part a sleeve set is pressed into: the definition's joined_to {how: pressed}, else the part on the same link whose
        surface the sleeves' surface touches most (share of sleeve surface points within touch_mm)."""
        if s in self._owner:
            return self._owner[s]
        e = self.claims.get(s) or {}
        decl = [j["part"] for j in e.get("joined_to") or [] if str(j.get("how", "")).startswith("pressed")]
        if decl:
            self._src[(s, "sleeve_owner")] = "definition"
            self._owner[s] = decl[0]
            return decl[0]
        self._owner[s] = self._touching_owner(s)
        self._src[(s, "sleeve_owner")] = "geometry (sleeve surface touches it)" if self._owner[s] else "none found"
        return self._owner[s]

    def _touching_owner(self, s):
        from scipy.spatial import cKDTree
        P, body = self.body.parts, self.body
        if s not in P:
            return None
        sb = np.array(P[s]["bounds_mm"], float)
        try:
            sm = body.mesh(s)
        except Exception:  # noqa: BLE001
            return None
        pts = _sample(sm, 4000, 1) if len(sm.faces) else np.asarray(sm.vertices)
        best, score = None, 0.0
        for c, p in P.items():
            if c == s or p["owning_link"] != P[s]["owning_link"] or self.is_hardware(c):
                continue
            cb = np.array(p["bounds_mm"], float)
            if np.any(cb[0] > sb[1] + 1) or np.any(cb[1] < sb[0] - 1):
                continue
            try:
                cm = body.mesh(c)
            except Exception:  # noqa: BLE001
                continue
            d, _ = cKDTree(_sample(cm, 60000, 2) if len(cm.faces) else cm.vertices).query(pts, distance_upper_bound=GEO["touch_mm"])
            f = float(np.mean(np.isfinite(d)))
            if f > score:
                best, score = c, f
        return best

    def sleeve_sets_of(self, n):
        """The sleeve sets pressed into part n (declared or touching), among the sleeve sets on its link whose box meets its box."""
        P = self.body.parts
        if n not in P:
            return []
        nb = np.array(P[n]["bounds_mm"], float)
        out = []
        for s, p in P.items():
            if not self.is_sleeve_set(s) or p["owning_link"] != P[n]["owning_link"]:
                continue
            sb = np.array(p["bounds_mm"], float)
            if np.any(sb[0] > nb[1] + 1) or np.any(sb[1] < nb[0] - 1):
                continue
            if self.sleeve_owner(s) == n:
                out.append(s)
        return sorted(out)

    # ------------------------------------------------------------------------------------------ mirrors
    def mirror_of(self, n):
        """The part that is n mirrored at the x = mirror_plane_x_mm plane: the definition's mirror_of, else the part of the same
        kind and role whose x-mirrored box matches n's box within mirror_tol_mm. None for a part on the plane or without one."""
        e = self.claims.get(n) or {}
        if e.get("mirror_of"):
            self._src[(n, "mirror_of")] = "definition"
            return e["mirror_of"]
        if self._mirror is None:
            self._mirror = self._mirror_pairs()
        self._src[(n, "mirror_of")] = "geometry (x-mirrored box)" if n in self._mirror else "none"
        return self._mirror.get(n)

    def _mirror_pairs(self):
        P = self.body.parts
        x0 = GEO["mirror_plane_x_mm"]
        B = {n: np.array(p["bounds_mm"], float) for n, p in P.items()}
        by = {}
        for n in P:
            if not self.is_hardware(n):          # identical screws at one spot tie; only parts are compared with a mirror
                by.setdefault((self.role(n), self.kind(n)), []).append(n)
        out = {}
        for grp in by.values():
            for n in grp:
                b = B[n]
                if abs((b[0][0] + b[1][0]) / 2 - x0) < 1.0:
                    continue
                mb = np.array([[2 * x0 - b[1][0], b[0][1], b[0][2]], [2 * x0 - b[0][0], b[1][1], b[1][2]]])
                cand = sorted((float(np.abs(B[c] - mb).max()), c) for c in grp if c != n)
                if cand and cand[0][0] < GEO["mirror_tol_mm"]:
                    out[n] = cand[0][1]
        return out

    # ------------------------------------------------------------------------------------------ joints and limbs
    def joint_mirror(self, j):
        if self._jm is None:
            self._jm = joint_mirrors(self.joints)
        return self._jm.get(j, (None, None))

    def limb_of(self):
        return limbs(self.joints, {p["owning_link"] for p in self.body.parts.values()})

    def mirror_pose(self, q, joints):
        """q with the angles of `joints` copied, mirrored, onto their mirror joints (angle x sign): the same motion on the
        other side. Joints without a mirror are left as they are."""
        out = dict(q)
        for j in joints:
            m, s = self.joint_mirror(j)
            if m and j in q:
                out[m] = s * q[j]
        return out

    def source(self, n, what):
        return self._src.get((n, what))


def joint_mirrors(joints):
    """{joint: (mirror joint, angle sign)} from the joint table alone: the joint whose origin is the joint's origin mirrored at
    x = mirror_plane_x_mm and whose axis is parallel to its mirrored axis. A reflection reverses the sense of rotation: the
    sign is -1 when the axis is the mirrored axis, +1 when it points the other way."""
    x0, out = GEO["mirror_plane_x_mm"], {}
    J = dict(joints)
    for a, ja in J.items():
        oa, xa = np.asarray(ja.get("origin_world_mm", [0, 0, 0]), float), np.asarray(ja.get("axis_world_neutral", [0, 0, 1]), float)
        if abs(oa[0] - x0) < 1.0:
            continue
        mo, ma = np.array([2 * x0 - oa[0], oa[1], oa[2]]), np.array([-xa[0], xa[1], xa[2]])
        for b, jb in J.items():
            if b == a:
                continue
            ob, xb = np.asarray(jb.get("origin_world_mm", [0, 0, 0]), float), np.asarray(jb.get("axis_world_neutral", [0, 0, 1]), float)
            d = float(np.dot(ma, xb) / (np.linalg.norm(ma) * np.linalg.norm(xb) or 1))
            if np.linalg.norm(ob - mo) < 1.0 and abs(d) > 0.999:
                out[a] = (b, -1.0 if d > 0 else 1.0)
                break
    return out


def limbs(joints, links=()):
    """{link: limb root joint}: the subtree under the first joint, on the way out from the root link, that has a mirror joint
    (a leg, an arm); links under no such joint (torso, head) are in no limb. From the joint table alone."""
    J = dict(joints)
    jm = joint_mirrors(J)
    child = {j["child"]: j for j in J.values()}
    out = {}
    for lk in set(links) | set(child):
        path, cur = [], lk
        while cur in child:
            path.append(child[cur]["name"]); cur = child[cur]["parent"]
        for jn in reversed(path):
            if jn in jm:
                out[lk] = jn
                break
    return out
