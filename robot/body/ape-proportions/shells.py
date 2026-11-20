"""Shell pass, step 1: openings cut by the motion they have to allow (Idle T-czv2d8).

The torso shells (belly and back) lose exactly what the legs sweep through over the target hip ranges, and the boot
shell loses what the shin and ankle sweep through over the target ankle ranges, each with CLEARANCE mm. The cut-outs
are therefore as small as the motion allows, and the shell that remains around the pelvis is the 'briefs'.

Targets (owner intent: ranges are what the body allows; hip internals allow flexion to -100 and roll to +30):
  hip pitch -90..+40, hip roll -45..+25, hip yaw -40..+40 (grid of combined poses);
  ankle pitch -35..+27.5, ankle roll -25..+25.

Method (voxel sweep; mesh-union sweeps fragment into millions of triangles on near-coplanar copies):
  - each moving link's parts are voxelized at PITCH mm in the link's neutral frame and grown by CLEARANCE with a
    distance transform;
  - a PITCH grid over the target shells' region is looked up in every link grid at every pose; any hit marks the
    point as swept;
  - the swept mask is smoothed and turned into one surface (marching cubes), which is subtracted from each shell once.
Shells stay mesh-defined until the CAD pass.
"""
from __future__ import annotations

import itertools

import numpy as np
import trimesh

CLEARANCE = 2.0
PITCH = 0.75
HIP_FLEX = float(__import__("os").environ.get("APE_HIP_FLEX", -90.0))  # deepest hip flexion the openings allow
HIP = dict(pitch=np.arange(HIP_FLEX, 40.1, 5.0), roll=(-45.0, -30.0, -15.0, 0.0, 10.0, 20.0, 25.0),
           yaw=(-40.0, -20.0, 0.0, 20.0, 40.0))
HIP_LINKS = ("pitch_link", "roll_link", "shin")
TORSO_SHELLS = ("center_front_shell", "center_back_shell")
ANKLE = dict(pitch=np.arange(-35.0, 27.6, 2.5), roll=np.arange(-25.0, 25.1, 5.0))
ANKLE_LINKS = ("shin", "ankle_pitch")
SKIP = ("cut_material", "adhesive_allocation", "authored_harness")


def _mf(m):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


def _tm(x):
    r = x.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=True)


def _solid_voxels(m, lo, shape):
    """Voxels of a closed mesh on the grid (lo, shape): dense surface samples, then the enclosed cells filled.
    (trimesh's voxelizer subdivides detailed meshes down to the voxel size and needs several GB.)"""
    from scipy.ndimage import binary_fill_holes
    n = int(min(4e6, max(2e4, m.area / (0.4 * PITCH) ** 2)))
    pts = np.vstack([trimesh.sample.sample_surface(m, n, seed=0)[0], m.vertices])  # seeded: the same voxels every run
    idx = np.round((pts - lo) / PITCH).astype(np.int32)
    a = np.maximum(idx.min(0) - 1, 0); b = np.minimum(idx.max(0) + 2, shape)
    sub = np.zeros(b - a, bool)
    k = idx - a; ok = np.all((k >= 0) & (k < sub.shape), 1)
    sub[tuple(k[ok].T)] = True
    sub = binary_fill_holes(sub)
    out = np.zeros(shape, bool); out[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = sub
    return out


class LinkGrid:
    """A link's parts as an occupancy grid in its neutral frame, grown by CLEARANCE."""
    def __init__(self, body, F, link):
        from scipy.ndimage import distance_transform_edt
        meshes = []
        for n, p in body.parts.items():
            if p["owning_link"] != link or F.is_fastener(n, p) or p.get("role") in SKIP:
                continue
            m = body.mesh(n)
            meshes.append(m if m.is_watertight else m.convex_hull)
        self.empty = not meshes
        if self.empty:
            return
        pad = CLEARANCE + 3 * PITCH
        lo = np.min([m.bounds[0] for m in meshes], 0) - pad; hi = np.max([m.bounds[1] for m in meshes], 0) + pad
        shape = np.ceil((hi - lo) / PITCH).astype(int) + 1
        occ = np.zeros(shape, bool)
        for m in meshes:
            occ |= _solid_voxels(m, lo, shape)
        self.hit = distance_transform_edt(~occ) * PITCH <= CLEARANCE
        self.lo, self.shape = lo, shape
        self.corners = np.array(list(itertools.product(*zip(lo, lo + (shape - 1) * PITCH))))


class Sweep:
    """A PITCH grid over a target region (in the target link's neutral frame), marked where any link grid reaches."""
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, float); self.shape = np.ceil((np.asarray(hi) - self.lo) / PITCH).astype(int) + 1
        self.mask = np.zeros(self.shape, bool)

    def add(self, g, T):
        """T maps target-neutral coordinates into the link grid's neutral frame."""
        if g.empty:
            return
        Ti = np.linalg.inv(T)
        w = trimesh.transform_points(g.corners, Ti)
        a = np.clip(np.floor((w.min(0) - self.lo) / PITCH).astype(int), 0, self.shape)
        b = np.clip(np.ceil((w.max(0) - self.lo) / PITCH).astype(int) + 1, 0, self.shape)
        if np.any(b <= a):
            return
        ii = np.stack(np.meshgrid(*[np.arange(a[k], b[k]) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        p = self.lo + ii * PITCH
        q = p @ T[:3, :3].T + T[:3, 3]
        j = np.round((q - g.lo) / PITCH).astype(np.int32)
        ok = np.all((j >= 0) & (j < g.shape), 1)
        hit = np.zeros(len(ii), bool); hit[ok] = g.hit[tuple(j[ok].T)]
        self.mask[tuple(ii[hit].T)] = True

    def surface(self):
        """The swept region as one closed, smoothed mesh."""
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        f = gaussian_filter(np.pad(self.mask, 2).astype(np.float32), 0.6)
        v, faces, _, _ = marching_cubes(f, 0.5)
        v = self.lo + (v - 2) * PITCH
        return trimesh.Trimesh(v, faces[:, ::-1], process=True)


def _subtract(target, env):
    import manifold3d as mf
    e = _mf(env)
    if e.status() != mf.Error.NoError:
        raise ValueError(f"swept surface is not a valid manifold: {e.status()}")
    m = _tm(_mf(target) - e)
    big = [q for q in m.split(only_watertight=False) if (q.volume if q.is_watertight else 0) > 20.0]
    return trimesh.util.concatenate(big) if big else m


def apply(new, h):
    F = h["F"]; log = h["log"]; notes = []
    body = F.Body(new)
    P = {p["name"]: p for p in new["parts"]}
    # torso shells minus the swept legs (torso frame = neutral world)
    sh = {s: F.load_mesh(P[s]["mesh"]) for s in TORSO_SHELLS}
    lo = np.min([m.bounds[0] for m in sh.values()], 0) - 1.0; hi = np.max([m.bounds[1] for m in sh.values()], 0) + 1.0
    sw = Sweep(lo, hi); npose = 0
    for s in "RL":
        grids = {lk: LinkGrid(body, F, f"{s}_{lk}") for lk in HIP_LINKS}
        for pitch, roll, yaw in itertools.product(HIP["pitch"], HIP["roll"], HIP["yaw"]):
            q = {f"{s}_hip_pitch": pitch, f"{s}_hip_roll": roll, f"{s}_hip_yaw": yaw}; npose += 1
            for lk, g in grids.items():
                sw.add(g, np.linalg.inv(body.link_T(f"{s}_{lk}", q)))
    env = sw.surface()
    notes.append(f"hip sweep: {npose} poses, swept region {sw.mask.sum() * PITCH ** 3 / 1000:.0f} cm3 inside the shell box")
    for shell, sm in sh.items():
        if not sm.is_watertight:
            notes.append(f"{shell}: not watertight, skipped"); continue
        out = _subtract(sm, env)
        P[shell].update(h["save_mesh"](shell, out))
        P[shell]["native_step_unmodified"] = P[shell].pop("native_step", P[shell].get("native_step_unmodified")); P[shell].pop("native_step_sha256", None)
        P[shell]["mesh_note"] = "leg openings cut by the legs swept over the target hip ranges (shells.py, voxel sweep)"
        notes.append(f"{shell}: {sm.volume:.0f} -> {out.volume:.0f} mm3")
        log.append((shell, "leg openings cut by the swept hips"))
    # boot shells minus the swept shin and ankle, in the foot's own frame
    for s in "RL":
        boot = P[f"{s}_foot_boot_shell"]
        bm = F.load_mesh(boot["mesh"])
        if not bm.is_watertight:
            notes.append(f"{boot['name']}: not watertight, skipped"); continue
        grids = {lk: LinkGrid(body, F, f"{s}_{lk}") for lk in ANKLE_LINKS}
        sw = Sweep(bm.bounds[0] - 1.0, bm.bounds[1] + 1.0)
        for pitch, roll in itertools.product(ANKLE["pitch"], ANKLE["roll"]):
            q = {f"{s}_ankle_pitch": pitch, f"{s}_ankle_roll": roll}
            Tf = body.link_T(f"{s}_foot", q)
            for lk, g in grids.items():
                sw.add(g, np.linalg.inv(body.link_T(f"{s}_{lk}", q)) @ Tf)
        if not sw.mask.any():
            notes.append(f"{boot['name']}: nothing swept"); continue
        out = _subtract(bm, sw.surface())
        boot.update(h["save_mesh"](boot["name"], out))
        boot["mesh_note"] = "boot collar cut by the shin and ankle swept over the target ankle ranges (shells.py, voxel sweep)"
        notes.append(f"{boot['name']}: {bm.volume:.0f} -> {out.volume:.0f} mm3")
        log.append((boot["name"], "collar cut by the swept ankle"))
    new["shell_pass"] = dict(hip_targets_deg=dict(pitch=[HIP_FLEX, 40], roll=[-45, 25], yaw=[-40, 40]),
                             ankle_targets_deg=dict(pitch=[-35, 27.5], roll=[-25, 25]), clearance_mm=CLEARANCE,
                             voxel_mm=PITCH, method="voxel sweep + marching cubes, one subtraction per shell")
    return notes


def main():
    """Run the shell pass as its own step on a built manifest (keeps the build's memory out of the cuts).

      python3 infra/tools/memguard.py --limit-gb 12 -- .venv/bin/python -B robot/body/ape-proportions/shells.py \\
          artifacts/robot/body/ape-proportions/assembly-parts.json artifacts/robot/body/ape-proportions/shell-pass
    """
    import hashlib
    import json
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent; root = here.parents[2]
    sys.path.insert(0, str(here.parent / "motor-refit"))
    import fit_check as F
    src, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    new = json.load(open(src)); log = []

    def save_mesh(name, m):
        npz = parts / f"{name}.npz"
        np.savez_compressed(npz, vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, np.int64))
        return dict(mesh=str(npz.relative_to(root)), mesh_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                    mesh_units="mm", bounds_mm=[m.bounds[0].round(3).tolist(), m.bounds[1].round(3).tolist()])
    notes = apply(new, dict(F=F, save_mesh=save_mesh, log=log))
    new["shell_pass"]["input_manifest"] = str(src)
    new["shell_pass"]["input_manifest_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    json.dump(new, open(out / "assembly-parts.json", "w"), indent=1)
    json.dump(dict(schema="cove50.ape-proportions.shell-pass.v1", idle_task="T-czv2d8", notes=notes,
                   edits=[dict(part=n, edit=e) for n, e in log]), open(out / "report.json", "w"), indent=1)
    for nt in notes:
        print(nt)


if __name__ == "__main__":
    main()
