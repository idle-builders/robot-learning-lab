"""Arm load envelope: the wrench a hand contact force puts on an arm connector, over the arm's poses and every force direction.

A load case {"hand_envelope": {...}} in a part's FEM setup (data/part_settings.json) stands for a set of wrenches, not one:

  force_N        the case force at the hand tip (e.g. 4 kg x 1.5 g = 58.84 N)
  hand_parts     part names or globs whose combined box gives the hand tip: x / y centre, lowest z (knuckle / fingertip contact)
  stage_link     the link the connector belongs to (its wrench is expressed in that link's neutral frame, i.e. the manifest frame the part is meshed in)
  step_deg       pose grid step over each joint's manifest range (ends included), default 15
  directions     force directions sampled on a half sphere (a force and its opposite give the same von Mises), default 48
  stall_factor   joint torque allowed as a multiple of the servo's 12 V stall torque (motor map), default 1.0
  cone_deg       optional: keep only directions within this angle of the line from the hand tip to the chain's first joint (the shoulder):
                 a push through the hand with floor friction mu = tan(cone_deg); without it every direction counts

The chain is every joint from the torso to the hand's link. Its most proximal joint is held at 0: turning it turns hand, axes and the
stage together, and the direction sphere is turned with them, so it adds no new wrench. Every other joint of the chain runs over its range.
For each pose and direction d the hand force is F = min(force_N, stall_factor * stall_j / |a_j . ((p_hand - o_j) x d)|) over the chain's joints j:
a servo cannot hold more than its stall, so beyond it the arm gives way at that joint instead of loading the structure further. The wrench at the
stage's origin is (F d, (p_hand - origin) x F d), turned back into the stage link's neutral frame. The arm's own weight is not added (about
0.3 kg against the case force); fall impacts that backdrive a servo beyond its stall are outside these cases (stall_factor shows the sensitivity).
"""
from __future__ import annotations

import fnmatch
import json
import math

import numpy as np

from checks.core import ROOT

DEFAULT_STEP_DEG = 15.0
DEFAULT_DIRECTIONS = 48


def _rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def half_sphere(n):
    """n unit directions spread evenly over the half sphere z >= 0 (Fibonacci lattice); together with their opposites they cover the sphere."""
    i = np.arange(n) + 0.5
    z = i / n
    r = np.sqrt(1 - z * z)
    phi = math.pi * (3 - math.sqrt(5)) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def stall_torques_Nmm():
    """Joint -> 12 V stall torque (N mm) from the motor map named by robot/model/current-model.json."""
    cur = json.load(open(ROOT / "robot/model/current-model.json"))
    mm = None
    for v in cur.values():
        if isinstance(v, dict) and isinstance(v.get("motor_map"), dict):
            mm = v["motor_map"]["path"]
    mm = json.load(open(ROOT / (mm or "robot/model/motor-map/motor-map.json")))
    tiers = mm["tiers"]
    return {j: 1000.0 * float(tiers[t]["stall_torque_Nm_12V"]) for j, t in mm["joints"].items() if t in tiers}


def hand_tip(body, patterns):
    names = [n for n in body.parts if any(fnmatch.fnmatch(n, p) for p in patterns)]
    if not names:
        raise ValueError(f"hand_envelope: no parts match {patterns}")
    b = np.array([body.parts[n]["bounds_mm"] for n in names], float)
    lo, hi = b[:, 0].min(axis=0), b[:, 1].max(axis=0)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]), names


def chain_to(body, link):
    """Joints from the root to the link, proximal first."""
    out = []
    while link in body.child_joint:
        j = body.child_joint[link]; out.append(j); link = j["parent"]
    return out[::-1]


def envelope(body, spec, origin):
    """(W (n, 6) wrenches N / N mm at origin in the stage's neutral frame, labels (n dicts), summary dict)."""
    p_hand0, hand_names = hand_tip(body, spec["hand_parts"])
    hand_link = body.parts[hand_names[0]]["owning_link"]
    # the hand's link: the deepest link among the hand parts that every other hand part's link chain passes through; fingers hang off the forearm
    links = {body.parts[n]["owning_link"] for n in hand_names}
    for l in links:
        if all(l in [j["child"] for j in chain_to(body, m)] + [m] for m in links):
            hand_link = l
    chain = chain_to(body, hand_link)
    names = [j["name"] for j in chain]
    stage_chain = [j["name"] for j in chain_to(body, spec["stage_link"])]
    if names[:len(stage_chain)] != stage_chain:
        raise ValueError(f"hand_envelope: stage link {spec['stage_link']} is not on the chain to {hand_link}")
    stall = stall_torques_Nmm()
    fac = float(spec.get("stall_factor", 1.0))
    lim = np.array([fac * stall.get(n, np.inf) for n in names])
    step = float(spec.get("step_deg", DEFAULT_STEP_DEG))
    grids = []
    for k, j in enumerate(chain):
        if k == 0:
            grids.append(np.array([0.0])); continue
        lo, hi = j["range_deg"]
        n = max(2, int(math.ceil((hi - lo) / step)) + 1)
        grids.append(np.linspace(lo, hi, n))
    D = half_sphere(int(spec.get("directions", DEFAULT_DIRECTIONS)))
    F_case = float(spec["force_N"])
    origin = np.asarray(origin, float)
    Ws, labels = [], []
    capped = 0
    by_joint = {n: 0 for n in names}
    mesh_q = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, len(chain))
    for q in mesh_q:
        R, t = np.eye(3), np.zeros(3)            # running transform x -> R x + t of the current link (proximal joints applied)
        ax_w, o_w, R_stage, t_stage = [], [], None, None
        for k, j in enumerate(chain):
            o, a = np.asarray(j["origin_world_mm"], float), np.asarray(j["axis_world_neutral"], float)
            a = a / np.linalg.norm(a)
            ax_w.append(R @ a); o_w.append(R @ o + t)
            Rj = _rot(a, q[k])
            R, t = R @ Rj, R @ (o - Rj @ o) + t
            if j["child"] == spec["stage_link"]:
                R_stage, t_stage = R.copy(), t.copy()
        if R_stage is None:
            raise ValueError("hand_envelope: stage link not reached")
        p = R @ p_hand0 + t
        org_w = R_stage @ origin + t_stage
        Dp = D
        if spec.get("cone_deg") is not None:
            u = o_w[0] - p; u = u / np.linalg.norm(u)
            Dp = D[np.abs(D @ u) >= math.cos(math.radians(float(spec["cone_deg"]))) - 1e-12]
        tau = np.array([[abs(float(ax_w[k] @ np.cross(p - o_w[k], d))) for k in range(len(chain))] for d in Dp])   # (nd, nj) per unit force
        with np.errstate(divide="ignore"):
            fj = np.where(tau > 1e-9, lim[None, :] / tau, np.inf)
        F = np.minimum(F_case, fj.min(axis=1))
        jb = fj.argmin(axis=1)
        for i, d in enumerate(Dp):
            f_w = F[i] * d
            w = np.concatenate([R_stage.T @ f_w, R_stage.T @ np.cross(p - org_w, f_w)])
            Ws.append(w)
            cap = bool(F[i] < F_case - 1e-9)
            capped += cap
            if cap:
                by_joint[names[jb[i]]] += 1
            labels.append(dict(pose_deg={n: round(float(v), 1) for n, v in zip(names[1:], q[1:])}, direction_stage_frame=np.round(R_stage.T @ d, 3).tolist(),
                               force_N=round(float(F[i]), 2), capped_by=names[jb[i]] if cap else None,
                               hand_lever_mm=round(float(np.linalg.norm(np.cross(p - org_w, d))), 1)))
    W = np.array(Ws)
    Fm = np.linalg.norm(W[:, :3], axis=1); Mm = np.linalg.norm(W[:, 3:], axis=1)
    summary = dict(wrenches=len(W), poses=len(mesh_q), directions_half_sphere=len(D), cone_deg=spec.get("cone_deg"), case_force_N=F_case, stall_factor=fac,
                   stall_torque_Nmm={n: (None if not np.isfinite(lim[k]) else round(lim[k] / fac, 1)) for k, n in enumerate(names)},
                   chain=names, held_at_0=names[0], step_deg=step, hand_tip_mm=np.round(p_hand0, 2).tolist(), hand_parts=hand_names, stage_link=spec["stage_link"],
                   origin_mm=np.round(origin, 2).tolist(),
                   stall_capped_wrenches=capped, stall_capped_by_joint={k: v for k, v in by_joint.items() if v},
                   force_N_quantiles=_q(Fm), moment_Nmm_quantiles=_q(Mm))
    return W, labels, summary


def _q(x):
    return {k: round(float(np.quantile(x, v)), 1) for k, v in (("min", 0), ("p50", .5), ("p90", .9), ("max", 1))}
