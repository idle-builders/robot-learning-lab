"""Motion-check scenarios for the pose sets that name one leg's joints (pose_sets.json leg_motion, fit_poses, range_ends).

mirror-leg-twin   clearance (--poses key) on two right-leg parts and, in a second run, on their mirror twins: the left-leg
                  parts must be moved as their right twins are (the poses mirrored onto the left leg, core.pose_set), so
                  each left pair, mapped through the mirror relation (checks.relations: x-mirrored boxes), measures what
                  its right twin measures. Reports per run the poses that move the parts' link and the pairs, and per
                  matched pair the difference of new overlap (mm3) and minimum distance (mm).

The part names here are the harness's choice; the twins are found by the relations, not by the names."""
from __future__ import annotations

import sys

from evaluate import ROOT, basics, run_tool, scenario, selected_manifest

RIGHT = ["R_thigh_roll_flip", "R_hip_pitch_to_roll_roll_flip"]


def _relations():
    sys.path.insert(0, str(ROOT / "robot/body"))
    from checks import core
    from checks.relations import Relations
    man = selected_manifest()
    body = core.load_body(man)
    return body, Relations(body, man)


def _moving_poses(body, scope, poses):
    """Poses at which a scoped part's link stands elsewhere than at the first pose (standing)."""
    import numpy as np
    links = {body.parts[n]["owning_link"] for n in scope}
    T0 = {lk: body.link_T(lk, poses[0][2]) for lk in links}
    return sum(1 for _, _, q in poses if any(not np.allclose(body.link_T(lk, q), T0[lk], atol=1e-9) for lk in links))


@scenario("mirror-leg-twin", "truth", "clearance on two right-leg parts and on their left mirror twins: the left leg moves as the "
          "right does, each mirrored pair measures what its right twin measures")
def s_twin(sid):
    body, rel = _relations()
    left = [rel.mirror_of(n) for n in RIGHT]
    from checks import core
    out, by = {}, {}
    for side, scope in (("right", RIGHT), ("left", left)):
        r = run_tool(f"{sid}-{side}", ["--parts", ",".join(scope), "--checks", "clearance", "--poses", "key"], limit_gb=6, timeout_s=1200)
        c = (r["report"] if isinstance(r["report"], dict) else {}).get("checks", {}).get("clearance", {})
        pairs = {(a, m): w for a, ms in (c.get("per_part") or {}).items() for m, w in ms.items()}
        poses = core.pose_set(body, "key", tuple(sorted({body.parts[n]["owning_link"] for n in scope})))
        by[side] = pairs
        out[side] = dict(basics(r), parts=scope, poses=len(poses), poses_moving_the_parts=_moving_poses(body, scope, poses),
                         pairs=len(pairs), pairs_with_new_overlap=sum(1 for w in pairs.values() if w["max_new_overlap_mm3"] > 0.1),
                         min_distance_mm=min((w["min_distance_mm"] for w in pairs.values()), default=None))
    matched, unmatched = [], []
    for (a, m), w in by["left"].items():
        ra, rm = rel.mirror_of(a), rel.mirror_of(m)
        t = by["right"].get((ra, rm))
        if t is None:
            unmatched.append(f"{a} x {m}")
            continue
        matched.append(dict(pair=f"{a} x {m}", twin=f"{ra} x {rm}",
                            d_new_overlap_mm3=round(w["max_new_overlap_mm3"] - t["max_new_overlap_mm3"], 2),
                            d_min_distance_mm=round(w["min_distance_mm"] - t["min_distance_mm"], 3)))
    worst = sorted(matched, key=lambda x: -max(abs(x["d_new_overlap_mm3"]), abs(x["d_min_distance_mm"])))
    return dict(out, matched_pairs=len(matched), right_pairs_without_left_twin=len(set(by["right"]) - {
                    (rel.mirror_of(a), rel.mirror_of(m)) for a, m in by["left"]}),
                left_pairs_without_right_twin=unmatched[:10],
                max_abs_d_new_overlap_mm3=max((abs(x["d_new_overlap_mm3"]) for x in matched), default=None),
                max_abs_d_min_distance_mm=max((abs(x["d_min_distance_mm"]) for x in matched), default=None),
                largest_differences=worst[:8])
