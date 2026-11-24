"""Joint ranges come from the body definitions only (owner 2026-11-23: "any configured value should come from the bom.yaml").
steer-joint-range   one joint's declared range changed only in a copy of the definitions (LAB_DEFINITIONS_ROOT): range_sweep's swept
                    and declared range of that joint, its range source and the region poses that set it move with it; the other
                    joints' sweeps, the poses that do not set it and the checks that read no range (mass, gaps) stay as they are.
The joint, the part and the new range are the harness's choice; the checks read the range from whatever definitions they are given."""
from __future__ import annotations

import json
import os
import shutil
import sys

from evaluate import EVAL, ROOT, results_of, run_tool, scenario, selected_manifest

JOINT = "neck_pitch"                        # a region (neck) joint: its range sets the region poses as well as the sweep
PART = "combined__neck_pitch_to_head_cradle"  # a printed connector on the head link: neck_yaw and neck_pitch both move it
NEW_HI = 60.0                               # beyond the joint's stress extent (joint_sweep_wide_deg 50), so the sweep widens too


def _poses(env_root):
    """Pose names and angles of the key / library sets for PART under the given definitions (None = the real ones)."""
    sys.path.insert(0, str(ROOT / "robot/body"))
    old = os.environ.get("LAB_DEFINITIONS_ROOT")
    try:
        if env_root:
            os.environ["LAB_DEFINITIONS_ROOT"] = str(env_root)
        else:
            os.environ.pop("LAB_DEFINITIONS_ROOT", None)
        from checks import core
        body = core.load_body(selected_manifest())
        lk = (body.parts[PART]["owning_link"],)
        return {w: {n: {j: a for j, a in q.items() if j == JOINT} for n, _, q in core.pose_set(body, w, lk)} for w in ("key", "library")}, \
            body.range_source.get(JOINT)
    finally:
        if old is None:
            os.environ.pop("LAB_DEFINITIONS_ROOT", None)
        else:
            os.environ["LAB_DEFINITIONS_ROOT"] = old


@scenario("steer-joint-range", "truth", "a joint range changed only in a copy of the definitions moves range_sweep's swept range and the "
          "region poses of that joint, and nothing unrelated")
def s_steer_range(sid):
    import yaml
    from scenarios.agnostic import _diff
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import bom_config
    import definitions as DEF
    d = EVAL / sid / "definitions"
    shutil.rmtree(EVAL / sid, ignore_errors=True); d.mkdir(parents=True)
    decl = bom_config.joints()[JOINT]
    before = list(decl["range_deg"])
    after = [before[0], NEW_HI]
    for i, f in enumerate(DEF.discover(root=ROOT)):
        doc = yaml.safe_load(f.read_text())
        if str(f.relative_to(ROOT)) == decl["file"]:
            ent = doc["joints"][int(decl["entry"].split("[")[1].rstrip("]"))]
            ent["range_deg"] = after
        (d / f"{i:03d}-{f.name}").write_text(yaml.safe_dump(doc, sort_keys=False, width=200))
    args = ["--parts", PART, "--checks", "range_sweep,clearance,mass,gaps"]
    a = run_tool(sid + "-as-is", args, limit_gb=6, timeout_s=2400)
    b = run_tool(sid + "-steered", args, limit_gb=6, timeout_s=2400, env=dict(os.environ, LAB_DEFINITIONS_ROOT=str(d)))
    ra, rb = results_of(a["report"]), results_of(b["report"])
    pa, sa = _poses(None)
    pb, sb = _poses(d)
    rs = lambda r: (r.get("range_sweep") or {}).get("per_joint") or {}  # noqa: E731
    sweeps = {jn: dict(as_is=dict(swept_deg=x.get("swept_deg"), declared_range_deg=x.get("declared_range_deg"),
                                  first_contact_deg=x.get("first_contact_deg")),
                       steered=dict(swept_deg=(rs(rb).get(jn) or {}).get("swept_deg"), declared_range_deg=(rs(rb).get(jn) or {}).get("declared_range_deg"),
                                    first_contact_deg=(rs(rb).get(jn) or {}).get("first_contact_deg")),
                       differs=bool(_diff(x, rs(rb).get(jn))))
              for jn, x in rs(ra).items()}
    moved = {w: sorted(n for n in set(pa[w]) | set(pb[w]) if pa[w].get(n) != pb[w].get(n)) for w in pa}
    return dict(rc=[a["rc"], b["rc"]], joint=JOINT, declared=[before, after], range_source=[sa, sb], range_sweep_per_joint=sweeps,
                poses_setting_joint_changed=moved,
                poses_unchanged={w: sum(1 for n in pa[w] if pa[w].get(n) == pb[w].get(n)) for w in pa},
                changed_checks=sorted(c for c in ra if _diff(ra.get(c), rb.get(c))),
                unchanged_checks=sorted(c for c in ra if not _diff(ra.get(c), rb.get(c))),
                clearance_diff=_diff(ra.get("clearance"), rb.get("clearance"))[:8])
