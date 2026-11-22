"""Runner scenarios (T-aprdsc, T-fdczae): manifest freshness, manifests outside the repository, rejected STEPs that are only
moved, reference volumes of multi-solid parts, approximate solids named on the summary line, the checks/data ignore trap."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from evaluate import REPORTS, ROOT, THIGH, basics, run_tool, scenario, selected_manifest


def _freshness(rep):
    return (rep or {}).get("manifest_freshness") if isinstance(rep, dict) else None


@scenario("manifest-freshness", "truth", "a run says when its manifest is older than what it is built from: the selected body "
          "(downstream of a roll-flip manifest written later) against the roll-flip manifest itself (nothing newer upstream)")
def s_fresh(sid):
    a = run_tool(sid + "-selected", ["--parts", THIGH, "--checks", "mass"])
    rf = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
    b = run_tool(sid + "-rollflip", ["--parts", THIGH, "--checks", "mass"], manifest=rf)
    fa, fb = _freshness(a["report"]), _freshness(b["report"])
    return dict(selected=dict(rc=a["rc"], findings=len(fa or []), kinds=sorted({x["kind"] for x in fa or []}),
                              printed=[ln for ln in a["log"].splitlines() if ln.startswith("manifest freshness")][:4]),
                roll_flip=dict(rc=b["rc"], findings=len(fb or []), detail=fb))


@scenario("manifest-outside-repo", "robust", "--manifest names a copy of the selected body outside the repository (a session "
          "scratchpad): the run measures it; the result cache stays off (its reads are not recorded)")
def s_outside(sid):
    d = Path(tempfile.mkdtemp(prefix="checks-outside-"))
    try:
        f = d / "baseline-copy.json"
        shutil.copy(selected_manifest(), f)
        cmd_args = ["--parts", THIGH, "--checks", "mass,gaps", "--manifest", str(f)]
        r = run_tool(sid, cmd_args, cache=True)
        rep = r["report"] if isinstance(r["report"], dict) else {}
        return dict(basics(r), manifest_in_report=rep.get("manifest"), cached=[n for n, c in rep.get("checks", {}).items() if c.get("cached")],
                    tail=[ln for ln in r["log"].splitlines() if ln.strip()][-3:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


@scenario("rejected-step-offset", "truth", "a STEP that is the part moved (pitch C047 cases, 12 mm in x): the screening solid says "
          "by how much instead of a bare 'frame or manifold mismatch'")
def s_offset(sid):
    r = run_tool(sid, ["--parts", "upper__R_pitch_C047_case,upper__L_pitch_C047_case", "--checks", "mass"])
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return dict(basics(r), how={n: v.get("how") for n, v in rep.get("solids", {}).items()},
                volume_off=rep.get("solid_volume_off_over_5pct"))


@scenario("reference-volume", "truth", "screening solids against their reference volume around the torso chassis and the ankle "
          "couplers: multi-solid sleeve STEPs compared with their fused STEP volume, not a stale recorded volume")
def s_refvol(sid):
    r = run_tool(sid, ["--parts", "combined__stage1_center_torso_spine_chassis,R_ankle_coupler_combined,L_ankle_coupler_combined",
                       "--checks", "gaps"], timeout_s=1200)
    rep = r["report"] if isinstance(r["report"], dict) else {}
    sleeves = {n: v for n, v in (rep.get("solid_volume_off_over_5pct") or {}).items()}
    return dict(basics(r), over_5pct=sleeves)


@scenario("approximate-solid-marked", "robust", "an overlap measured against a voxel-filled open supplier mesh (fan bridge plate x "
          "SameSky fan) says so on its own summary line")
def s_approx(sid):
    r = run_tool(sid, ["--parts", "fan_bridge_plate", "--checks", "gaps,static_audit"])
    lines = [ln for ln in r["log"].splitlines() if "SameSky_4010_fan" in ln and ("mm3" in ln or "mm distance" in ln)]
    return dict(basics(r), lines=lines[:4], marked=sum("[approximate:" in ln for ln in lines), of=len(lines))


@scenario("checks-data-addable", "robust", "tracked files under robot/body/checks/data are not ignored (git add of one used to "
          "refuse the whole command)")
def s_ignore(sid):
    p = subprocess.run(["git", "check-ignore", "-v", "robot/body/checks/data/part_settings.json", "robot/body/checks/data/x.json"],
                       cwd=ROOT, capture_output=True, text=True)
    return dict(ignored=p.stdout.strip().splitlines(), rc=p.returncode)
