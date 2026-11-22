"""Scenarios for the screw checks (screws / mount_check, driver_access, fastener_motion, connectivity, interfaces.py):
real run.py runs on the selected body, each on the parts a body session reported a tool artefact on (T-vza9m4, T-fdvj93,
T-fbpzck, T-5ptw44, 2026-11-22). Each returns the measurements the report was about, so a run before and after a change
to the tool shows the artefact and its fix side by side. The reported parts are evidence, not knowledge of the tool."""
from __future__ import annotations

import json
import sys

from evaluate import ROOT, basics, run_tool, scenario

HEAD = "combined__head_sensor_compute_carrier"     # the ROCK4D compute board's four screws pass it
FAN = "fan_bridge_plate"                            # fan plate screws: sleeve over the fan's counterbore, plate on the fan face
STAGE3 = "combined__stage3_front_power_electronics_carrier"
STAGE4 = "combined__stage4_rear_imu_adc_carrier"
THIGH = "R_thigh_roll_flip"                         # the H-cage: shell-post screws, thigh shells over the case screws
HIP_PITCH = "R_hip_pitch_to_roll_roll_flip"          # its A9 hip screws are named M3x4 and modelled 8 mm long
SHOULDER = "R_shoulder_pitch_to_roll_combined,R_shoulder_roll_to_yaw_combined"
ANKLE = "R_ankle_pitch_ST3025_output,L_ankle_pitch_ST3025_output"
INDICATOR, FRONT_SHELL = "ape_body_shell__chest_indicator_relief4_r33", "center_front_shell"


def screws_of(rep, check="screws"):
    """{screw: row} of a check's per_part (each screw once)."""
    out = {}
    for d in ((rep or {}).get("checks", {}).get(check, {}).get("per_part") or {}).values():
        out.update(d.get("screws") or {})
    return out


def holes_of(rows, part):
    """{screw: (axis offset mm, tilt deg, hole dia mm)} of the clamped hole of `part` in each screw row."""
    return {k: next(((h["axis_offset_mm"], h["axis_tilt_deg"], h["hole_dia_mm"]) for h in r.get("clamped_holes", []) if h["part"] == part), None)
            for k, r in sorted(rows.items())}


def layer_gaps(rows, pred=lambda c: True):
    """{screw: [(upper, lower, min, max, level gap, bears)]} of the interface layers."""
    return {k: [(c["upper"], c["lower"], c["min_gap_mm"], c.get("max_gap_mm"), c.get("level_gap_mm"), c.get("bears"))
                for c in r.get("layers", []) if c["interface"] and pred(c)] for k, r in sorted(rows.items())}


def gap_findings(rows):
    return {k: [(f["kind"], f["value"], f.get("note")) for f in r.get("findings", []) if f["kind"].startswith(("layers_gap", "head_seat"))]
            for k, r in sorted(rows.items())}


@scenario("fasteners-board-holes", "truth", "ROCK4D mount holes (stray wedges, open 244k-face mesh): fitted hole offset vs the "
          "concentric spacers / washers, two runs")
def s_board_holes(sid):
    a = run_tool(sid + "-a", ["--parts", HEAD, "--checks", "screws"])
    b = run_tool(sid + "-b", ["--parts", HEAD, "--checks", "screws"])
    ra, rb = screws_of(a["report"]), screws_of(b["report"])
    board = "ROCK4D_official"
    on_board = {k: r for k, r in ra.items() if any(h["part"] == board for h in r.get("clamped_holes", []))}
    others = {k: [(h["part"], h["axis_offset_mm"]) for h in r["clamped_holes"] if h["part"] != board] for k, r in on_board.items()}
    return dict(basics(a), board_hole=holes_of(on_board, board), other_holes_on_the_same_screws=others,
                board_hole_second_run=holes_of({k: rb[k] for k in on_board if k in rb}, board),
                identical_runs=holes_of(on_board, board) == holes_of({k: rb[k] for k in on_board if k in rb}, board))


@scenario("fasteners-clamp-level", "truth", "clamp stack where a sleeve ends over a counterbore while its part bears (fan plate, "
          "H-cage shell posts): layer gaps per pair and per level, gap findings")
def s_clamp_level(sid):
    out = {}
    for tag, prt in (("fan", FAN), ("thigh", THIGH)):
        r = run_tool(f"{sid}-{tag}", ["--parts", prt, "--checks", "screws"])
        rows = screws_of(r["report"])
        sel = {k: v for k, v in rows.items() if any(c["upper"].endswith("_steel_sleeves") for c in v.get("layers", []))}
        out[tag] = dict(rc=r["rc"], exec_s=r["exec_s"], layers=layer_gaps(sel, lambda c: not c["upper"] == c.get("lower")),
                        gap_findings=gap_findings(sel),
                        sleeve_findings={k: [(f["kind"], f["value"], f.get("note")) for f in v["findings"] if "sleeve" in f["kind"]] for k, v in sorted(sel.items())},
                        screws_with_gap_over_0p05mm=((r["report"] or {}).get("checks", {}).get("screws", {}).get("bad_events") or {}).get("screws_with_gap_over_0p05mm"))
    return out


@scenario("fasteners-layer-pairing", "truth", "board stacks (power supervisor, main precharge, IMU, ADC): board-to-carrier gap "
          "max over the rays (an earlier segment of the board paired with the carrier)")
def s_layer_pairing(sid):
    out = {}
    for tag, prt, gb in (("stage3", STAGE3, 4), ("stage4", STAGE4, 6)):
        r = run_tool(f"{sid}-{tag}", ["--parts", prt, "--checks", "screws"], limit_gb=gb)
        rows = screws_of(r["report"])
        board = {k: [(c["upper"], c["lower"], c["min_gap_mm"], c.get("max_gap_mm")) for c in v.get("layers", [])
                     if c["interface"] and c["lower"] in (prt, prt + "_steel_sleeves")] for k, v in sorted(rows.items())}
        out[tag] = dict(rc=r["rc"], exec_s=r["exec_s"], into_carrier=board,
                        max_gap_into_carrier_mm=max((x[3] for v in board.values() for x in v if x[3] is not None), default=None),
                        small_gap_layers=[(k, c["upper"], c["lower"], c["min_gap_mm"]) for k, v in sorted(rows.items()) for c in v.get("layers", [])
                                          if c["interface"] and c.get("bears", True) and 0.0 < c["min_gap_mm"] <= 0.5])
    return out


@scenario("fasteners-screw-identity", "truth", "screws named without a thread or with a stale length (shoulder U roll proxies, "
          "A9 hip, PA-named M2 screws): seen by screws / driver_access / fastener_motion, spec source, tool")
def s_identity(sid):
    r = run_tool(sid + "-shoulder", ["--parts", SHOULDER, "--checks", "screws,driver_access,fastener_motion"], limit_gb=6, timeout_s=1200)
    t = run_tool(sid + "-torso-hip", ["--parts", f"{STAGE4},{HIP_PITCH}", "--checks", "screws,driver_access"], limit_gb=6, timeout_s=1200)
    rep = r["report"] if isinstance(r["report"], dict) else {}
    fm = {k for d in (rep.get("checks", {}).get("fastener_motion", {}).get("per_part") or {}).values() for k in d}
    sc, da = screws_of(rep), screws_of(rep, "driver_access")
    proxies = lambda names: sorted(k for k in names if "_screw_" in k)    # noqa: E731  the reported proxies: no thread token in their name
    dt, st = screws_of(t["report"], "driver_access"), screws_of(t["report"])
    sc_all = {**sc, **st}
    return dict(basics(r), torso_hip_rc=t["rc"], screws_measured=len(sc), proxies_in_screws=len(proxies(sc)), proxies_in_driver_access=len(proxies(da)),
                proxies_in_fastener_motion=len(proxies(fm)),
                spec=sorted({(v.get("thread"), v.get("nominal_length_mm"), v.get("head_standard"), v.get("spec_source"), v.get("name_length_mm")) for v in sc_all.values()}, key=str),
                modelled_vs_nominal_length_mm=sorted({(k.rsplit("_", 1)[0], v.get("length_mm"), v.get("nominal_length_mm"), v.get("name_length_mm")) for k, v in sc_all.items()
                                                      if v.get("name_length_mm") is not None}, key=str),
                driver_errors={k: v["error"] for k, v in {**da, **dt}.items() if "error" in v},
                driver_tools=sorted({(v.get("thread"), v.get("head_standard"), v.get("head_type"), v.get("drive_size")) for v in {**da, **dt}.values() if "error" not in v}, key=str),
                driver_check_s=(rep.get("checks", {}).get("driver_access") or {}).get("seconds"))


@scenario("fasteners-driver-shells", "truth", "driver paths on the H-cage at standing: obstruction by parts vs by the thigh shells "
          "that go on last")
def s_driver_shells(sid):
    r = run_tool(sid, ["--parts", THIGH, "--checks", "driver_access"])
    rows = screws_of(r["report"], "driver_access")
    return dict(basics(r), per_screw={k: dict(obstruction_mm3=v.get("obstruction_mm3"), first=(v.get("first_part"), v.get("first_distance_mm")),
                                              shell_obstruction_mm3=v.get("shell_obstruction_mm3"), covered_later_by=v.get("covered_later_by"),
                                              tool=(v.get("head_type"), v.get("drive_size")), path_mm=v.get("path_length_mm"))
                                      for k, v in sorted(rows.items())},
                bad_events=((r["report"] or {}).get("checks", {}).get("driver_access") or {}).get("bad_events"))


@scenario("fasteners-ankle-output", "truth", "ankle pitch output wheels registered to the ST3025 definition: member, fit rms, "
          "defined holes per face and the screws in them (R had been the horn alone)")
def s_ankle(sid):
    r = run_tool(sid, ["--parts", ANKLE, "--checks", "screws"])
    per = ((r["report"] or {}).get("checks", {}).get("screws", {}).get("per_part") or {})
    faces = {n: [(f["face"], f["holes"], f["used"]) for f in (d.get("reverse") or {}).get("interface_faces", [])] for n, d in per.items()}
    sys.path.insert(0, str(ROOT / "robot/body/fastening")); sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
    import fit_check as F
    import interfaces as I
    import mount_check as MC
    man = json.loads(MC.DEFAULT_MANIFEST.read_text()); body = F.Body(man)
    sub = type(body).__new__(type(body)); sub.__dict__.update(body.__dict__)
    sub.parts = {n: body.parts[n] for n in ANKLE.split(",")}
    inst = I.instances(sub, man)["instances"]          # the cached registrations run.py used
    return dict(basics(r), interface_faces=faces,
                instances={x["part"]: dict(member=x["member"], fit_rms_mm=x["fit_rms_mm"], mirrored=x["mirrored"], hole_snap=x["hole_snap"]) for x in inst},
                terminals={k: (v["terminal"] or {}).get("kind") for k, v in sorted(screws_of(r["report"]).items())})


@scenario("fasteners-coprinted", "truth", "connectivity of a part printed with its shell on the second nozzle: edge kinds, contact-only "
          "breaks; once as the definitions are, once with the joining declared on the part's definition entry (worktree only)")
def s_coprinted(sid):
    def measure(tag):
        r = run_tool(f"{sid}-{tag}", ["--parts", f"{INDICATOR},{FRONT_SHELL}", "--checks", "connectivity"])
        c = ((r["report"] or {}).get("checks", {}).get("connectivity") or {})
        pp = (c.get("per_part") or {}).get(INDICATOR) or {}
        return dict(rc=r["rc"], exec_s=r["exec_s"], indicator_edges=pp.get("edges"), indicator_chain=pp.get("chain_to_core"),
                    break_points_at_indicator=[b for b in c.get("contact_only_breaks", []) if INDICATOR in (b["from_part"], b["to_part"])],
                    edges_joined=(c.get("bad_events") or {}).get("edges_joined", 0), contact_only_break_points=(c.get("bad_events") or {}).get("contact_only_break_points"))
    out = dict(as_declared=measure("as-declared"))
    # the declaration a definition owner would add (schema property + entry), injected into this worktree's copy and restored
    if (ROOT / ".git").is_dir():
        out["declared"] = "skipped: rewrites definition files in place, run it in a worktree"
        return out
    import yaml
    schema_f = ROOT / "robot/body/bom/schema.json"
    defs_f = next(p for p in (ROOT / "robot").rglob("*.bom.yaml") if f"match: {INDICATOR}" in p.read_text())
    keep = {schema_f: schema_f.read_bytes(), defs_f: defs_f.read_bytes()}
    try:
        sch = json.loads(keep[schema_f])
        sch["$defs"]["part"]["properties"].setdefault("joined_to", {"type": "array", "items": {"type": "object"}})
        schema_f.write_text(json.dumps(sch, indent=1))
        doc = yaml.safe_load(keep[defs_f])
        e = next(e for e in doc["parts"] if e["match"] == INDICATOR)
        e["joined_to"] = [dict(part=FRONT_SHELL, how="co_printed")]
        defs_f.write_text(yaml.safe_dump(doc, sort_keys=False))
        out["declared"] = measure("declared")
    finally:
        for f, b in keep.items():
            f.write_bytes(b)
    return out


@scenario("fasteners-head-pocket", "truth", "hip yaw connector screws whose head sits in a pocket (R_A14 front PA2x4, reported 0.952 mm "
          "off against its sleeve's 0.001): clamped hole offsets and recesses")
def s_head_pocket(sid):
    r = run_tool(sid, ["--parts", "R_hip_yaw_to_pitch_combined", "--checks", "screws"])
    rows = screws_of(r["report"])
    return dict(basics(r), holes={k: [(h["part"], h["axis_offset_mm"], h["hole_dia_mm"]) for h in v.get("clamped_holes", [])] for k, v in sorted(rows.items())},
                offset_max_mm={k: v.get("hole_offset_max_mm") for k, v in sorted(rows.items())})
