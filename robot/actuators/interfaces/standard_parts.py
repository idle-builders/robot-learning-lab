"""Standard-part interface definitions: where every purchased part can be screwed to (owner 2026-11-18, Idle T-ze9wtj).

Each motor is defined from its own official STEP (owner): every tapped hole that opens on the part's outside is read
from the B-rep (position of its opening, outward axis, tap diameter, modelled depth), labelled by member (case / horn /
idler) and face, and given its thread and the maximum tap depth from the maker's drawing. Case-assembly screws (holes the
servo's own screws occupy) are listed separately and are not mount holes. Generic hardware (screw, washer, nut, sleeve
families) is defined from ISO / maker tables below.

  .venv/bin/python -B robot/actuators/interfaces/standard_parts.py      # writes standard-parts.json next to this file

Frames: each motor's definition is in its official STEP's own frame (mm); the mount check registers every instance in
the body to that frame. Sources (URL, sha256, retrieval date) are in sources.json; the files live in
artifacts/robot/actuators/interfaces/sources/ (external inputs, kept in the private store).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "artifacts/robot/actuators/interfaces/sources"
OUT = HERE / "standard-parts.json"

# tap drill (modelled hole) diameter -> thread
TAP = [(1.25, 1.45, "M1.7"), (1.55, 1.75, "M2"), (2.0, 2.15, "M2.5"), (2.4, 2.6, "M3")]

# Generic hardware. Clearance and head sizes: ISO 273 (fine..medium), ISO 7380 / ISO 7045 / ISO 4762 heads; PA = B-PA
# thread-forming screws for plastics (major dia 2.0 for PA2), pilot hole per maker guidance ~0.8 d.
SCREW_FAMILIES = {
    "M1.7": dict(major_dia_mm=1.7, tap_dia_mm=1.35, clearance_dia_mm=[1.85, 2.0], head_dia_mm=[3.0, 3.4]),
    "M2": dict(major_dia_mm=2.0, tap_dia_mm=1.6, clearance_dia_mm=[2.2, 2.4], head_dia_mm=[3.5, 3.8]),
    "PA2": dict(major_dia_mm=2.0, tap_dia_mm=1.6, clearance_dia_mm=[2.2, 2.4], head_dia_mm=[3.5, 3.8],
                note="thread-forming screw for plastics; terminal is a 1.6-1.7 mm pilot hole in plastic, no nut"),
    "M2.5": dict(major_dia_mm=2.5, tap_dia_mm=2.05, clearance_dia_mm=[2.7, 2.9], head_dia_mm=[4.5, 5.0]),
    "M3": dict(major_dia_mm=3.0, tap_dia_mm=2.5, clearance_dia_mm=[3.2, 3.4], head_dia_mm=[5.5, 5.7]),
}
# reference thread engagement (mm) by the material of the terminal, as a multiple of the major diameter
ENGAGEMENT_REF = {"aluminium": 1.5, "steel": 1.0, "plastic": 2.0, "brass": 1.5}
SLEEVE = dict(material="304 stainless tube, pressed in (owner M-pxbsxq)", rule="every hole where a printed part is "
              "clamped under a screw head or nut carries one; holes a screw threads into carry none (owner 2026-11-18)",
              bore_for={"M2": [2.2, 2.4], "PA2": [2.2, 2.4], "M2.5": [2.7, 2.9], "M3": [3.2, 3.4]},
              head_bearing="the head (or washer) should bear on the sleeve's steel end: report the bearing ring width "
                           "min(head r, sleeve outer r) - sleeve inner r (mm)")

# Motors. members: solid indices of the official STEP; case_screws: hole openings the servo's own screws occupy (not
# mount holes), as (x, y, z) within 0.3 mm; drawing: thread / max tap depth per member+face from the maker's drawing.
MOTORS = {
    "ST3025": dict(
        maker="Waveshare (Feetech ST-3025-C002)", tiers=["fast leg: hip pitch, knee, ankle pitch"],
        step="artifacts/cove50-engineering-competition/elbow-studio/direct-servo/r1/source/ST3025-MODEL/ST3025-MODEL.stp",
        case_exterior=[0, 4, 5], members={"horn": [2, 7], "idler": [8, 9, 15]}, case_material="aluminium",
        horn_material="aluminium",
        frame="output axis along +z through (x 10, y 0); horn on +z",
        output_axis=((10.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        case_screws=[(s * 17.65, t * 7.65, z) for s in (-1, 1) for t in (-1, 1) for z in (-21.45, -19.45)],
        drawing=dict(source="Waveshare ST3025 product drawing (wiki); tap depths from the official STEP",
                     max_tap_depth_mm={}),
    ),
    "ST3215": dict(
        maker="Waveshare ST3215 12 V (= Feetech ST-3215 12 V, C047 case)", tiers=["light: hip yaw, ankle roll"],
        step="artifacts/robot/actuators/interfaces/sources/st3215-3d/ST3215.step",
        case_exterior=[0, 1, 2], members={"horn": [6], "idler": [7], "shaft": [5]}, case_material="plastic", horn_material="aluminium",
        frame="output axis along +y through (x -25.5, z 0); horn on +y",
        output_axis=((-25.5, 0.0, 0.0), (0.0, 1.0, 0.0)),
        case_screws=[(-32.6, 4.9, s * 9.35) for s in (-1, 1)] + [(7.25, y, s * 10.0) for s in (-1, 1) for y in (4.9, -15.1)],
        drawing=dict(source="Waveshare ST3215-2D.zip ST3215.pdf (SCS215, 2022-06-08): 4 M2 per face, horn 4-M3 on "
                            "dia 14, output 10.11 from the edge; no tap depth given", max_tap_depth_mm={}),
    ),
    "STS3250": dict(
        maker="Feetech STS3250 (ST-3250-C001)", tiers=["strong: hip roll, shoulder pitch/roll/yaw, elbow"],
        step="artifacts/robot/actuators/interfaces/sources/feetech_sts3250.step",
        case_exterior=[0, 1, 2], members={"horn": [11], "idler": [12], "shaft": [9, 10]}, case_material="aluminium", horn_material="aluminium",
        frame="output axis along +y through (x -25.5, z 0); horn on +y (same case frame as ST3215)",
        output_axis=((-25.5, 0.0, 0.0), (0.0, 1.0, 0.0)),
        case_screws=[(-32.6, y, s * 9.35) for s in (-1, 1) for y in (3.4, 4.9)] + [(7.25, 4.9, s * 10.0) for s in (-1, 1)],
        drawing=dict(source="Feetech STS3250 product specification A/0 2024-01-16 p6-7: 4-M2.0 per face, horn 4-M3 "
                            "on dia 14 (4.5 thick), idler 4-M3 (3.1), rocker screw M3x6, case aluminium. STEP is a "
                            "community model (step.parts, earthtojake); Feetech publishes none", max_tap_depth_mm={}),
    ),
    "XM335": dict(
        maker="ROBOTIS DYNAMIXEL XM335-T323-T", tiers=["neck yaw, neck pitch"],
        step="artifacts/robot/actuators/interfaces/sources/XM335.stp",
        case_exterior=[0, 1, 2], members={"horn": [5, 6], "idler": [3, 4]}, case_material="plastic", horn_material="aluminium",
        frame="output axis along +z through (x 0, y 0); horn on +z",
        output_axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        case_screws=[],
        drawing=dict(source="ROBOTIS XM335.pdf (2026-01-12): horn 4-M2 TAP DP4.5 PCD 12; idler 4-M2 TAP DP2 PCD 12; "
                            "side/back/bottom M2 TAP DP2; 2-M1.7 TAP DP3 front and rear",
                     max_tap_depth_mm={"horn": 4.5, "idler": 2.0, "case:M2": 2.0, "case:M1.7": 3.0}),
    ),
}


# Official sources (owner-approved downloads 2026-11-18); fetched again by URL and sha256 when missing.
SOURCES = {
    "st3215-3d/ST3215.step": dict(url="https://files.waveshare.com/upload/5/59/ST3215-3D.zip", archive="ST3215-3D.zip",
                                 archive_sha256="2735561e0c1dc899b35d4740273e5340f15967028d2dd4014feed03d8175562c",
                                 page="https://www.waveshare.com/wiki/ST3215_Servo"),
    "st3215-2d/ST3215.pdf": dict(url="https://files.waveshare.com/upload/0/08/ST3215-2D.zip", archive="ST3215-2D.zip",
                                archive_sha256="bd034f5301ec271a77a87e8bf5f6bb9995bb3348ba728b97ae3c58dffa79dc1b",
                                page="https://www.waveshare.com/wiki/ST3215_Servo"),
    "feetech-sts3250-spec.pdf": dict(url="https://www.feetechrc.com/Data/feetechrc/upload/file/20260622/6391772424007467984928587.pdf",
                                     sha256="92926b0571809316d17f0d9cb3d8da8924a7f234f950c7eabc7330be0665c2d7",
                                     page="https://www.feetechrc.com/en/562636.html"),
    "feetech_sts3250.step": dict(url="https://media.githubusercontent.com/media/earthtojake/step.parts/c6113328a5695b976a010a203a90fe86191769bf/catalog/step/feetech_sts3250.step",
                                 sha256="cf46f17da455e1f158114791bb31404c24d925e8a758bbd6189f8ee815a571bf",
                                 page="https://www.step.parts/parts/feetech_sts3250", note="community model; Feetech publishes no STEP"),
    "XM335.stp": dict(url="https://www.dropbox.com/scl/fi/nwt3zokelb7jezcbcdx7y/XM335.stp?rlkey=287d8xerqsq98xqw5fjy873gx&dl=1",
                      sha256="df2e22d7110e8de1db42a71e409d6cc3984195fcdbb0144b79e0415441b407e7",
                      page="https://emanual.robotis.com/docs/en/dxl/x/xm335-t323/#drawings (robotis.com download.php?no=2236)"),
    "XM335.pdf": dict(url="https://www.dropbox.com/scl/fi/t7w6essqmcdgsg0x55jp0/XM335.pdf?rlkey=5da1pnrbs90754nd3vqlsxlpt&dl=1",
                      sha256="4beaa90293f6ca656951e91b336faa422ef271a72259562f7508836a3762e9aa",
                      page="https://emanual.robotis.com/docs/en/dxl/x/xm335-t323/#drawings (robotis.com download.php?no=2235)"),
}


def fetch_sources():
    """Download any missing official source into artifacts/robot/actuators/interfaces/sources/ and check its sha256."""
    import subprocess
    import zipfile
    for rel, s in SOURCES.items():
        dst = SRC / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = SRC / (s.get("archive") or rel)
        subprocess.run(["curl", "-sL", "--max-time", "300", "-A", "Mozilla/5.0", "-o", str(tmp), s["url"]], check=True)
        want = s.get("archive_sha256") or s.get("sha256")
        if want and sha(tmp) != want:
            raise RuntimeError(f"{tmp.name}: sha256 {sha(tmp)} != recorded {want}")
        if s.get("archive"):
            zipfile.ZipFile(tmp).extractall(dst.parent)


def thread_of(d):
    for lo, hi, t in TAP:
        if lo <= d <= hi:
            return t
    return None


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def classify_inside(solids):
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    def inside(p):
        for s in solids:
            c = BRepClass3d_SolidClassifier(s.wrapped, gp_Pnt(*map(float, p)), 1e-6)
            if c.State() in (TopAbs_IN, TopAbs_ON):
                return True
        return False
    return inside


def face_label(axis_out, frame_axes=("x", "y", "z")):
    k = int(np.argmax(np.abs(axis_out)))
    return ("+" if axis_out[k] > 0 else "-") + frame_axes[k]


def build_motor(key, spec):
    import sys
    sys.path.insert(0, str(HERE))
    from build123d import import_step
    from holes import holes_of_shape
    path = ROOT / spec["step"]
    shape = import_step(str(path))
    solids = shape.solids()
    member_of = {i: m for m, idx in spec["members"].items() for i in idx}
    inside = classify_inside(solids)
    centre = np.mean([[s.center().X, s.center().Y, s.center().Z] for s in solids], axis=0)
    mount, case_screw, other = [], [], []
    seen = []
    for i, so in enumerate(solids):
        member = member_of.get(i, "case")
        if member == "shaft":
            continue
        for h in holes_of_shape(so):
            if h["span_deg"] < 300:
                continue
            thr = thread_of(h["dia_mm"])
            for end, far in ((h["end_a"], h["end_b"]), (h["end_b"], h["end_a"])):
                ax = (end - far) / max(np.linalg.norm(end - far), 1e-9)       # outward along the hole
                if inside(end + 0.5 * ax):
                    continue                                                  # this end is closed / inside the servo
                if float(ax @ (end - centre)) <= 0:
                    continue                                                  # opens toward the servo, not out of it
                key_pt = tuple(np.round(end, 2))
                if any(np.linalg.norm(np.array(key_pt) - np.array(s)) < 0.05 for s in seen):
                    continue
                seen.append(key_pt)
                # tapped through the member (the far end opens too, e.g. a horn flange): the thread there is only the
                # member's thickness, so no screw can engage more than modelled_depth_mm
                through = not inside(far - 0.5 * ax)
                row = dict(member=member, solid=i, face=face_label(ax), opening_mm=np.round(end, 3).tolist(),
                           axis_out=np.round(ax, 4).tolist(), tap_dia_mm=h["dia_mm"], modelled_depth_mm=h["depth_mm"],
                           thread=thr, through=bool(through))
                p0, d0 = (np.array(v, float) for v in spec["output_axis"])
                on_axis = np.linalg.norm((end - p0) - ((end - p0) @ d0) * d0) < 0.2 and abs(float(ax @ d0)) > 0.99
                if on_axis:
                    row["member"] = "idler" if float(ax @ d0) < 0 else "horn"
                    row["centre"] = True
                if any(np.linalg.norm(end - np.array(c)) < 0.3 for c in spec["case_screws"]):
                    case_screw.append(row)
                elif thr is None:
                    other.append(row)
                else:
                    mount.append(row)
    # labels and drawing data
    mount.sort(key=lambda r: (r["member"], r["face"], r["thread"], tuple(r["opening_mm"])))
    counts = {}
    md = spec["drawing"]["max_tap_depth_mm"]
    mat = {"case": spec["case_material"], "horn": spec["horn_material"], "idler": spec["horn_material"]}
    for r in mount:
        base = f"{r['member']}{r['face']}" + (".centre" if r.get("centre") else "")
        counts[base] = counts.get(base, 0) + 1
        r["hole_id"] = f"{key}.{base}.{counts[base]}"
        r["drawing_max_tap_depth_mm"] = md.get(r["member"], md.get(f"{r['member']}:{r['thread']}"))
        major = SCREW_FAMILIES[r["thread"]]["major_dia_mm"]
        r["terminal_material"] = mat[r["member"]]
        ref = ENGAGEMENT_REF[mat[r["member"]]] * major
        r["material_reference_engagement_mm"] = round(ref, 2)
        # the reference is what the part can give: the material rule (x major dia), capped by the drawing's tap depth
        # and, for a hole tapped through its member, by the thread that exists (e.g. STS3250 horn: 4-M3 through a
        # 2.5 mm flange, Feetech spec p7; 1.5 x 3 = 4.5 mm is not reachable there)
        caps = [x for x in (r["drawing_max_tap_depth_mm"], r["modelled_depth_mm"] if r["through"] else None) if x]
        r["reference_engagement_mm"] = round(min([ref] + caps), 2)
        r["reference_basis"] = ("tapped through the member: its thickness" if r["through"] and r["modelled_depth_mm"] < ref
                                else "drawing tap depth" if r["drawing_max_tap_depth_mm"] and r["drawing_max_tap_depth_mm"] < ref
                                else f"{ENGAGEMENT_REF[mat[r['member']]]} x major dia ({mat[r['member']]})")
    return dict(maker=spec["maker"], tiers=spec["tiers"], frame=spec["frame"],
                step=spec["step"], step_sha256=sha(path), members={k: v for k, v in spec["members"].items()},
                case_exterior=spec["case_exterior"],
                case_material=spec["case_material"], drawing=spec["drawing"]["source"],
                mount_holes=mount, case_screw_holes=case_screw, other_holes=other,
                summary=counts, output_axis=spec["output_axis"])


def main():
    fetch_sources()
    out = dict(schema="robot-learning-lab.standard-parts.v1",
               note="Built by standard_parts.py from each maker's official STEP and drawing; edit the script, not this "
                    "file. Frames are the STEP files' own; the mount check registers each instance to them.",
               screw_families=SCREW_FAMILIES, engagement_reference_x_major_dia=ENGAGEMENT_REF, sleeve=SLEEVE,
               motors={k: build_motor(k, v) for k, v in MOTORS.items()})
    OUT.write_text(json.dumps(out, indent=1))
    for k, m in out["motors"].items():
        print(k, m["summary"], "case screws", len(m["case_screw_holes"]), "other", len(m["other_holes"]))


if __name__ == "__main__":
    main()
