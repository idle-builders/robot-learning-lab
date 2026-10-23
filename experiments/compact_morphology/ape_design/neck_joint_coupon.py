"""Head yaw/pitch packaging and shell-pitch screen for the sketch-06 ape.

The two retained MJCF head axes are coincident. These source-sized XL330 boxes
are an axis-separation hypothesis only; horn offsets and the actual neck drive
cannot be inferred from a rectangular outer envelope.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

import sketch_shell as shell
from hip_joint_coupon import add, case, cyl, overlaps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-neck-joint-coupon-20261002"
XL330_MM = (20., 34., 26.)
PITCHES = (-45, 0, 45)


def pitch(shape: bd.Shape, scale: float, angle: int) -> bd.Shape:
    pivot = (-4*scale, 0, 277*scale)
    return bd.Pos(*pivot) * bd.Rot(0, angle, 0) * bd.Pos(*(-v for v in pivot)) * shape.scale(scale)


def overlap(a: bd.Shape, b: bd.Shape) -> float:
    hit = a.intersect(b)
    return round(sum(s.volume for s in hit), 1) if hit else 0.


def screen(height: int) -> dict:
    s = height/350
    yaw = case("yaw_XL330", (-4*s, 0, 281*s), XL330_MM,
               ("torso_shell", "head_shell"), s)
    pitch_motor = case("pitch_XL330", (-4*s, 0, 311*s), XL330_MM,
                       ("head_shell",), s)
    torso = next(p.shape for p in shell.PARTS if p.name == "torso_shell").scale(s)
    head = next(p.shape for p in shell.PARTS if p.name == "head_shell")
    poses = [{"head_pitch_deg": a,
              "original_torso_head_shell_overlap_mm3": overlap(torso, pitch(head,s,a))}
             for a in PITCHES]
    return {
        "height_mm": height,
        "model_axes": "retained MJCF has coincident head yaw/pitch; proposed two-case centres differ by 30mm at 35cm",
        "source_servo": "ROBOTIS XL330-M288-T 20x34x26mm, 18g each; 0.52Nm stall only, vendor continuous estimate ~0.10Nm",
        "yaw_case": yaw, "pitch_case": pitch_motor,
        "case_envelope_clash_with_1mm_clearance": overlaps(yaw, pitch_motor),
        "poses": poses,
        "warning": "Case box corner fit and original shell BRep intersection only. Need actual shaft coordinates, independently supported yaw/pitch arrangement, head cable twist, face camera clearance and pose sweep for real head/neck mechanics.",
    }


def render(report: dict, angle: int) -> None:
    s = report["height_mm"]/350
    scene = trimesh.Scene()
    torso = next(p.shape for p in shell.PARTS if p.name == "torso_shell")
    head = next(p.shape for p in shell.PARTS if p.name == "head_shell")
    add(scene,"approved_torso_skin",torso.scale(s),"shell")
    add(scene,"approved_head_skin",pitch(head,s,angle),"shell")
    for key in ("yaw_case", "pitch_case"):
        item = report[key]
        motor = bd.Pos(*item["center_mm"]) * bd.Box(*XL330_MM)
        if key == "pitch_case":
            pivot = (-4*s, 0, 277*s)
            motor = bd.Pos(*pivot) * bd.Rot(0,angle,0) * bd.Pos(*(-v for v in pivot)) * motor
        add(scene,item["name"],motor,"motor")
    add(scene,"yaw_axis_reference",cyl(1.5,40*s,(-4*s,0,281*s)),"axis")
    add(scene,"pitch_axis_reference",cyl(1.5,45*s,(-4*s,0,311*s),"y"),"axis")
    scene.export(OUT/f"neck-{report['height_mm']}-{angle}.glb")


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    shell.build()
    reports=[screen(h) for h in (350,500,600)]
    for r in reports:
        for a in PITCHES: render(r,a)
        print(r["height_mm"],"mm: yaw/pitch outside corners",
              len(r["yaw_case"]["outside_corners"]),
              len(r["pitch_case"]["outside_corners"]),
              "neck skin overlap at +45",
              r["poses"][-1]["original_torso_head_shell_overlap_mm3"],flush=True)
    (OUT/"screen.json").write_text(json.dumps(reports,indent=2)+"\n")
    (OUT/"index.html").write_text(Path(__file__).with_name("neck_joint_viewer.html").read_text())


if __name__ == "__main__": main()
