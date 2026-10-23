"""Compare official camera STEP with drawing-based support without hiding revision drift."""

import hashlib
import argparse
import math
import shutil
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build123d import import_step, Pos, Rot, Compound, export_step, GeomType
from lab.camera_support import carrier_parts, hardware_parts, specification, interface
from lab.leg_mechanism import overlap, Part, render_core
from lab.robot import make_robot
from lab.config import load_config


def registration(raw, transform):
    substrate = max(raw.solids(), key=lambda shape: shape.volume)
    pcb = transform * substrate
    found = []
    for edge in pcb.edges():
        if edge.geom_type == GeomType.CIRCLE and abs(edge.radius - 1.1) < 1e-4:
            point = list(edge.arc_center)[1:]
            if not any(math.dist(point, p) < 1e-4 for p in found):
                found.append(point)
    expected = [[-10.5, -12.4], [10.5, -12.4], [-10.5, 0.1], [10.5, 0.1]]
    residuals = [min((math.dist(p, e) for p in found), default=1000) for e in expected]
    lens = transform * min(raw.solids(), key=lambda shape: shape.bounding_box().min.Z)
    bounds = lens.bounding_box()
    optical = [
        bounds.max.X,
        (bounds.min.Y + bounds.max.Y) / 2,
        (bounds.min.Z + bounds.max.Z) / 2,
    ]
    return dict(
        status="pass"
        if len(found) == 4
        and max(residuals) < 1e-4
        and math.dist(optical, [0, 0, 0]) < 1e-4
        else "fail",
        hole_centers_yz_mm=found,
        hole_residuals_mm=residuals,
        lens_front_center_mm=optical,
        scope="BRep hole-circle and front mechanical lens-center registration; not optical principal-plane calibration",
    )


def run(output, step_stack=False):
    output.mkdir(parents=True, exist_ok=False)
    source = Path("docs/design/references/vision/camera-module-3-wide-model-simple.stp")
    raw = import_step(source)
    transform = Pos(-8.805, -12.5, -14.4) * Rot(0, -90, 0)
    registered = registration(raw, transform)
    misplaced = registration(raw, Pos(0, 1, 0) * transform)
    if registered["status"] != "pass" or misplaced["status"] != "fail":
        raise ValueError("vendor registration or deliberate displacement check failed")
    shapes = [transform * s for s in raw.solids()]
    spec = specification()
    if step_stack:
        spec.update(
            pcb_rear_depth_mm=8.805,
            pcb_thickness_mm=0.73152,
            depth_basis="official Wide STEP outer copper planes; unmeasured clamping surface",
        )
    current = carrier_parts(spec) | hardware_parts(spec)
    rows = []
    for name, candidate in current.items():
        a = candidate.bounding_box()
        hits = []
        for i, s in enumerate(shapes):
            b = s.bounding_box()
            if any(
                list(a.max)[k] < list(b.min)[k] or list(b.max)[k] < list(a.min)[k]
                for k in range(3)
            ):
                continue
            v = overlap(candidate, s)
            if v > 0.001:
                hits.append(dict(vendor_solid_index=i, overlap_mm3=v))
        rows.append(dict(part=name, hits=hits))
    module = Compound(children=shapes)
    module.label = "official_camera_module_3_wide"
    export_step(module, output / "camera-optical-frame.step")
    parts = [
        Part("vendor_camera", "head", module, "camera", "vision", (0.3, 0.48, 0.35, 1))
    ]
    parts += [
        Part(n, "head", s, "support", "vision", (0.5, 0.6, 0.65, 0.4))
        for n, s in current.items()
    ]
    render_core(
        parts,
        {p.name: p.shape for p in parts},
        output,
        make_robot(load_config(Path("variants/pip-camera-support.yaml"))),
        azimuth=145,
        elevation=-20,
    )
    r = dict(
        status="fail" if any(row["hits"] for row in rows) else "pass",
        scope="official STEP against drawing-based support; source discrepancy audit, not physical-fit qualification",
        source=str(source),
        candidate_stack="vendor_STEP_planes"
        if step_stack
        else "drawing_interpretation",
        interface=interface(spec),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        vendor_solid_count=len(shapes),
        transform=dict(rotation="Y=-90 degrees", translation_mm=[-8.805, -12.5, -14.4]),
        drawing_vs_step=dict(
            pcb_thickness_mm=[1.12, 0.73152],
            lens_front_to_pcb_rear_mm=[8.3, 8.805],
            whole_depth_mm=[12, 11.4],
            rear_connector_width_mm=[19.61, 22.92],
        ),
        support_intersections=rows,
        registration=registered,
        displaced_registration_negative_control=misplaced,
    )
    dependencies = [
        source,
        Path("electronics/camera-support.yaml"),
        Path("lab/camera_support.py"),
        Path("lab/camera_mount.py"),
        Path("tools/audit_camera_vendor.py"),
        Path("lab/vision.py"),
        Path("electronics/vision.yaml"),
        Path("lab/leg_mechanism.py"),
        Path("lab/config.py"),
        Path("lab/robot.py"),
        Path("lab/styling.py"),
        Path("robot/base.yaml"),
        Path("variants/pip-camera-support.yaml"),
        Path("electronics/power-system.yaml"),
        Path("electronics/logic-supply/verification.yaml"),
        Path("docs/design/references/vision/sources.json"),
    ]
    r["source_hashes"] = {}
    for path in dependencies:
        dest = output / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        r["source_hashes"][str(path)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    export_step(
        Compound(children=[p.shape for p in parts]), output / "candidate-assembly.step"
    )
    r["output_hashes"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file() and "source" not in p.relative_to(output).parts
    }
    (output / "report.json").write_text(json.dumps(r, indent=2) + "\n")
    (output / "report.sha256").write_text(
        hashlib.sha256((output / "report.json").read_bytes()).hexdigest() + "\n"
    )
    print(json.dumps(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-stack", action="store_true")
    args = parser.parse_args()
    run(args.output, args.step_stack)
