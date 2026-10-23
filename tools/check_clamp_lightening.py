"""Export subtractive clamp candidates and check explicit retained stock."""

import argparse
from pathlib import Path
import traceback

from build123d import Box, Pos, export_step, export_stl, import_step

from lab.clamp_lightening import candidates
from lab.leg_mechanism import servo_parts
from tools.check_camera_support import copy_source, sha, write


def volume(shape):
    if shape is None:
        return 0.0
    if isinstance(shape, (list, tuple)):
        return sum(volume(s) for s in shape)
    return sum(s.volume for s in shape.solids())


def missing(witness, candidate):
    return volume(witness.cut(candidate))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    files = [
        Path(p)
        for p in (
            "lab/clamp_lightening.py",
            "lab/joint_module.py",
            "lab/leg_mechanism.py",
            "tools/check_clamp_lightening.py",
            "robot/actuator-catalog.yaml",
            "docs/design/references/waveshare-st3215/ST3215.step",
        )
    ]
    report = dict(status="running", scope=__doc__, physical_status="not_run")
    report["source_hashes"] = copy_source(output, {str(p): p for p in files})
    try:
        shapes = candidates()
        stored = {}
        for name, alternatives in shapes.items():
            for label, shape in alternatives.items():
                key = f"{name}-{label}"
                export_step(shape, output / f"{key}.step")
                # STEP readback separates coincident topology for witnesses.
                stored[key] = import_step(output / f"{key}.step")
                if label in ("original", "recessed", "windowed"):
                    export_stl(
                        shape,
                        output / f"{key}.stl",
                        tolerance=0.04,
                        angular_tolerance=0.08,
                    )
        fixed, horns = servo_parts()
        for label, shape in (("servo", fixed), ("horns", horns)):
            export_step(shape, output / f"{label}.step")
            export_stl(
                shape, output / f"{label}.stl", tolerance=0.08, angular_tolerance=0.10
            )
        rows, negatives = [], []
        for name, sign in (("clamp_lower", -1), ("clamp_upper", 1)):
            original = stored[f"{name}-original"]
            seat = stored[f"{name}-seating_witness"]
            bolts = stored[f"{name}-bolt_stock_witness"]
            for label in ("original", "recessed", "windowed"):
                shape = stored[f"{name}-{label}"]
                metrics = dict(
                    added_volume_mm3=volume(shape.cut(original)),
                    removed_volume_mm3=volume(original.cut(shape)),
                    missing_seating_stock_mm3=missing(seat, shape),
                    missing_bolt_surround_mm3=missing(bolts, shape),
                    housing_overlap_mm3=volume(shape.intersect(fixed)),
                    horn_overlap_mm3=volume(shape.intersect(horns)),
                )
                okay = (
                    shape.is_valid
                    and len(shape.solids()) == 1
                    and all(
                        value < 1e-5
                        for key, value in metrics.items()
                        if key != "removed_volume_mm3"
                    )
                )
                rows.append(
                    dict(
                        name=name,
                        candidate=label,
                        status="pass" if okay else "fail",
                        volume_mm3=shape.volume,
                        retained_fraction=shape.volume / original.volume,
                        solids=len(shape.solids()),
                        valid=shape.is_valid,
                        **metrics,
                    )
                )
            chosen = stored[f"{name}-windowed"]
            for label, cutter, witness in (
                ("seat_cut", Pos(-12.5, sign * 12.9, 0) * Box(4, 4, 4), seat),
                ("bolt_cut", Pos(13.615, sign * 7, 8) * Box(8, 4, 8), bolts),
            ):
                loss = missing(witness, chosen - cutter)
                negatives.append(
                    dict(
                        name=name,
                        probe=label,
                        missing_mm3=loss,
                        status="pass" if loss > 1 else "fail",
                    )
                )
        report.update(
            status="partial"
            if all(r["status"] == "pass" for r in rows + negatives)
            else "fail",
            geometry_checks="pass"
            if all(r["status"] == "pass" for r in rows + negatives)
            else "fail",
            rows=rows,
            negative_controls=negatives,
            stock_definition={
                "seating_skin_mm": 1.2,
                "retained_wall_after_recess_mm": 1.5,
                "bolt_bore_radius_mm": 1.65,
                "protected_bolt_radius_mm": 3.65,
                "recess_mm": [34, 1.5, 18],
                "recess_corner_radius_mm": 2,
                "ear_windows_yz_mm": [12.1, 8],
                "ear_window_corner_radius_mm": 1.5,
            },
            not_run=[
                "clamp preload and friction",
                "strength, fatigue and creep",
                "print qualification",
                "new full robot composition/dynamics",
                "assembly tool access",
                "physical fit",
            ],
            limitations=[
                "Subtractive stock preserves selected nominal interfaces, not stiffness or retention.",
                "Vendor CAD proxy does not establish C018 mechanical interchange.",
                "Study not enabled in primary variants or BOM.",
            ],
        )
    except Exception as exc:
        report.update(status="fail", error=str(exc))
        (output / "error.log").write_text(traceback.format_exc())
    report["output_hashes"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "report.json", report)
    print(
        {k: report[k] for k in ("status", "geometry_checks", "error") if k in report},
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(p.parse_args().output)["status"] == "fail")
