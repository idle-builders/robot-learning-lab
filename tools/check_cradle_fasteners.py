"""Nominal fastener stack around the corrected fit cradle; not a loaded fixture."""
import argparse
import json
from pathlib import Path
import yaml
from build123d import Compound, Cylinder, Pos, RegularPolygon, export_step, extrude
from lab.bench_fit import actuator_cradle
from tools.build_bench_fit import render, volume
from tools.check_camera_support import copy_source, write


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = ["lab/bench_fit.py", "lab/joint_module.py", "robot/bench-fit.yaml",
             "robot/actuator-catalog.yaml", "robot/fastener-catalog.yaml",
             "tools/check_cradle_fasteners.py", "tools/build_bench_fit.py"]
    sources = copy_source(output, {p: Path(p) for p in paths})
    bench = yaml.safe_load(Path("robot/bench-fit.yaml").read_text())
    parts, metrics = actuator_cradle(bench)
    top = parts["clamp_cap"].bounding_box().max.Z
    closure = 2 * bench["actuator_cradle"]["case_clearance"]
    washer = Cylinder(3.5, .5) - Cylinder(1.6, 1)
    nut = extrude(RegularPolygon(2.75, 6, major_radius=False), amount=2.4)
    nut -= Pos(0, 0, 1.2) * Cylinder(1.25, 4)
    checks, rows = [], []
    for state, travel in (("open", 0), ("nominally-closed", closure)):
        cap = Pos(0, 0, -travel) * parts["clamp_cap"]
        case = Pos(0, 0, -travel / 2) * parts["case_reference"]
        surface = top - travel
        for length in (35, 40, 45):
            # Screw introduced from beneath: washer [-.5,0], head [-3.5,-.5].
            tip = length - .5
            nut_bottom, nut_top = surface + .5, surface + .5 + 2.4
            projection = tip - nut_top
            # Two nominal pitches beyond the full nut envelope is a study criterion.
            # It is not a formal tolerance or strength approval.
            rows.append(dict(state=state, length_mm=length, nut_bottom_mm=nut_bottom,
                             nut_top_mm=nut_top, tip_mm=tip, tip_projection_mm=projection,
                             status="pass" if projection >= 1 else "fail"))
        hardware = {}
        for index, (x, y) in enumerate(metrics["clamp_bolt_centers_xy_mm"]):
            screw = Pos(x, y, 22) * Cylinder(1.5, 45)
            screw += Pos(x, y, -2) * Cylinder(2.75, 3)
            stack = dict(screw=screw, bottom_washer=Pos(x, y, -.25)*washer,
                         top_washer=Pos(x, y, surface+.25)*washer,
                         nut=Pos(x, y, surface+.5)*nut)
            for name, shape in stack.items():
                overlap = sum(volume(shape.intersect(stock))
                              for stock in (parts["cradle"], cap, case))
                checks.append(dict(state=state, index=index, part=name,
                                   stock_intersection_mm3=overlap,
                                   status="pass" if overlap < 1e-6 else "fail"))
                hardware[f"{name}_{index}"] = shape
        export_step(Compound(children=[parts["cradle"], cap, case, *hardware.values()]),
                    output / f"{state}.step")
        render({"cradle":parts["cradle"], "cap":cap, "case":case,
                "hardware":Compound(children=list(hardware.values()))},
               output, state, [-.0125,0,.02], .24)
    catalog = yaml.safe_load(Path("robot/fastener-catalog.yaml").read_text())["parts"]
    bom = [{"id":key,"quantity":quantity,"candidate":catalog[key]["procurement_candidate"],
            "source":catalog[key]["source"],"status":"conditional_dimensional_sample"}
           for key,quantity in (("screw_m3_45",4),("nut_m3",4),("washer_m3",8))]
    report = dict(status="partial", numerical_status="pass" if all(c["status"]=="pass" for c in checks) else "fail",
                  scope=__doc__, source_hashes=sources, checks=checks, screw_length_comparison=rows,
                  bom=bom, selected_length_mm=45, plate_to_cap_open_mm=top,
                  nominal_cap_closure_mm=closure, head_projection_below_plate_mm=3.5,
                  mounting_requirement="Heads project 3.5 mm below the plate: a flat bench mounting surface interferes. Provide clearance and support; fixture attachment/tool access remains unqualified.",
                  physical_status="not_run", thread_form="Envelope only; thread runout, chamfers, tolerances and actual engagement unmeasured",
                  not_run=["tightening torque and preload", "housing crush/slip", "support and fixture mounting", "tool access", "powered/loaded operation", "new X1 slice"])
    write(output / "report.json", report)
    write(output / "bom.json", bom)
    if report["numerical_status"] != "pass":raise ValueError("Hardware intersects fixture")
    print(json.dumps({k:report[k] for k in ("status","numerical_status","screw_length_comparison")}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(**vars(parser.parse_args()))
