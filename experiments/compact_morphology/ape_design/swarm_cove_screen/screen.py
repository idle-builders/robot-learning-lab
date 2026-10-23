"""Bounded neutral proximal package screen for the committed 500 mm Cove shell.

Run from the repository root:
    .venv/bin/python experiments/compact_morphology/ape_design/swarm_cove_screen/screen.py

The source shapes come from swarm_shell_v2/build.py; this file changes no design.
Each OCCT operation runs in a child process with a 20 second limit. Units are mm.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE.parent / "swarm_shell_v2" / "build.py"
REPORT = HERE / "report.json"
LIMIT_SECONDS = 20
FUZZY_MM = 0.01


def probe(package_path: str, shell_path: str, method: str, center: list[float]) -> dict:
    """Run one exact OCCT operation without importing the source builder."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepTools import BRepTools
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS_Shape
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.gp import gp_Pnt

    def read(path: str):
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, path, BRep_Builder()):
            raise RuntimeError(f"BREP read failed: {path}")
        return shape

    def volume(shape):
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return props.Mass()

    def solid_count(shape):
        n = 0
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            n += 1
            explorer.Next()
        return n

    package, shell = read(package_path), read(shell_path)
    if method == "distance":
        distance = BRepExtrema_DistShapeShape(package, shell)
        distance.Perform()
        classifier = BRepClass3d_SolidClassifier(shell, gp_Pnt(*center), 1e-6)
        return {
            "done": distance.IsDone(),
            "boundary_distance_mm": distance.Value() if distance.IsDone() else None,
            "package_center_in_shell": str(classifier.State()).split(".")[-1],
            "package_valid": BRepCheck_Analyzer(package).IsValid(),
            "shell_valid": BRepCheck_Analyzer(shell).IsValid(),
            "package_solids": solid_count(package),
            "shell_solids": solid_count(shell),
            "package_volume_mm3": volume(package),
            "shell_volume_mm3": volume(shell),
        }
    first, second = (shell, package) if method.endswith("ba") else (package, shell)
    operation = BRepAlgoAPI_Common(first, second)
    if method.startswith("fuzzy"):
        operation.SetFuzzyValue(FUZZY_MM)
    operation.Build()
    if not operation.IsDone():
        return {"done": False, "intersection_mm3": None}
    result = operation.Shape()
    return {
        "done": True,
        "result_valid": BRepCheck_Analyzer(result).IsValid(),
        "result_solids": solid_count(result),
        "intersection_mm3": volume(result) if not result.IsNull() else 0.0,
    }


def run_probe(package_path: Path, shell_path: Path, method: str, center: list[float]) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--probe",
               str(package_path), str(shell_path), method, json.dumps(center)]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=LIMIT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {LIMIT_SECONDS}s"}
    if result.returncode:
        return {"error": (result.stderr or result.stdout).strip()[-500:]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"non-JSON probe result: {result.stdout[-300:]}"}


def main() -> None:
    import build123d as b

    rel_source = SOURCE.relative_to(ROOT)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel_source)],
                           cwd=ROOT).returncode == 0
    if not clean:
        raise RuntimeError(f"source differs from committed HEAD: {rel_source}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    spec = importlib.util.spec_from_file_location("cove_committed_builder", SOURCE)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    variant = next(v for v in builder.VARIANTS if v.name == "cove")
    parts, _ = builder.build(variant)
    shells = {part.name: part.shape for part in parts}
    packages = {name: shape for name, shape, _ in builder.packages(variant)}
    pairs = []
    for side in ("L", "R"):
        pairs.extend([
            (f"{side}_hip_RSF25x35p8", "torso", [(-1 if side == "L" else 1) * 14, 0, 228.9]),
            (f"{side}_hip_RSF25x35p8", "pelvis_v", [(-1 if side == "L" else 1) * 14, 0, 228.9]),
            (f"{side}_hip_RSF25x35p8", f"{side}_thigh", [(-1 if side == "L" else 1) * 14, 0, 228.9]),
            (f"{side}_shoulder_RSF25x35p8", "torso", [(-1 if side == "L" else 1) * 44, 0, 335.9]),
        ])
    methods = ("distance", "common_ab", "common_ba", "fuzzy_ab", "fuzzy_ba")
    records = []
    with tempfile.TemporaryDirectory(prefix="cove-screen-") as directory:
        temp = Path(directory)
        paths = {}
        for name, shape in {**shells, **packages}.items():
            if name not in {name for pair in pairs for name in pair[:2]}:
                continue
            path = temp / f"{name}.brep"
            if not b.export_brep(shape, path):
                raise RuntimeError(f"BREP export failed: {name}")
            paths[name] = path
        for package_name, shell_name, center in pairs:
            checks = {method: run_probe(paths[package_name], paths[shell_name], method, center)
                      for method in methods}
            distance = checks["distance"]
            exact = [checks[name] for name in ("common_ab", "common_ba")]
            fuzzy = [checks[name] for name in ("fuzzy_ab", "fuzzy_ba")]
            sound = (distance.get("done") and distance.get("package_valid") and
                     distance.get("shell_valid") and distance.get("package_solids") == 1 and
                     distance.get("shell_solids") == 1 and
                     all(q.get("done") and q.get("result_valid") for q in exact + fuzzy))
            volumes = [q.get("intersection_mm3") for q in exact]
            fuzzy_volumes = [q.get("intersection_mm3") for q in fuzzy]
            if (sound and min(volumes) > 0.05 and
                    abs(volumes[0] - volumes[1]) < 0.1):
                verdict = "exact_penetration"
            elif (sound and max(volumes + fuzzy_volumes) < 0.05 and
                  distance["boundary_distance_mm"] > FUZZY_MM and
                  distance["package_center_in_shell"] == "TopAbs_OUT" and
                  distance["shell_volume_mm3"] > distance["package_volume_mm3"]):
                verdict = "positive_gap"
            else:
                verdict = "unknown"
            record = {"package": package_name, "shell": shell_name,
                      "verdict": verdict, "checks": checks}
            records.append(record)
            d = distance.get("boundary_distance_mm")
            print(f"{package_name} / {shell_name}: {verdict}; "
                  f"boundary distance {d:.3f} mm" if d is not None else
                  f"{package_name} / {shell_name}: {verdict}; distance unknown", flush=True)
    report = {
        "source": {"path": str(rel_source), "git_commit": commit,
                   "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   "variant": "cove", "coordinate_unit": "mm"},
        "method": {"operands": "source package BRep and named source shell BRep",
                   "operation_timeout_seconds": LIMIT_SECONDS,
                   "fuzzy_tolerance_mm": FUZZY_MM,
                   "gap_rule": "both exact and fuzzy Commons empty; valid single-solid operands; positive exact boundary distance; package center outside shell; shell volume exceeds package volume",
                   "meaning": "neutral static source geometry only; no motion, supports, mounts, manufacture tolerance, or assembly clearance"},
        "pairs": records,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        print(json.dumps(probe(sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5]))))
    else:
        main()
