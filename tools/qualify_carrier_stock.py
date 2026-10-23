"""Join bounded added-stock evidence to the exact carrier candidate exports."""

import argparse
import ast
from itertools import combinations
from pathlib import Path

from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def definition(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node, include_attributes=False)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.dump(node, include_attributes=False)
    raise ValueError(f"Missing {name} in {path}")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    scan_folder = Path("artifacts/carrier-robot-stock-v5")
    mutual_folder = Path("artifacts/carrier-mutual-stock-v2")
    module_folder = Path("artifacts/yoke-spacer-study-v2")
    scan, mutual, module = [
        verify_report(p) for p in (scan_folder, mutual_folder, module_folder)
    ]
    inputs = [p / "report.json" for p in (scan_folder, mutual_folder, module_folder)]
    if scan["status"] != "pass" or mutual["status"] != "pass":
        raise ValueError("Added stock has retained failures")
    archived_mapping = Path(scan["registered_mapping_source"])
    if sha(archived_mapping) != scan["registered_mapping_sha256"]:
        raise ValueError("Registered mapping source changed")
    if definition(
        scan_folder / "source/tools/check_carrier_robot_stock.py", "POSES"
    ) != definition(archived_mapping, "POSES"):
        raise ValueError("Mutual and robot scans use different poses")
    if definition(
        scan_folder / "source/lab/yoke_spacers.py", "linked_spacers"
    ) != definition(module_folder / "source/lab/yoke_spacers.py", "linked_spacers"):
        raise ValueError("Scanned carrier recipe differs from exported module")
    common_sources = sorted(set(scan["source_hashes"]) & set(mutual["source_hashes"]))
    for path in common_sources:
        if scan["source_hashes"][path] != mutual["source_hashes"][path]:
            raise ValueError(f"Stock and mutual shared source differs: {path}")
    if (
        scan["source_hashes"]["lab/leg_mechanism.py"]
        != module["source_hashes"]["lab/leg_mechanism.py"]
    ):
        raise ValueError("Ring recipe changed")
    rows = []
    for name in ("pip", "scout"):
        folder = Path(f"artifacts/{name}-carrier-composed-v1")
        candidate = verify_report(folder)
        inputs.append(folder / "report.json")
        source = candidate["source_assembly"]
        for f in ("assembly.step", "report.json", "resolved-config.json"):
            path = f"{source}/{f}"
            if scan["source_hashes"][path] != candidate["source_bindings"][path]:
                raise ValueError("Scan and composition source differ")
        if candidate["source_bindings"][str(module_folder / "report.json")] != sha(
            module_folder / "report.json"
        ):
            raise ValueError("Candidate module changed")
        robot = next(r for r in scan["robots"] if r["robot"] == name)
        names = {p["name"] for p in candidate["parts"]}
        if set(robot["module_ids"]) != set(candidate["replacements"]):
            raise ValueError("Scan and replacement modules differ")
        if (
            len(scan["poses"]) != 3
            or [r["pose"] for r in robot["poses"]] != scan["poses"]
        ):
            raise ValueError("Incomplete pose coverage")
        checks = 0
        for pose in robot["poses"]:
            expected = {
                (module, target)
                for module in robot["module_ids"]
                for target in names
                if target != module
            }
            actual = {(r["module"], r["target"]) for r in pose["rows"]}
            if (
                actual != expected
                or len(pose["rows"]) != len(expected)
                or pose["hits"]
                or any(
                    r["status"] != "pass" or r["intersection_volume_mm3"] > 1e-6
                    for r in pose["rows"]
                )
            ):
                raise ValueError("Incomplete or failed pair matrix")
            checks += len(actual)
        mutual_rows = [r for r in mutual["rows"] if r["robot"] == name]
        expected_mutual = {
            (pose, a, b)
            for pose in scan["poses"]
            for a, b in combinations(sorted(robot["module_ids"]), 2)
        }
        actual_mutual = {(r["pose"], r["module_a"], r["module_b"]) for r in mutual_rows}
        if (
            actual_mutual != expected_mutual
            or len(mutual_rows) != 198
            or any(r["status"] != "pass" for r in mutual_rows)
        ):
            raise ValueError("Incomplete mutual stock scan")
        if (
            candidate["geometry_build_status"] != "pass"
            or candidate["material_retention_status"] != "pass"
        ):
            raise ValueError("Candidate export identity failed")
        rows.append(
            dict(
                robot=name,
                assembly=str(folder),
                poses=scan["poses"],
                stock_vs_source_scene_pairs=checks,
                mutual_stock_pairs=len(mutual_rows),
                piece_counts=candidate["piece_counts"],
                added_stock_status="pass",
                inherited_whole_robot_status=candidate["status"],
            )
        )
    write(
        output / "report.json",
        dict(
            status="partial",
            added_stock_status="pass",
            scope="Exact candidate/source identity plus three-pose new-stock collision coverage against source assembly and mutual additions only",
            compared_common_source_hashes=common_sources,
            robots=rows,
            physical_status="not_run",
            full_carrier_clearance_status="not_run",
            source_bindings={str(p): sha(p) for p in inputs},
            source_hashes=copy_source(
                output, {str(Path(__file__).relative_to(Path.cwd())): Path(__file__)}
            ),
            limits=[
                "Passing new-stock checks does not qualify original rings or other retained pairs.",
                "No continuous motion, complete installation, screws/tools, harness, load or physical test.",
                "No candidate dynamics or learned policy result. Source assembly failures remain failed.",
            ],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
