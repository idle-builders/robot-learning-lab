"""Inventory installed printed BOM groups against composed STEP solids."""

import argparse
import json
from pathlib import Path

from build123d import import_step

from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report

ROOT = Path(__file__).resolve().parents[1]


def dimensions(shape):
    box = shape.bounding_box()
    return [float(box.max.X - box.min.X), float(box.max.Y - box.min.Y), float(box.max.Z - box.min.Z)]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bom_path = ROOT / "robot/prototype-bom.json"
    bom = json.loads(bom_path.read_text())
    entries = [e for e in bom["entries"] if e.get("make_or_buy") == "make" and e.get("part_ids_by_robot")]
    report = {"status": "fail", "scope": __doc__.strip(), "source_reports": {}, "groups": []}
    paths = [Path(__file__).relative_to(ROOT), Path("robot/prototype-bom.json")]
    for name in ("pip", "scout"):
        paths += [Path(f"artifacts/{name}-integral-head-composed-v2/{n}") for n in ("report.json", "resolved-config.json", "assembly.step")]
    report["source_hashes"] = copy_source(output, {str(p): ROOT / p for p in paths})
    seen_by_robot = {}
    for name in ("pip", "scout"):
        assembly = ROOT / f"artifacts/{name}-integral-head-composed-v2"
        retained = verify_report(assembly)
        if retained.get("assembly_step_roundtrip") != "pass" or retained.get("output_hashes", {}).get("assembly.step") != sha(assembly / "assembly.step"):
            raise ValueError(f"{name} assembly STEP identity failed")
        shapes = list(import_step(assembly / "assembly.step").children)
        parts = parts_from_step(shapes, retained["parts"])
        by_name = {p.name: p for p in parts}
        if len(by_name) != len(parts):
            raise ValueError(f"{name} duplicate manifest part names")
        seen = {}
        robot_groups = []
        for entry in entries:
            ids = entry.get("part_ids_by_robot", {}).get(name, [])
            if not ids:
                continue
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} duplicate IDs in BOM group {entry['id']}")
            missing = [part_id for part_id in ids if part_id not in by_name]
            if missing:
                raise ValueError(f"{name} BOM group {entry['id']} missing assembly parts: {missing}")
            solids = []
            for part_id in ids:
                if part_id in seen:
                    raise ValueError(f"{name} duplicate group ownership for {part_id}: {seen[part_id]} and {entry['id']}")
                seen[part_id] = entry["id"]
                part = by_name[part_id]
                group_volume = float(part.shape.volume)
                actual_solids = list(part.shape.solids())
                if not actual_solids or group_volume <= 0:
                    raise ValueError(f"{name} invalid or zero-volume group {part_id}")
                group_solid_volume = 0.0
                for solid_index, solid in enumerate(actual_solids):
                    valid = bool(solid.is_valid)
                    volume = float(solid.volume)
                    if not valid or volume <= 0:
                        raise ValueError(f"{name} invalid or zero-volume solid {part_id}[{solid_index}]")
                    group_solid_volume += volume
                    solids.append({"part_id": part_id, "solid_index": solid_index, "owner": part.owner, "role": part.role, "volume_mm3": volume, "bounding_dimensions_mm": dimensions(solid), "valid": valid})
                if abs(group_solid_volume - group_volume) > max(0.01, group_volume * 1e-6):
                    raise ValueError(f"{name} solid volume mismatch for {part_id}")
            robot_groups.append({"group_id": entry["id"], "label": entry["label"], "group_count": len(ids), "solid_count": len(solids), "solids": solids, "total_volume_mm3": sum(s["volume_mm3"] for s in solids)})
        seen_by_robot[name] = seen
        actual_solid_count = sum(g["solid_count"] for g in robot_groups)
        report["source_reports"][name] = {"assembly": str((assembly / "assembly.step").relative_to(ROOT)), "manifest_part_count": len(parts), "step_child_count": len(shapes), "printed_group_count": len(seen), "printed_solid_count": actual_solid_count, "unmapped_assembly_parts": sorted(set(by_name) - set(seen))}
        report["groups"].append({"robot": name, "groups": robot_groups, "printed_group_count": len(seen), "printed_solid_count": actual_solid_count, "total_volume_mm3": sum(g["total_volume_mm3"] for g in robot_groups)})
    report["status"] = "pass"
    report["output_hashes"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "report.json"}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
