"""Display covered-assembly production assumptions and staged candidate inventory."""

import argparse
import json
import os
from pathlib import Path
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output, analysis, composed_pip=None, composed_scout=None, assembly_review=None):
    output.mkdir(parents=True, exist_ok=False)
    report = verify_report(analysis)
    if report["status"] != "planning_estimate" or len(report["robots"]) != 2:
        raise ValueError("Cost analysis incomplete")
    source_paths = [
        analysis / "report.json",
        analysis / "prototype-bom.json",
    ]
    data = dict(
        robots=report["robots"],
        assumptions=report["assumptions"],
        contingency=report["cost_assumptions"]["contingency_fraction"],
        target_usd=report["cost_assumptions"]["target_usd"],
        bom=json.loads((analysis / "prototype-bom.json").read_text()),
        hardware=report["study"]["hardware"],
        scope=report.get("scope", "Covered-shin candidate inventory; complete assembly and physical qualification remain open."),
    )
    for robot in data["robots"]:
        if (
            robot["cost"]["inventory_groups"]
            != data["bom"]["inventory_counts"][robot["name"]]
        ):
            raise ValueError("Displayed BOM count differs")
        robot["source_assembly"] = report["study"]["assemblies"][robot["name"]]
    if composed_pip is not None or composed_scout is not None:
        if composed_pip is None or composed_scout is None:
            raise ValueError("Both composed robot inputs are required")
        for robot in data["robots"]:
            folder = {"pip": composed_pip, "scout": composed_scout}[robot["name"]]
            if json.loads((folder / "resolved-config.json").read_text())["appearance"] != robot["name"]:
                raise ValueError("Composed robot identity differs from cost candidate")
            complete = verify_report(folder)
            agreement = complete.get("cost_inventory_agreement", {})
            if (complete.get("geometry_build_status") != "pass" or agreement.get("status") != "pass"
                    or agreement.get("source") != str(analysis)
                    or agreement.get("printed_solids") != robot["inventory"]["printed_solid_count"]):
                raise ValueError("Complete assembly does not match this candidate cost inventory")
            robot["source_assembly"] = str(folder)
            source_paths.append(folder / "report.json")
        if assembly_review is None:
            raise ValueError("A matching assembly review is required for composed candidates")
        presented = verify_report(assembly_review)
        for folder in [composed_pip, composed_scout]:
            if presented["source_bindings"].get(str(folder / "report.json")) != sha(folder / "report.json"):
                raise ValueError("Assembly inspector describes different composed candidates")
        source_paths.append(assembly_review / "report.json")
        data["assembly_view"] = os.path.relpath(assembly_review / "index.html", output)
        data["composed"] = True
        data["scope"] = "Complete shin-covered robots with configured linked carriers, matched to the retained mass and production-cost inventory. Prices and physical fit remain unqualified."
    template = Path("ui/covered-production-cost.html")
    (output / "index.html").write_text(
        template.read_text().replace("../covered-production-cost-v3/report.json", os.path.relpath(analysis / "report.json", output)).replace(
            "__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c")
        )
    )
    files = [
        Path(__file__).relative_to(Path.cwd()),
        template,
        Path("tools/compose_forearm_enclosure.py"),
        Path("docs/design/covered-production-budget.md"),
    ]
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Presentation integrity only; production cost and physical qualification remain open",
            price_qualified=False,
            physical_status="not_run",
            source_bindings={str(p): sha(p) for p in source_paths},
            source_hashes=copy_source(output, {str(p): p for p in files}),
            output_hashes={"index.html": sha(output / "index.html")},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--composed-pip", type=Path)
    p.add_argument("--composed-scout", type=Path)
    p.add_argument("--assembly-review", type=Path)
    a = p.parse_args()
    run(a.output, a.analysis, a.composed_pip, a.composed_scout, a.assembly_review)
