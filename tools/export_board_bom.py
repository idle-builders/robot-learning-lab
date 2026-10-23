"""Verify and present KiCad's native IPC-2581 BOM for a retained candidate board."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def component_rows(xml, parts, instances):
    """Keep native grouping/counts; resolve catalog identity from the bound manifest."""
    rows, seen, excluded = [], set(), []
    for item in ET.fromstring(xml).findall(".//{*}Bom/{*}BomItem"):
        refs = item.findall("{*}RefDes")
        names = [r.attrib["name"] for r in refs]
        if not names or int(item.attrib["quantity"]) != len(names):
            raise ValueError("native BOM quantity differs from reference count")
        if len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("duplicate native BOM reference")
        seen.update(names)
        if item.attrib["category"] == "DOCUMENT":
            if any(r.attrib["populate"] != "false" for r in refs):
                raise ValueError("populated documentary footprint")
            if set(names).intersection(instances):
                raise ValueError("expected component excluded from native BOM")
            excluded.extend(names)
            continue
        if any(r.attrib["populate"] != "true" for r in refs):
            raise ValueError("expected populated candidate component")
        if any(ref not in instances for ref in names):
            raise ValueError("unknown native component reference")
        definitions = {instances[ref] for ref in names}
        if len(definitions) != 1:
            raise ValueError("native group combines distinct catalog definitions")
        part = parts[definitions.pop()]
        values = [
            x.attrib["textualCharacteristicValue"]
            for x in item.findall("{*}Characteristics/{*}Textual")
            if x.attrib["textualCharacteristicName"] == "Value"
        ]
        if len(values) != 1 or values[0] not in (part["mpn"], part["value"]):
            raise ValueError("native PCB value differs from catalog definition")
        rows.append(dict(
            references=sorted(names), quantity=len(names), native_value=values[0],
            mpn=part["mpn"], manufacturer=part["manufacturer"],
            source=part["source"],
            identity_basis="native MPN" if values[0] == part["mpn"] else "bound manifest; native value is an alias",
        ))
    if seen - set(excluded) != set(instances):
        raise ValueError("native BOM missing expected components")
    return rows, sorted(excluded)


def export(board, stem, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", physical_status="not_run", source=str(board),
                  scope="Native component counts and bound catalog identities; not fabrication release")
    files = [board / "parts.json", board / "layout.json", board / (stem + ".kicad_pcb")]
    if (board / "components.json").exists():
        files.append(board / "components.json")
    report["source_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report["tool_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        command = [str(Path(".tools/KiCad/KiCad.app/Contents/MacOS/kicad-cli").resolve()),
                   "pcb", "export", "ipc2581", "--bom-col-mfg-pn", "Value", "--output",
                   str(output / "board.ipc2581.xml"), str(board / (stem + ".kicad_pcb"))]
        report["command"] = command
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (output / "native-export.log").write_text(process.stdout + process.stderr)
        report["native_exit_code"] = process.returncode
        process.check_returncode()
        parts = json.loads((board / "parts.json").read_text())
        instances = (json.loads((board / "components.json").read_text())
                     if (board / "components.json").exists() else
                     {r: c["part"] for r, c in json.loads((board / "layout.json").read_text())["components"].items()})
        rows, excluded = component_rows((output / "board.ipc2581.xml").read_text(), parts, instances)
        report.update(status="pass", component_count=sum(r["quantity"] for r in rows),
                      grouped_rows=len(rows), excluded_document_references=excluded)
        (output / "components.json").write_text(json.dumps(rows, indent=2) + "\n")
        lines = [f"# {stem}: candidate board components", "",
                 "Counts and grouping come from native KiCad IPC-2581 export and are checked against the retained source manifest. Catalog identities are candidates, not order release. Physical tests have not run.", "",
                 "| References | Quantity | Manufacturer / MPN | Identity basis |",
                 "|---|---:|---|---|"]
        for row in rows:
            lines.append(f"| {', '.join(row['references'])} | {row['quantity']} | {row['manufacturer']} [{row['mpn']}]({row['source']}) | {row['identity_basis']} |")
        lines += ["", "Documentary mounting footprints excluded: " + ", ".join(excluded) + ".",
                  "", "PCB, assembly/test, mating harness, installed mounting hardware, process losses and overbuy are outside this component list.",
                  "", "The prior Atopile JSON export is empty for these manually defined atomic parts; it is retained as an export limitation. No empty export is treated as a complete BOM."]
        (output / "components.md").write_text("\n".join(lines) + "\n")
    except Exception as error:
        report["error"] = str(error)
    report["output_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in output.iterdir() if p.is_file()}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, type=Path)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = export(args.board, args.stem, args.output)
    print(json.dumps(result))
    raise SystemExit(result["status"] != "pass")
