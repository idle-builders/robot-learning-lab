"""Compare retained complete assemblies with linked-carrier candidate overlays."""

import argparse
import ast
import base64
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

from build123d import import_step
from lab.artifact_store import retain_dependency
from lab.config import RobotConfig
from lab.robot import make_robot
from tools.build_product_review import wrapper
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.review_integral_head_service import embedded_dataset, add_asset


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__).relative_to(Path.cwd()),
        Path("tools/build_product_review.py"),
        Path("tools/review_integral_head_service.py"),
        Path("tools/cap_mesh.py"),
        Path("ui/composed-head.fragment.html"),
        Path("ui/product-review.css"),
        Path("lab/config.py"),
        Path("lab/robot.py"),
    ]
    sources = copy_source(output, {str(p): p for p in paths})
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    original = Path("artifacts/integral-head-composed-review-v2")
    retained = verify_report(original)
    bind(original / "report.json")
    data = embedded_dataset(bind(original / "index.html"))
    registered_folder = Path("artifacts/carrier-registration-v1")
    registered = verify_report(registered_folder)
    bind(registered_folder / "report.json")
    if registered["status"] != "pass":
        raise ValueError("Original ring display registration failed")
    qualified_folder = Path("artifacts/carrier-stock-qualification-v2")
    qualified = verify_report(qualified_folder)
    bind(qualified_folder / "report.json")
    if qualified["added_stock_status"] != "pass":
        raise ValueError("Carrier stock qualification has no passing bounded result")
    scan_folder = Path("artifacts/carrier-robot-stock-v5")
    verify_report(scan_folder)
    bind(scan_folder / "report.json")
    scan_source = bind(scan_folder / "source/tools/check_carrier_robot_stock.py")
    tree = ast.parse(scan_source.read_text())
    poses = ast.literal_eval(
        next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "POSES" for t in n.targets)
        )
    )
    candidates = []
    for baseline in data["robots"]:
        name = baseline["name"].split(" · ")[0].lower()
        folder = Path(f"artifacts/{name}-carrier-composed-v1")
        report = verify_report(folder)
        bind(folder / "report.json")
        if report["geometry_build_status"] != "pass":
            raise ValueError("Candidate geometry export failed")
        source_report = report["source_assembly"] + "/report.json"
        if (
            report["source_bindings"][source_report]
            != retained["source_bindings"][source_report]
        ):
            raise ValueError("Baseline display differs from candidate source")
        by_name = {p["name"]: p for p in report["parts"]}
        if set(by_name) != {p["name"] for p in baseline["parts"]}:
            raise ValueError("Displayed part inventory differs")
        if report["source_bindings"][str(registered_folder / "report.json")] != sha(
            registered_folder / "report.json"
        ):
            raise ValueError("Ring mapping differs from candidate composition")
        for part in baseline["parts"]:
            if part["name"] in report["replacements"]:
                path = bind(registered_folder / f"{name}-{part['name']}-mapped.step")
                filename = f"{name}-{part['name']}-original.stl"
                part["asset"] = add_asset(data, import_step(path), output / filename)[0]
                part["download"] = filename
        candidate = deepcopy(baseline)
        for part in candidate["parts"]:
            if part["owner"] != by_name[part["name"]]["owner"]:
                raise ValueError("Part frame changed")
            if part["name"] in report["replacements"]:
                step = bind(folder / f"{part['name']}.step")
                filename = f"{name}-{part['name']}.stl"
                part["asset"] = add_asset(data, import_step(step), output / filename)[0]
                part["download"] = filename
            else:
                part.pop("download", None)
        counts = report["piece_counts"]
        candidate.update(
            name=name.capitalize() + " · linked carriers",
            source=str(folder),
            inherited_status=report["status"],
            stock_summary=f"{report['part_count']} CAD groups · {counts['candidate_printed_solid_count']} printed solids · 72 fewer than baseline · geometry export PASS · full robot FAIL · physical NOT RUN",
        )
        if qualified["source_bindings"][str(folder / "report.json")] != sha(
            folder / "report.json"
        ):
            raise ValueError("Qualified candidate changed")
        if report["source_hashes"]["lab/robot.py"] != sha(Path("lab/robot.py")):
            raise ValueError("Pose kinematics changed")
        cfg = RobotConfig.model_validate_json(
            bind(folder / "resolved-config.json").read_text()
        )
        robot = make_robot(cfg)
        candidate["clearance_label"] = "Added carrier stock in this pose"
        candidate["samples"] = []
        for pose_name, overrides in poses:
            fk = robot.forward_kinematics(overrides)
            candidate["samples"].append(
                dict(
                    name=pose_name,
                    kind="pose",
                    phase="",
                    status="pass",
                    clearance_status="pass",
                    flagged=[],
                    pair=["No added carrier stock intersection"],
                    gap_note="Added carrier stock only, using conservative surface bounds and exact non-pruned intersections. Original part pairs, continuous motion, assembly/tool paths and physical fit remain unqualified.",
                    frames=[
                        fk[owner].flatten().tolist() for owner in candidate["owners"]
                    ],
                )
            )
        baseline["clearance_label"] = "Original head composition at neutral"
        baseline["name"] = name.capitalize() + " · original rings"
        baseline["stock_summary"] = (
            f"{len(baseline['parts'])} CAD groups · {counts['source_printed_solid_count']} printed solids · original eight-ring groups retained · whole robot FAIL · physical NOT RUN"
        )
        candidates.extend([candidate, baseline])
    data["robots"] = candidates
    encoded = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode())
    ).decode()
    template = Path("ui/composed-head.fragment.html").read_text()
    template = template.replace(
        "Complete robot candidates · geometry and planning mass",
        "Complete robots · spacer assembly comparison",
    )
    template = template.replace(
        'data-id="context" type="checkbox" checked', 'data-id="context" type="checkbox"'
    )
    template = template.replace(
        "Neutral changed-stock screen:", "${robot.clearance_label}:"
    )
    start = template.index('  <p><a href="../pip-integral-head-composed')
    end = template.index('  <div data-id="error"', start)
    template = (
        template[:start]
        + """  <p><a href="../pip-carrier-composed-v1/assembly.step">Pip candidate STEP</a> · <a href="../scout-carrier-composed-v1/assembly.step">Scout candidate STEP</a> · <a href="../pip-carrier-composed-v1/report.json">Pip composition evidence</a> · <a href="../scout-carrier-composed-v1/report.json">Scout composition evidence</a> · <a href="../yoke-spacer-review-v2/index.html">Module fit and installation studies</a> · <a href="../carrier-stock-qualification-v2/report.json">Three-pose added-stock qualification</a></p>
  <details><summary>Scope and source</summary><div class="detail text-small" data-id="source"></div><p>Twelve groups per robot replace eight loose rings with two four-ring carriers. The two gripper groups retain sixteen loose rings. These complete candidate exports preserve the previous robot failures. They do not update the primary variants, physical BOM, mass/inertia models or control results. Strength, creep, continuous installation, screws/tools and physical fit remain unqualified. Imported body meshes are display approximations; download STEP for source geometry.</p></details>
"""
        + template[end:]
    )
    (output / "index.html").write_text(
        wrapper(template.replace("__HEAD_MOUNT_DATA__", encoded))
    )
    (output / "review.css").write_text(Path("ui/product-review.css").read_text())
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Presentation integrity only; robot qualification remains failed/incomplete",
            source_hashes=sources,
            source_bindings=bindings,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )


def run_covered(output, pip, scout):
    """Reuse the existing shin inspector for verified complete covered assemblies."""
    output.mkdir(parents=True, exist_ok=False)
    original = Path('artifacts/shin-cover-review-v1')
    original_report = verify_report(original)
    data = embedded_dataset(original/'index.html')
    clearance_folder = Path('artifacts/covered-carriers-clearance-v2')
    clearance = verify_report(clearance_folder)
    if clearance['status'] != 'pass':
        raise ValueError('Retained named-pair screen did not pass')
    bindings = {str(p):sha(p) for p in [original/'report.json',original/'index.html',clearance_folder/'report.json']}
    candidates = []
    dependencies = {}
    for name,folder in [('pip',pip),('scout',scout)]:
        report = verify_report(folder)
        if report.get('geometry_build_status') != 'pass' or report.get('cost_inventory_agreement',{}).get('status') != 'pass':
            raise ValueError('Complete assembly/export/inventory is not verified')
        delta = Path(report['matched_mass_source'])
        source = Path(report['source_assembly'])
        source_report = verify_report(source)
        for shared in ['artifacts/shin-cover-study-v1/report.json','artifacts/shin-cover-qualification-v1/report.json']:
            if source_report['source_bindings'][shared] != original_report['source_bindings'][shared]:
                raise ValueError('Displayed shin source differs from complete assembly')
        for shared in [str(delta/'report.json'),str(source/'report.json')]:
            if clearance['source_bindings'][shared] != report['source_bindings'][shared]:
                raise ValueError('Named-pair clearance describes different components')
        candidates_in_source = [r for r in data['robots'] if r['name'].lower().startswith(name+' ·') and 'shin covers' in r['name']]
        if len(candidates_in_source) != 1:raise ValueError('Ambiguous covered display source')
        item = deepcopy(candidates_in_source[0])
        records = {p['name']:p for p in report['parts']}
        if set(records) != {p['name'] for p in item['parts']}:
            raise ValueError('Complete CAD and display inventories differ')
        for part in item['parts']:
            if part['owner'] != records[part['name']]['owner']:
                raise ValueError('Display owner differs from composed CAD')
            part.pop('download',None)
            if part['name'] in report['replacements']:
                step=folder/(part['name']+'.step')
                # Local carrier inputs must match the exact cleared delta bytes.
                if sha(step) != clearance['source_bindings'][str(delta/(part['name']+'.step'))]:
                    raise ValueError('Displayed carrier is not the checked component')
                path=output/(name+'-'+part['name']+'.stl')
                part['asset']=add_asset(data,import_step(step),path)[0]
                part['download']=path.name
                bindings[str(step)]=sha(step)
        config=RobotConfig.model_validate_json((folder/'resolved-config.json').read_text())
        robot=make_robot(config)
        if config.appearance != name:raise ValueError('Wrong robot assembly selected')
        item['samples']=[]
        for pose in clearance['robots'][name]:
            frames=robot.forward_kinematics(pose['joints_rad'])
            item['samples'].append(dict(name=pose['pose'],kind='pose',phase='',status=pose['status'],clearance_status=pose['status'],flagged=[],pair=['No carrier/shin overlap in checked pairs'],gap_note='Only the twelve carrier groups versus twenty added/replaced shin groups at this fixed pose. Other pairs, continuous paths, tool access, full contacts and physical fit remain unqualified.',frames=[frames[o].flatten().tolist() for o in item['owners']]))
        count=report['piece_counts']['candidate_printed_solid_count']
        masses=report['mass_profiles']
        item.update(name=name.title()+' · covers + carriers',source=str(folder),inherited_status=report['status'],assembly_part_count=len(records),stock_summary=f"{len(records)} CAD groups · {count} printed solids · STEP identity and cost inventory PASS · planning mass {masses['lower']['mass_kg']:.3f}/{masses['nominal']['mass_kg']:.3f}/{masses['upper']['mass_kg']:.3f} kg · whole robot FAIL · physical NOT RUN")
        bindings[str(folder/'report.json')]=sha(folder/'report.json')
        dependencies[name]=retain_dependency(folder)
        candidates.append(item)
    data['robots']=candidates
    encoded=base64.b64encode(gzip.compress(json.dumps(data,separators=(',',':')).encode())).decode()
    template=Path('ui/shin-cover.fragment.html').read_text().replace('Printed shin covers · complete robot comparison','Pip and Scout · complete covered-carrier candidates').replace('Added-cover screen:', 'Named carrier/shin pairs:')
    start=template.index('  <p><a href="../shin-cover-study-v1')
    end=template.index('  <div data-id="error"',start)
    links=' · '.join(f'<a href="{os.path.relpath(folder/output_name,output)}">{label}</a>' for folder,output_name,label in [(pip,'assembly.step','Pip complete STEP'),(scout,'assembly.step','Scout complete STEP'),(pip,'report.json','Pip assembly evidence'),(scout,'report.json','Scout assembly evidence')])
    template=template[:start]+f'''  <p>{links}</p>
  <p><a href="../covered-carriers-clearance-v2/report.json">Named-pair clearance</a> · <a href="../../docs/design/covered-carriers.md">Mass, costs, build plan and limits</a></p>
  <details><summary>Scope and source</summary><div class="detail text-small" data-id="source"></div><p>Full composed CAD matches the configured carriers, covered mass models and candidate cost inventory. Existing whole-robot failures remain. Covers have open backs; wiring, complete protection, tool access, strength and physical fit remain open. Display meshes approximate unchanged STEP surfaces; no new contact model or training is implied.</p></details>
'''+template[end:]
    (output/'index.html').write_text(wrapper(template.replace('__HEAD_MOUNT_DATA__',encoded)))
    (output/'review.css').write_bytes(Path('ui/product-review.css').read_bytes())
    files=[Path(__file__).relative_to(Path.cwd()),Path('tools/review_integral_head_service.py'),Path('tools/cap_mesh.py'),Path('ui/shin-cover.fragment.html'),Path('ui/product-review.css')]
    sources=copy_source(output,{str(p):p for p in files})
    write(output/'report.json',dict(status='pass',scope='Display/source integrity for the complete composed candidates; whole-product and physical qualification remain open',source_hashes=sources,source_bindings=bindings,dependency_manifests=dependencies,output_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--covered-pip", type=Path)
    p.add_argument("--covered-scout", type=Path)
    args = p.parse_args()
    if args.covered_pip or args.covered_scout:
        if not (args.covered_pip and args.covered_scout):
            p.error("both covered assemblies are required")
        run_covered(args.output, args.covered_pip, args.covered_scout)
    else:
        run(args.output)
