"""Build a local paired review without concealing failed enclosure probes."""

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(covers, previews, output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def link(path, label, image=False):
        bindings[str(path)] = sha(path)
        target = escape(os.path.relpath(path, output))
        return (
            f'<img src="{target}" alt="{escape(label)}">'
            if image
            else f'<a href="{target}">{escape(label)}</a>'
        )

    cards = []
    for cover, preview in zip(covers, previews, strict=True):
        report = json.loads((cover / "report.json").read_text())
        view = json.loads((preview / "report.json").read_text())
        if view["source_bindings"][str(cover / "report.json")] != sha(
            cover / "report.json"
        ):
            raise ValueError("preview uses another cover report")
        for root, r in [(cover, report), (preview, view)]:
            for name, digest in r["output_hashes"].items():
                if sha(root / name) != digest:
                    raise ValueError(f"changed output {name}")
        for name, digest in report["source_hashes"].items():
            if sha(cover / "source" / name) != digest:
                raise ValueError(f"changed source {name}")
        title = Path(report["variant"]).stem.removesuffix("-head-service").title()
        rows = []
        for pose in report["poses"]:
            cases = [r for r in report["rows"] if r["pose"] == pose]
            failed = [r for r in cases if r["status"] == "fail"]
            closest = min(cases, key=lambda r: r["distance_mm"])
            rows.append(
                f"<tr><td>{escape(pose)}</td><td>{'fail' if failed else 'pass'}</td><td>{closest['distance_mm']:.2f} mm</td><td>{escape(closest['nearest'])}</td></tr>"
            )
        mass = sum(v["total_added_g"] for v in report["mass"].values())
        services = sum(r["status"] == "pass" for r in report["service"])
        card = f"""<section><h2>{title}</h2><p><strong>Candidate outcome: {escape(report["status"])}</strong> · added planning mass {mass:.1f} g per robot.</p>
        {link(preview / "preview.png", title + " with experimental forearm covers", True)}
        <p>This full-robot overlay is visual evidence only. Its cover checks use detailed arms, not the complete torso/head/leg assembly.</p>
        <details><summary>Inspect detailed arm assembly</summary>{link(cover / "mechanism.png", "Detailed arms and cover hardware", True)}</details>
        <p>STEP named-part round trip: {"pass" if report["step_identity_pass"] else "fail"}. Neutral removal samples: {services}/{len(report["service"])} pass. Non-interface clearance target: {report["clearance_target_mm"]} mm.</p>
        <details><summary>Inspect all {len(report["poses"])} pose outcomes</summary><table><tr><th>Pose</th><th>Outcome</th><th>Nearest gap</th><th>Neighbor</th></tr>{"".join(rows)}</table></details>
        <p>{link(cover / "report.json", "Exact checks and source identities")} · {link(preview / "posed-cover-parts.step", "Placed cover parts STEP")} · {link(preview / "part-frames.json", "Part ownership and frames")} · {link(preview / "report.json", "Preview provenance")}</p></section>"""
        cards.append(card)
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forearm enclosure review</title>
    <style>body{background:#edf0eb;color:#203934;font:16px/1.5 system-ui;margin:0}main{max-width:1200px;margin:auto;padding:30px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;border-radius:18px;padding:22px}img{width:100%;border-radius:12px}table{font-size:12px;border-collapse:collapse}td,th{padding:6px;text-align:left;border-bottom:1px solid #ddd}details{overflow:auto;margin:14px 0}summary{cursor:pointer}a{color:#206b61}@media(max-width:800px){.cards{grid-template-columns:1fr}}</style>
    <main><h1>Removable forearm enclosures</h1><p>Two open-ended shells per arm, with four screws into separate spine-side inserts. This is a design experiment. Failed motion cases remain visible; the covers have not been promoted into the passed integrated robot assemblies or control trials.</p>
    <div class="cards">"""
        + "".join(cards)
        + """</div><p>Continuous motion, pinch protection, harness routing, physical retention, full-body clearance and manufacturing costs remain unqualified. Hardware and polymer masses use planning densities. The approximately $1,000 production selling-price target remains open.</p></main></html>"""
    )
    (output / "review.html").write_text(html)
    script = Path(__file__).resolve()
    saved = output / "source" / script.name
    saved.parent.mkdir()
    shutil.copy2(script, saved)
    r = dict(
        status="pass",
        scope="review integrity, not mechanical approval",
        source_bindings=bindings,
        source_hashes={script.name: sha(saved)},
        output_hashes={"review.html": sha(output / "review.html")},
    )
    (output / "report.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(dict(status="pass", output=str(output))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--covers", nargs="+", type=Path, required=True)
    p.add_argument("--previews", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.covers, a.previews, a.output)
