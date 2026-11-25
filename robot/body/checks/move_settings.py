"""Move per-part settings from data/part_settings.json into the body definitions (phase 2 of Idle T-ghvncz): for every part
the named definition files claim, its blocks (process, expected_hardware, fit_classes, covers, exact_fit, loadpath_slices,
geometry_thresholds, fem, edges) go onto the claiming `parts` entry of its definition (a part on a multi-part entry is split
out onto its own entry, same line), and leave part_settings.json. One definition file (or section) at a time; tell the owner
first. The checks read the definition first (checks/ctx.py overlay_definitions), so a value left behind is shadowed, and the
bom check reports it.

  PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/move_settings.py --files <definition>.bom.yaml[,...] [--write]

Without --write it prints what would move. --write edits the definition files and part_settings.json in the working tree."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robot/body/bom")); sys.path.insert(0, str(ROOT / "robot/body"))
import definitions as DEF  # noqa: E402
from checks.ctx import SETTING_BLOCKS  # noqa: E402

PS = ROOT / "robot/body/checks/data/part_settings.json"


def load_yaml(path):
    import yaml
    text = path.read_text()
    head = "".join(l + "\n" for l in text.split("\n") if l.startswith("#"))
    return head, yaml.safe_load(text)


def dump_yaml(path, head, doc):
    import yaml
    path.write_text(head + yaml.safe_dump(doc, sort_keys=False, width=140, allow_unicode=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--files", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    files = a.files.split(",")
    man = ROOT / (a.manifest or __import__("bom_config").body_manifest())
    names = [p["name"] for p in json.loads(man.read_text())["parts"]]
    D = DEF.load()
    claims = D.claims(names)
    S = json.loads(PS.read_text())
    moved = {}
    docs = {f: load_yaml(ROOT / f) for f in files}
    for n, v in list(S.items()):
        if n.startswith("_"):
            continue
        e = claims.get(n)
        if not e or e["file"] not in files:
            continue
        blocks = {k: v[k] for k in SETTING_BLOCKS if k in v}
        if not blocks:
            continue
        head, doc = docs[e["file"]]
        entries = doc["parts"]
        cur = entries[e["index"]]
        hit = DEF.match(names, cur["match"])
        if len(hit) > 1:            # split the part out onto its own entry of the same line
            m = [x for x in DEF.names_of(cur["match"]) if x != n]
            if len(m) == len(DEF.names_of(cur["match"])):
                print(f"skip {n}: matched by a glob in {e['file']} parts[{e['index']}]; split it by hand"); continue
            cur["match"] = m if len(m) > 1 else m[0]
            if "quantity" in cur:
                cur["quantity"] -= 1
            new = {k: val for k, val in cur.items() if k not in SETTING_BLOCKS}
            new["match"] = n
            if "quantity" in cur:
                new["quantity"] = 1
            new.update(blocks)
            entries.append(new)
        else:
            for k, val in blocks.items():
                cur[k] = val
        moved[n] = (e["file"], sorted(blocks))
        for k in blocks:
            del S[n][k]
        if not S[n]:
            del S[n]
    for n, (f, ks) in moved.items():
        print(f"{n}: {', '.join(ks)} -> {f}")
    print(f"{len(moved)} parts, {sum(len(k) for _, k in moved.values())} blocks")
    if a.write and moved:
        for f, (head, doc) in docs.items():
            dump_yaml(ROOT / f, head, doc)
        PS.write_text(json.dumps(S, indent=2, ensure_ascii=False))          # the file's own format (no trailing newline)


if __name__ == "__main__":
    main()
