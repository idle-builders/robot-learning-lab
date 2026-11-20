"""Recreate the selected body manifest from scripts and inputs in an isolated copy, and compare it with the selected one.

The ordered builder steps are data: robot/body/bom/chain.json. This script

  setup      makes WORK/repo: a copy of the working tree's code (rsync, without artifacts/, .cache/, .venv*, data/, .git,
             .tools, ...), with .venv, .tools, data and .artifact-store linked from the main checkout, and an artifacts/
             tree in which every top-level folder and every artifacts/robot/* folder except body is a link to the main
             checkout's, artifacts/robot/body/{arm,neck-head,torso} and the leg section outputs are links (read-only
             inputs, chain.json section_inputs), and the folders the chain writes start empty. Scripts derive ROOT from
             __file__, so every output lands inside the copy.
  --dry-run  lists the steps and checks that every declared input resolves (on disk, through infra/storage/path-moves.json,
             or as a file an earlier step writes) without running anything.
  --step ID  runs one step in the copy under infra/tools/memguard.py (waits until no other memguard job runs first).
  run        runs every step from --from ID (default the first) in order and stops at the first failure.
  compare    compares the copy's manifests with the main checkout's (the final one against the snapshot of the selected
             manifest taken at setup): part names added / removed, native STEP and mesh sha256 (raw and content: STEP
             without its HEADER, npz by array bytes), bounds difference, owning link / role changes, counts per kind.
             Scores and counts only.

Results: WORK/runs.json (per step: return code, seconds, memguard peak GB, log), WORK/compare.json and WORK/compare.md.
WORK defaults to .cache/work/body-rebuild (derived, delete when done).

  .venv/bin/python -B robot/body/bom/rebuild_chain.py setup
  .venv/bin/python -B robot/body/bom/rebuild_chain.py --dry-run
  .venv/bin/python -B robot/body/bom/rebuild_chain.py --step assembly
  .venv/bin/python -B robot/body/bom/rebuild_chain.py run [--from ID]
  .venv/bin/python -B robot/body/bom/rebuild_chain.py compare
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHAIN = json.loads((Path(__file__).resolve().parent / "chain.json").read_text())
WORK = Path(os.environ.get("BODY_REBUILD_WORK", ROOT / ".cache/work/body-rebuild"))
REPO = WORK / "repo"
RUNS = WORK / "runs.json"
SNAPSHOT = WORK / "selected" / "assembly-parts-roll-flip.json"

RSYNC_EXCLUDE = ["/artifacts/", "/.cache/", "/.venv/", "/.venv-control/", "/.venv-runpod/", "/data/", "/.git/", "/.tools/",
                 "node_modules/", "/.artifact-store/", "/.artifact-incoming/", "/.public-upstream/", "/.bookkeeping/",
                 "/.tmp/", ".DS_Store", "__pycache__/"]
LINK_TOP = [".venv", ".tools", "data", ".artifact-store"]


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(src, dst)


def setup():
    WORK.mkdir(parents=True, exist_ok=True)
    REPO.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--delete"] + [f"--exclude={e}" for e in RSYNC_EXCLUDE] + [f"{ROOT}/", f"{REPO}/"]
    # keep the artifacts tree and caches made inside the copy across re-syncs
    cmd[3:3] = ["--filter=P /artifacts/", "--filter=P /.cache/"] + [f"--filter=P /{n}" for n in LINK_TOP]
    sh(cmd)
    for n in LINK_TOP:
        if (ROOT / n).exists():
            link(REPO / n, ROOT / n)
    art = REPO / "artifacts"
    art.mkdir(exist_ok=True)
    for p in sorted((ROOT / "artifacts").iterdir()):
        if p.name != "robot":
            link(art / p.name, p)
    for p in sorted((ROOT / "artifacts/robot").iterdir()):
        if p.name != "body":
            link(art / "robot" / p.name, p)
    (art / "robot/body").mkdir(parents=True, exist_ok=True)
    for rel in CHAIN["section_inputs"]["linked_read_only"]:
        if (ROOT / rel).exists():
            link(REPO / rel, ROOT / rel)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    if not SNAPSHOT.exists():
        shutil.copy2(ROOT / CHAIN["target_manifest"], SNAPSHOT)
    rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_head=sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
               git_status_lines=len(sh(["git", "-C", str(ROOT), "status", "--porcelain"]).splitlines()),
               selected_snapshot=str(SNAPSHOT.relative_to(ROOT)), selected_snapshot_sha256=sha(SNAPSHOT),
               chain_sha256=sha(Path(__file__).resolve().parent / "chain.json"),
               linked=LINK_TOP + CHAIN["section_inputs"]["linked_read_only"])
    (WORK / "setup.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def moves():
    return json.loads((ROOT / "infra/storage/path-moves.json").read_text())["moves"]


def resolve(root: Path, rel: str) -> Path | None:
    """A manifest path under ROOT: on disk, else under its post-restructure name. None when neither exists."""
    p = root / rel
    if p.exists():
        return p
    for old, new in moves():
        if rel == old or rel.startswith(old + "/"):
            q = root / (new + rel[len(old):])
            return q if q.exists() else None
    return None


def dry_run():
    root = REPO if REPO.exists() else ROOT
    produced = set()
    bad = 0
    print(f"root: {root}")
    for i, s in enumerate(CHAIN["steps"], 1):
        print(f"{i:2d}. {s['id']}: {' '.join(s['command'])}")
        for rel in s["inputs"]:
            r = resolve(root, rel)
            state = "on disk" if r else ("written by an earlier step" if any(rel == o or rel.startswith(o + "/") for o in produced) else "MISSING")
            bad += state == "MISSING"
            print(f"      in  {rel}: {state}")
        for o in s["outputs"] + s["in_place_edits"]:
            produced.add(o)
            print(f"      out {o}")
    print(f"{bad} declared inputs missing")
    return bad


def other_jobs():
    out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
    jobs = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        # a real memguard process: python running memguard.py (waiting shells only mention it in their -c string)
        if len(argv) > 1 and re.search(r"python[0-9.]*$", argv[0]) and argv[1].endswith("memguard.py") and int(pid) != os.getpid():
            jobs.append(line.strip()[:160])
    return jobs


def load_runs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def run_step(sid: str) -> int:
    s = next(x for x in CHAIN["steps"] if x["id"] == sid)
    t_wait = time.time()
    while other_jobs():
        time.sleep(15)
    waited = time.time() - t_wait
    mg = CHAIN["memguard"]
    cmd = ["python3", "infra/tools/memguard.py", "--limit-gb", str(s.get("limit_gb", mg["limit_gb"])),
           "--timeout-s", str(s.get("timeout_s", mg["timeout_s"])), "--"] + s["command"]
    logs = WORK / "logs"; logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{sid}.log"
    marker = time.time()
    t0 = time.time()
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    secs = time.time() - t0
    text = log.read_text(errors="replace")
    peak = [float(x) for x in re.findall(r"memguard: peak ([0-9.]+) GB", text)]
    stopped = re.findall(r"memguard: (.*), stopping", text)
    # files in the linked (main-checkout) section inputs changed while the step ran (other sessions may also write there)
    touched = []
    for rel in CHAIN["section_inputs"]["linked_read_only"]:
        p = ROOT / rel
        for f in ([p] if p.is_file() else p.rglob("*") if p.is_dir() else []):
            try:
                if f.is_file() and f.stat().st_mtime >= marker:
                    touched.append(str(f.relative_to(ROOT)))
            except OSError:
                pass
    out = s.get("manifest_out")
    rec = dict(step=sid, command=" ".join(cmd), return_code=rc, seconds=round(secs, 1), waited_for_other_jobs_s=round(waited, 1),
               memguard_peak_gb=peak[-1] if peak else None, memguard_stop=stopped or None, log=str(log.relative_to(ROOT)),
               tail=text.splitlines()[-15:], main_checkout_linked_files_modified_during_step=touched,
               manifest_out_sha256=sha(REPO / out) if out and (REPO / out).exists() else None,
               finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    runs = load_runs(); runs[sid] = rec
    RUNS.write_text(json.dumps(runs, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "tail"}, indent=1))
    print("\n".join(rec["tail"]))
    return rc


def run_all(start: str | None):
    ids = [s["id"] for s in CHAIN["steps"]]
    for sid in ids[ids.index(start) if start else 0:]:
        rc = run_step(sid)
        if rc != 0:
            print(f"stopped at {sid} (return code {rc})")
            return rc
    return 0


# ---------------------------------------------------------------- compare

def content_sha(p: Path | None):
    if p is None:
        return None
    if p.suffix.lower() in (".step", ".stp"):
        b = p.read_bytes()
        b = re.sub(rb"HEADER;.*?ENDSEC;", b"", b, count=1, flags=re.S)
        return hashlib.sha256(b).hexdigest()
    if p.suffix == ".npz":
        import numpy as np
        h = hashlib.sha256()
        with np.load(p) as z:
            for k in sorted(z.files):
                a = z[k]; h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.hexdigest()
    if p.suffix.lower() == ".stl":
        b = p.read_bytes()
        return hashlib.sha256(b[80:] if not b[:5].lower() == b"solid" else b).hexdigest()   # binary STL: skip the 80-byte header
    return sha(p)


def file_cmp(rel_a, rel_b, root_a, root_b):
    if not rel_a and not rel_b:
        return "none"
    pa = resolve(root_a, rel_a) if rel_a else None
    pb = resolve(root_b, rel_b) if rel_b else None
    if pa is None or pb is None:
        return "missing_selected" if pa is None and pb is not None else "missing_rebuilt" if pb is None and pa is not None else "missing_both"
    if sha(pa) == sha(pb):
        return "raw_equal"
    return "content_equal" if content_sha(pa) == content_sha(pb) else "differs"


def bounds_diff(a, b):
    try:
        return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    except TypeError:
        return None


def kind(p):
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import inventory
    return inventory.kind_of(p)


def compare_manifests(path_a: Path, root_a: Path, path_b: Path, root_b: Path):
    """a = reference (main checkout / selected), b = rebuilt (copy)."""
    A = {p["name"]: p for p in json.loads(path_a.read_text())["parts"]}
    B = {p["name"]: p for p in json.loads(path_b.read_text())["parts"]}
    added, removed = sorted(set(B) - set(A)), sorted(set(A) - set(B))
    rows, counts = [], {}
    for n in sorted(set(A) & set(B)):
        a, b = A[n], B[n]
        r = dict(name=n, kind=kind(a), step=file_cmp(a.get("native_step"), b.get("native_step"), root_a, root_b),
                 mesh=file_cmp(a.get("mesh"), b.get("mesh"), root_a, root_b),
                 bounds_max_diff_mm=bounds_diff(a.get("bounds_mm"), b.get("bounds_mm")),
                 owning_link=None if a.get("owning_link") == b.get("owning_link") else [a.get("owning_link"), b.get("owning_link")],
                 role=None if a.get("role") == b.get("role") else [a.get("role"), b.get("role")],
                 path_changed=[k for k in ("native_step", "mesh") if a.get(k) != b.get(k)],
                 recorded_sha_equal={k: a.get(k) == b.get(k) for k in ("native_step_sha256", "mesh_sha256") if k in a or k in b})
        rows.append(r)
        c = counts.setdefault(r["kind"], Counter())
        c["parts"] += 1
        c[f"step_{r['step']}"] += 1
        c[f"mesh_{r['mesh']}"] += 1
        c["owning_link_changed"] += bool(r["owning_link"]); c["role_changed"] += bool(r["role"])
        bd = r["bounds_max_diff_mm"]
        if bd is not None:
            c["bounds_diff_over_0.01mm"] += bd > 0.01; c["bounds_diff_over_0.1mm"] += bd > 0.1; c["bounds_diff_over_1mm"] += bd > 1.0
    bds = [r["bounds_max_diff_mm"] for r in rows if r["bounds_max_diff_mm"] is not None]
    total = Counter()
    for c in counts.values():
        total.update(c)
    top_keys = {k: v for k, v in json.loads(path_a.read_text()).items() if k != "parts"}
    top_b = {k: v for k, v in json.loads(path_b.read_text()).items() if k != "parts"}
    return dict(reference=str(path_a), rebuilt=str(path_b), parts_reference=len(A), parts_rebuilt=len(B),
                added=added, removed=removed, counts_total=dict(total), counts_per_kind={k: dict(v) for k, v in counts.items()},
                bounds_max_diff_mm=max(bds) if bds else None,
                top_level_keys_differing=sorted(k for k in set(top_keys) | set(top_b) if top_keys.get(k) != top_b.get(k)),
                parts=rows)


def compare():
    runs = load_runs()
    out = dict(setup=json.loads((WORK / "setup.json").read_text()) if (WORK / "setup.json").exists() else None,
               steps=[], manifests={})
    for s in CHAIN["steps"]:
        r = runs.get(s["id"])
        out["steps"].append(dict(id=s["id"], ran=r is not None, return_code=r and r["return_code"], seconds=r and r["seconds"],
                                 memguard_peak_gb=r and r["memguard_peak_gb"], memguard_stop=r and r["memguard_stop"],
                                 linked_files_modified=r and r["main_checkout_linked_files_modified_during_step"]))
    seen = set()
    for s in CHAIN["steps"]:
        m = s.get("manifest_out")
        if not m or m in seen or not (REPO / m).exists():
            continue
        seen.add(m)
        ref = SNAPSHOT if m == CHAIN["target_manifest"] else ROOT / m
        if ref.exists():
            out["manifests"][m] = compare_manifests(ref, ROOT, REPO / m, REPO)
    (WORK / "compare.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# Body rebuild comparison", "", f"Setup: {out['setup']}", "", "## Steps", "",
         "| step | ran | rc | seconds | peak GB |", "|---|---|---|---|---|"]
    L += [f"| {x['id']} | {x['ran']} | {x['return_code']} | {x['seconds']} | {x['memguard_peak_gb']} |" for x in out["steps"]]
    for m, c in out["manifests"].items():
        L += ["", f"## {m}", "", f"reference {c['reference']} ({c['parts_reference']} parts) vs rebuilt ({c['parts_rebuilt']} parts)",
              f"added {len(c['added'])}: {c['added'][:40]}", f"removed {len(c['removed'])}: {c['removed'][:40]}",
              f"bounds max diff mm: {c['bounds_max_diff_mm']}", f"top-level keys differing: {c['top_level_keys_differing']}",
              "", "| kind | " + " | ".join(sorted(c["counts_total"])) + " |", "|---" * (len(c["counts_total"]) + 1) + "|"]
        for k, v in sorted(c["counts_per_kind"].items()):
            L.append(f"| {k} | " + " | ".join(str(v.get(x, 0)) for x in sorted(c["counts_total"])) + " |")
        L.append("| total | " + " | ".join(str(c["counts_total"][x]) for x in sorted(c["counts_total"])) + " |")
    (WORK / "compare.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", choices=["setup", "run", "compare"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step")
    ap.add_argument("--from", dest="start")
    a = ap.parse_args()
    if a.dry_run:
        sys.exit(1 if dry_run() else 0)
    if a.step:
        sys.exit(run_step(a.step))
    if a.action == "setup":
        setup()
    elif a.action == "run":
        sys.exit(run_all(a.start))
    elif a.action == "compare":
        compare()
    else:
        ap.print_help()
