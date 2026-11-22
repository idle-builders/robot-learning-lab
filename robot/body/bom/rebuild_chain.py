"""Recreate the selected body manifest from scripts and inputs in an isolated copy, and compare it with the selected one.

The ordered builder steps are derived from the body definitions (robot/**/<builder>.bom.yaml: each generator's command,
inputs, outputs, `after` and resources; robot/body/bom/definitions.py), followed by the assembly check, the BOM, the bom
check of the target (robot/design.json body_manifest) and the design-dashboard record written to a file
(robot/body/bom/publish_body.py; the main checkout publishes it with `publish_body.py --publish`). This script

  setup [--commit REV]  makes WORK/repo (code from REV via git archive when given, else the working tree): a copy of the working tree's code (rsync, without artifacts/, .cache/, .venv*, data/, .git,
             .tools, ...), with .venv, .tools, data and .artifact-store linked from the main checkout, and an artifacts/
             tree in which every top-level folder and every artifacts/robot/* folder except body is a link to the main
             checkout's, the outputs of the section builders (generators with rebuild false) are links (read-only
             inputs: outputs of the definitions' generators with rebuild false), and the folders the chain writes start empty. Scripts derive ROOT from
             __file__, so every output lands inside the copy.
  --dry-run  lists the steps and checks that every declared input resolves (on disk, through infra/storage/path-moves.json,
             or as a file an earlier step writes) without running anything.
  --step ID  runs one step in the copy under infra/tools/memguard.py (memguard waits for a slot in the heavy-job budget itself).
  run        runs every step from --from ID (default the first) in order and stops at the first failure.
  compare    compares the copy's manifests with the main checkout's live ones (the final one also records whether the
             selected manifest changed since the snapshot taken at setup): part names added / removed, native STEP and mesh sha256 (raw and content: STEP
             without its HEADER, npz by array bytes; same_file when both resolve to one file through links or path moves), reference
             files that no longer hash to their recorded sha256 (input changed since the reference was built), bounds difference, owning link / role changes, counts per kind.
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
MEMGUARD = dict(limit_gb=6, timeout_s=900)          # per-step default; a generator's `resources` override it


def derived_chain():
    """The rebuild as data, derived from the body definitions (robot/**/<builder>.bom.yaml, robot/body/bom/definitions.py):
    generators with rebuild true in the order their declared inputs and outputs give; read-only section folders = outputs of
    generators with rebuild false; produced folders = the shortest folder of each rebuild output under artifacts/robot/body
    that neither holds nor lies in a read-only output; writable copies = their `writes_into`. Then the BOM tool's own steps on
    the target (robot/design.json body_manifest): the assembly check, the BOM, the bom check, and the dashboard record
    (robot/body/bom/publish_body.py --out; publishing happens from the main checkout with --publish)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import definitions as DEF
    D = DEF.load()
    order = D.generator_order()
    ro = [o for g in D.generators if not g.get("rebuild", True) and not g.get("presentation") for o in g.get("outputs", []) if o.startswith("artifacts/")]
    produced, copied, steps = [], [], []
    for gid in order:
        g = D.generator(gid)
        steps.append(dict(id=gid, command=g["command"], inputs=g.get("inputs", []), outputs=g.get("outputs", []), in_place_edits=g.get("in_place_edits", []),
                          manifest_out=g.get("manifest_out"), file=g["file"], **(g.get("resources") or {})))
        for o in g.get("outputs", []):
            parts = o.split("/")
            for k in range(4, len(parts) + 1):
                pre = "/".join(parts[:k])
                if not pre.startswith("artifacts/robot/body/"):
                    continue
                if any(r == pre or r.startswith(pre + "/") or pre.startswith(r + "/") for r in ro):
                    if not any(pre.startswith(r + "/") for r in ro):
                        continue
                    if any(r.startswith(pre + "/") or r == pre for r in ro):
                        continue
                if pre not in produced:
                    produced.append(pre)
                break
        copied += [w for w in g.get("writes_into", []) if w not in copied]
    target = json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]
    M = ["--manifest", target]
    steps += [dict(id="assembly-check", command=[".venv/bin/python", "-B", "robot/body/checks/run.py"] + M + ["--checks", "assembly", "--label", "assembly"],
                   inputs=[target], outputs=[".cache/work/cad-speed/checks/assembly/report.json"], in_place_edits=[], manifest_out=None, timeout_s=1800, file="(BOM tool)"),
              dict(id="bom", command=[".venv/bin/python", "-B", "robot/body/bom/build.py"] + M, inputs=[target], outputs=[".cache/work/bom"], in_place_edits=[],
                   manifest_out=None, limit_gb=4, timeout_s=600, file="(BOM tool)"),
              dict(id="bom-check", command=[".venv/bin/python", "-B", "robot/body/checks/run.py"] + M + ["--checks", "bom", "--label", "bom"],
                   inputs=[target], outputs=[".cache/work/cad-speed/checks/bom/report.json"], in_place_edits=[], manifest_out=None, limit_gb=4, timeout_s=900, file="(BOM tool)"),
              # the dashboard record of the selected body from the same definitions and these outputs; in the copy it is only
              # written to a file (publish_body.py refuses --publish outside the main checkout): publish from main with --publish
              dict(id="dashboard-record", command=[".venv/bin/python", "-B", "robot/body/bom/publish_body.py", "--manifest", target, "--out", ".cache/work/bom/dashboard-record.json"],
                   inputs=[target], outputs=[".cache/work/bom/dashboard-record.json"], in_place_edits=[], manifest_out=None, limit_gb=4, timeout_s=600, file="(BOM tool)")]
    return dict(target_manifest=target, memguard=MEMGUARD, steps=steps, findings=D.findings,
                section_inputs=dict(linked_read_only=ro, produced_by_chain=produced, copied_writable=copied))


CHAIN = derived_chain()
WORK = Path(os.environ.get("BODY_REBUILD_WORK", ROOT / ".cache/work/body-rebuild"))
REPO = WORK / "repo"
RUNS = WORK / "runs.json"
SNAPSHOT = WORK / "selected" / Path(CHAIN["target_manifest"]).name

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


def own_paths() -> tuple[list[str], list[str]]:
    """(produced, copied): folders the chain writes in the copy. produced start empty; copied start as a copy of the
    main checkout's folder (a hook rewrites some files of a section folder that also holds section inputs)."""
    si = CHAIN["section_inputs"]
    return list(si["produced_by_chain"]), list(si.get("copied_writable", []))


def link_except(rel: str, produced: list[str], copied: list[str] = ()):
    """Link ROOT/rel into the copy; when a chain output lies inside it (a produced folder inside a section folder), make rel a real
    folder and link its children one by one: a produced path starts empty, a copied path as a copy, so no step can
    write through a link into the main checkout."""
    mine = [q for q in list(produced) + list(copied) if q.startswith(rel + "/")]
    if not mine:
        link(REPO / rel, ROOT / rel)
        return
    if (REPO / rel).is_symlink():
        (REPO / rel).unlink()  # an older setup linked the whole folder
    (REPO / rel).mkdir(parents=True, exist_ok=True)
    for c in sorted((ROOT / rel).iterdir()):
        sub = f"{rel}/{c.name}"
        if (REPO / sub).is_symlink() and (sub in mine or any(q.startswith(sub + "/") for q in mine)):
            (REPO / sub).unlink()
        if sub in produced:
            (REPO / sub).mkdir(parents=True, exist_ok=True)
        elif sub in copied:
            if not (REPO / sub).exists():
                shutil.copytree(c, REPO / sub, symlinks=True)
        elif any(q.startswith(sub + "/") for q in mine):
            link_except(sub, produced, copied)
        else:
            link(REPO / sub, c)
    for q in mine:  # produced folders that do not exist in the main checkout yet
        if q in produced and q.count("/") == rel.count("/") + 1:
            (REPO / q).mkdir(parents=True, exist_ok=True)


def setup(commit=None):
    WORK.mkdir(parents=True, exist_ok=True)
    REPO.mkdir(parents=True, exist_ok=True)
    src = ROOT
    if commit:      # the copy's code and data from a git commit, not from the working tree (other sessions' edits stay out)
        src = WORK / "commit-export"
        shutil.rmtree(src, ignore_errors=True); src.mkdir(parents=True)
        arch = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", commit], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(src)], input=arch, check=True)
    cmd = ["rsync", "-a", "--delete"] + [f"--exclude={e}" for e in RSYNC_EXCLUDE] + [f"{src}/", f"{REPO}/"]
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
    produced, copied = own_paths()
    for rel in CHAIN["section_inputs"]["linked_read_only"]:
        if (ROOT / rel).exists():
            link_except(rel, produced, copied)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    if not SNAPSHOT.exists():
        shutil.copy2(ROOT / CHAIN["target_manifest"], SNAPSHOT)
    rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code_from=commit or "working tree", git_head=sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
               git_status_lines=len(sh(["git", "-C", str(ROOT), "status", "--porcelain"]).splitlines()),
               selected_snapshot=str(SNAPSHOT.relative_to(ROOT)), selected_snapshot_sha256=sha(SNAPSHOT),
               definitions_sha256={str(f.relative_to(ROOT)): sha(f) for f in __import__("definitions").discover()},
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
    for f in CHAIN["findings"]:
        print(f"definition finding: {f['kind']} {f['file']} {f['entry']}: {f['what']}")
    print("read-only section outputs:", CHAIN["section_inputs"]["linked_read_only"])
    print("produced in the copy:", CHAIN["section_inputs"]["produced_by_chain"])
    print("copied writable:", CHAIN["section_inputs"]["copied_writable"])
    produced = set()
    bad = 0
    print(f"root: {root}")
    for i, s in enumerate(CHAIN["steps"], 1):
        print(f"{i:2d}. {s['id']} ({s.get('file')}): {' '.join(s['command'])}")
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


MAX_JOBS, MAX_SUM_GB = 4, 24.0     # docs/operations/working-rules.md (owner 2026-11-20): at most 4 heavy jobs, caps summing to 24 GB


def other_jobs():
    """[(command, limit GB)] of the running memguard jobs: a python process whose first argument is memguard.py
    (waiting shells only mention it inside their -c string, so they do not count)."""
    out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
    jobs = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        if len(argv) > 1 and re.search(r"python[0-9.]*$", argv[0], re.I) and argv[1].endswith("memguard.py") and int(pid) != os.getpid():
            m = re.search(r"--limit-gb\s+([0-9.]+)", cmd)
            jobs.append((line.strip()[:160], float(m.group(1)) if m else 6.0))
    return jobs


def budget_full(limit_gb):
    jobs = other_jobs()
    return len(jobs) >= MAX_JOBS or sum(g for _, g in jobs) + limit_gb > MAX_SUM_GB


def load_runs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def run_step(sid: str) -> int:
    s = next(x for x in CHAIN["steps"] if x["id"] == sid)
    # memguard itself waits for a slot in the shared budget (registry ~/.cache/memguard, real jobs only); a wait loop here
    # counted every waiting memguard wrapper as a running job and never started under load (2026-11-22)
    waited = 0.0
    mg = CHAIN["memguard"]
    cmd = ["python3", "infra/tools/memguard.py", "--limit-gb", str(s.get("limit_gb", mg["limit_gb"])),
           "--timeout-s", str(s.get("timeout_s", mg["timeout_s"])), "--"] + s["command"]
    if shutil.which("caffeinate"):
        # macOS: keep the machine awake while the step runs. memguard's timeout is wall time, so a step that ran while
        # the laptop slept was stopped with almost no CPU time (2026-11-20, roll-flip-cad: 997 s wall, asleep 18:06-18:21).
        cmd = ["caffeinate", "-i", "-s"] + cmd
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
    own = [ROOT / q for q in sum(own_paths(), [])]  # the copy has its own folders there; main's change by other sessions
    for rel in CHAIN["section_inputs"]["linked_read_only"]:
        p = ROOT / rel
        for f in ([p] if p.is_file() else p.rglob("*") if p.is_dir() else []):
            try:
                if any(f.is_relative_to(o) for o in own):
                    continue
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
    if pa.resolve() == pb.resolve():
        return "same_file"  # both resolve (path-moves, links into the main checkout) to one file: not a reproduction
    if sha(pa) == sha(pb):
        return "raw_equal"
    return "content_equal" if content_sha(pa) == content_sha(pb) else "differs"


def ref_matches(part, key, root):
    """Does the reference's file still hash to the sha256 its manifest recorded? False = the input changed since the
    reference was built (e.g. a section rebuilt later), so equality with it says nothing about that reference."""
    p = resolve(root, part[key])
    return None if p is None else sha(p) == part[f"{key}_sha256"]


def bounds_diff(a, b):
    try:
        return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    except TypeError:
        return None


def kind(p):
    return p.get("role") or "unknown"

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
                 recorded_sha_equal={k: a.get(k) == b.get(k) for k in ("native_step_sha256", "mesh_sha256") if k in a or k in b},
                 reference_file_matches_recorded_sha={k: ref_matches(a, k, root_a) for k in ("native_step", "mesh") if a.get(k) and a.get(f"{k}_sha256")})
        rows.append(r)
        c = counts.setdefault(r["kind"], Counter())
        c["parts"] += 1
        c[f"step_{r['step']}"] += 1
        c[f"mesh_{r['mesh']}"] += 1
        c["owning_link_changed"] += bool(r["owning_link"]); c["role_changed"] += bool(r["role"])
        c["reference_file_changed_since_recorded"] += sum(v is False for v in r["reference_file_matches_recorded_sha"].values())
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
        # against the main checkout's live file: the chain never writes there (setup makes every chain output a folder
        # of the copy), and the selected body moves while a run takes an hour (2026-11-20: the setup snapshot of the
        # selected body was 2 h older than main's and showed 59 stale differences). The snapshot's sha stays recorded.
        ref = ROOT / m
        if ref.exists():
            out["manifests"][m] = compare_manifests(ref, ROOT, REPO / m, REPO)
            if m == CHAIN["target_manifest"] and SNAPSHOT.exists():
                out["manifests"][m]["reference_sha256"] = sha(ref)
                out["manifests"][m]["setup_snapshot_sha256"] = sha(SNAPSHOT)
                out["manifests"][m]["selected_changed_since_setup"] = sha(ref) != sha(SNAPSHOT)
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
    ap.add_argument("--commit", help="setup: take the copy's code and definitions from this git commit (e.g. HEAD), not the working tree")
    a = ap.parse_args()
    if a.dry_run:
        sys.exit(1 if dry_run() else 0)
    if a.step:
        sys.exit(run_step(a.step))
    if a.action == "setup":
        setup(a.commit)
    elif a.action == "run":
        sys.exit(run_all(a.start))
    elif a.action == "compare":
        compare()
    else:
        ap.print_help()
