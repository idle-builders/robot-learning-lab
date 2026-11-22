"""Remote read-only dashboard: a small index in the pod, the files in the bucket.

The local results hub stays the only place that writes. After each publish, upload
and commit on main this updates the web bucket (`ROBOTLAB_WEB_BUCKET` or `web_bucket` in
data/private-archive.json), which holds only what
the remote dashboard shows:

    site/                                       index (the hub's API answers) and UI;
                                                the pod copies this in every minute
    files/design/<study>/<version>/<file>       3D models, images and viewer data of
                                                the current design studies
    files/videos/<result>/<clip>.mp4            learning videos

The pod serves site/ itself and streams files/ from the bucket. Files are uploaded
once: a key under files/ names one study version or one clip, both immutable, so an
existing key is never sent again. Models go up as web copies (gltfpack: quantised,
meshopt-compressed, every triangle and named node kept). Videos are copied inside
object storage from the private archive. Keys nothing refers to any more are deleted.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

import results_hub as hub
from infra.storage.archive_config import client as archive_client

ROOT = hub.ROOT
WORK = ROOT / ".cache" / "web-export"
def _web_bucket():
    local = ROOT / "data" / "private-archive.json"
    config = json.loads(local.read_text()) if local.exists() else {}
    return os.environ.get("ROBOTLAB_WEB_BUCKET", config.get("web_bucket"))


WEB_BUCKET = _web_bucket()  # private, set locally; public clones have none
SITE, FILES = "site/", "files/"
GLTFPACK = ["npx", "-y", "gltfpack@1.0.0"]
GLTFPACK_ARGS = ["-cc", "-kn", "-km"]  # quantise + meshopt; keep named nodes (joint rig) and materials
PATH_PATTERN = re.compile(r"^(artifacts|studies|robot|control)/[^\s\"']+\.[A-Za-z0-9]+$")
mimetypes.add_type("model/gltf-binary", ".glb")


class QuietHandler(hub.Handler):
    def log_message(self, *args):
        pass


class Hub:
    """The results hub's own handler on a private port, so every answer is the real one."""

    def __enter__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()

    def get(self, path, **query):
        url = self.base + path + ("?" + urlencode(query) if query else "")
        try:
            with urllib.request.urlopen(url, timeout=600) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if 400 <= error.code < 500:  # no BOM, no reviews, retired result
                return None
            raise


def name_for(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", str(part)).strip("-") or "x"


def referenced_paths(value):
    """Source paths a viewer reads that are named inside a result's payload (parts, joints)."""
    if isinstance(value, str):
        if PATH_PATTERN.match(value):
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from referenced_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from referenced_paths(item)


def current(rows: list) -> list:
    """The latest version of each skill and each design study; standalone results as they are."""
    latest = {}
    for row in rows or []:
        group = (("skill", row["skill_id"]) if row.get("skill_id") is not None else
                 ("study", row["study_id"]) if row.get("study_id") else ("id", row["id"]))
        rank = (row.get("version_num") or 0, row.get("version_at") or "", row.get("updated_at") or "")
        if group not in latest or rank > latest[group][0]:
            latest[group] = (rank, row)
    keep = {row["id"] for _, row in latest.values()}
    return [row for row in rows or [] if row["id"] in keep]


class Index:
    """What the remote dashboard shows, and which bucket file each view opens."""

    def __init__(self, out: Path):
        self.out = out
        self.manifest = {"assets": {}, "clips": {}, "clip_nums": {}, "results": {}, "lists": {},
                         "aliases": {}, "version_ids": {}, "skills": {}, "skill_versions": {},
                         "design_boms": {}, "design_reviews": {}}
        self.files = {}  # bucket key -> ("design", path, sha) | ("clip", archive key)

    def write_json(self, folder: str, key: str, value) -> str:
        name = f"snapshot/{folder}/{name_for(key)}.json"
        target = self.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        return name

    def design_files(self, result: dict) -> list:
        """The study's 3D models and the data files its 3D viewer reads (part manifest, joints); nothing else.
        The viewer will not draw a model without its part manifest."""
        folder = f"{FILES}design/{safe(result['study_id'])}/{safe(result.get('study_version') or result['id'])}/"
        kept, taken = [], set()

        def add(path, sha):
            name = safe(Path(path).name)
            if name in taken:
                name = safe(Path(path).parent.name) + "-" + name
            taken.add(name)
            self.files[folder + name] = ("design", path, sha)
            self.manifest["assets"][path] = folder + name

        for asset in result.get("assets") or []:
            if asset.get("kind") == "model" and asset["path"].endswith(".glb"):
                add(asset["path"], asset.get("sha256"))
                kept.append(asset)
        payload = {k: v for k, v in result.items() if k not in {"assets", "clips", "versions"}}
        # A generated body view (robot/body/bom/publish_body.py) names every part's STEP / mesh and each generator's
        # manifest as text for people; the viewer opens none of them, so they are not design files to upload.
        payload["sections"] = [s for s in payload.get("sections") or [] if (s.get("payload") or {}).get("kind") != "body_view"]
        for path in sorted(set(referenced_paths(payload)) - {a["path"] for a in kept}):
            if path.endswith((".json", ".glb")):
                add(path, None)
        return kept

    def build(self):
        with Hub() as api:
            for dashboard in ("learning", "design"):
                rows = current(api.get("/api/results", dashboard=dashboard))
                if dashboard == "design":  # the selected study of each section, nothing else
                    chosen = set(api.get("/api/design-selection").values())
                    # studies the owner wants to review remotely before selecting them (robot/design.json "remote_review")
                    chosen |= set(json.loads((ROOT / "robot/design.json").read_text()).get("remote_review") or [])
                    rows = [row for row in rows if row.get("study_id") in chosen]
                self.manifest["lists"][dashboard] = self.write_json("list", dashboard, rows)
                for row in rows:
                    result = api.get("/api/result", id=row["id"])
                    if result is None:
                        continue
                    result["versions"] = []  # the remote view shows the current state only
                    # Design files and learning videos, nothing else.
                    result["assets"] = self.design_files(result) if dashboard == "design" else []
                    self.manifest["results"][row["id"]] = self.write_json("result", row["id"], result)
                    if dashboard == "design":
                        for endpoint, bucket in (("/api/design-bom", "design_boms"), ("/api/design-reviews", "design_reviews")):
                            value = api.get(endpoint, result=row["id"])
                            if value is not None:
                                self.manifest[bucket][row["id"]] = self.write_json(bucket, row["id"], value)
            for endpoint, key in (("/api/design", "design"), ("/api/design-selection", "design_selection"),
                                  ("/api/robot-bom", "robot_bom")):
                value = api.get(endpoint)
                self.manifest[key] = self.write_json("misc", key, value) if value is not None else None
        with hub.connect() as db:
            exported = set(self.manifest["results"])
            taken = set()  # one clip per current run: its primary clip, else its first
            for row in db.execute("""SELECT c.id, c.clip_num, c.bucket_key, c.result_id FROM clips c JOIN results r ON r.id = c.result_id
                                     WHERE c.verified_at IS NOT NULL AND c.bucket_key IS NOT NULL
                                     ORDER BY c.result_id, c.id IS NOT r.primary_clip_id, c.clip_num, c.id"""):
                if row["result_id"] not in exported or row["result_id"] in taken:
                    continue
                taken.add(row["result_id"])
                key = f"{FILES}videos/" + "/".join(safe(part) for part in row["id"].split(":", 1)) + ".mp4"
                self.files[key] = ("clip", row["bucket_key"])
                self.manifest["clips"][row["id"]] = {"id": row["id"], "clip_num": row["clip_num"], "url": "/" + key}
                if row["clip_num"] is not None:
                    self.manifest["clip_nums"][str(row["clip_num"])] = row["id"]
            for row in db.execute("SELECT id, version_id, skill_id, version_num, archived_at FROM results"):
                if row["id"] not in exported:
                    continue
                if row["version_id"] is not None:
                    self.manifest["version_ids"][str(row["version_id"])] = row["id"]
                if row["skill_id"] is not None and row["archived_at"] is None:
                    self.manifest["skill_versions"][f"{row['skill_id']}:{row['version_num']}"] = row["id"]
                    self.manifest["skill_versions"][f"{row['skill_id']}:{row['version_id']}"] = row["id"]
            for row in db.execute("""SELECT skill_id, id FROM results WHERE skill_id IS NOT NULL AND archived_at IS NULL
                                     ORDER BY version_num, version_id"""):
                if row["id"] in exported:
                    self.manifest["skills"][str(row["skill_id"])] = row["id"]
            for old, new in db.execute("SELECT old_id, new_id FROM result_aliases"):
                if new in exported:
                    self.manifest["aliases"][old] = new

    def site(self):
        ui = hub.UI
        for item in ui.iterdir():
            if item.is_file() and item.suffix != ".html":
                shutil.copyfile(item, self.out / item.name)
        viewer = self.out / "ui" / "session-inspector"
        viewer.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "infra/dashboard/session-inspector-ui/viewer3d.js", viewer / "viewer3d.js")
        pages = {"index.html": ["", "policies", "perception", "planning"], "design.html": ["design"]}
        for source, folders in pages.items():
            html = (ui / source).read_text()
            html = re.sub(r'<script type="module" src="/app\.js[^"]*"></script>',
                          '<link rel="stylesheet" href="/static.css"><script type="module" src="/static.js"></script>', html)
            assert "/static.js" in html, f"{source}: app.js entry point not found"
            for folder in folders:
                target = self.out / folder / "index.html"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(html)
        (self.out / "snapshot" / "missing.html").write_text(MISSING_PAGE)

    def write_manifest(self):
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.manifest.update(exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), source_commit=commit)
        (self.out / "snapshot" / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, separators=(",", ":")))


MISSING_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Not on the remote view</title><style>body{font:15px system-ui;margin:40px auto;max-width:640px;padding:0 16px;color:#222;background:#fff}
code{word-break:break-all}@media(prefers-color-scheme:dark){body{background:#151515;color:#ddd}}</style>
<h1>Only on the lab computer</h1><p>The remote view carries design models, images and learning videos.
Open this file from the local dashboard.</p><p><code id="p"></code></p>
<script>document.getElementById('p').textContent=new URLSearchParams(location.search).get('path')||''</script>"""


# --- files/ in the bucket ------------------------------------------------------------------
def model_cache_key(sha: str) -> str:
    return f"glb:{sha}:{' '.join(GLTFPACK + GLTFPACK_ARGS)}"  # shared with earlier exports


def cached_model(sha: str | None) -> Path | None:
    index_path = WORK / "cache-index.json"
    if not sha or not index_path.exists():
        return None
    entry = json.loads(index_path.read_text()).get(model_cache_key(sha))
    return WORK / "cache" / entry["file"] if entry and (WORK / "cache" / entry["file"]).exists() else None


def web_model(source: Path, sha: str) -> Path:
    """gltfpack web copy, cached by source content."""
    if ready := cached_model(sha):
        return ready
    cache, index_path = WORK / "cache", WORK / "cache-index.json"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name_for(model_cache_key(sha))}.glb"
    result = subprocess.run([*GLTFPACK, "-i", str(source), "-o", str(target), *GLTFPACK_ARGS], capture_output=True, text=True)
    if result.returncode or not target.exists():
        raise RuntimeError(f"gltfpack failed: {result.stderr.strip()[-200:]}")
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[model_cache_key(sha)] = {"file": target.name, "suffix": ".glb"}
    index_path.write_text(json.dumps(index, indent=0))
    return target


def put_design_file(s3, bucket: str, key: str, path: str, sha: str | None):
    kind = mimetypes.guess_type(key)[0] or "application/octet-stream"
    if key.endswith(".glb") and (ready := cached_model(sha)):
        return s3.upload_file(str(ready), bucket, key, ExtraArgs={"ContentType": kind})
    with hub.opened_source_asset(path) as source:
        if key.endswith(".glb"):
            source = web_model(source, sha or hub.digest_file(source))
        s3.upload_file(str(source), bucket, key, ExtraArgs={"ContentType": kind})


def web_client():
    """This storage now and then leaves a request hanging; give up fast and retry."""
    return archive_client(read_timeout=20)


def sync_files(index: Index, bucket: str) -> dict:
    """Upload what is missing under files/, then delete what nothing refers to."""
    s3 = web_client()
    present = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=FILES):
        present.update(item["Key"] for item in page.get("Contents", []))
    sent, failed = 0, []
    for key, source in sorted(index.files.items()):
        if key in present:
            continue
        try:
            if source[0] == "clip":
                s3.copy_object(Bucket=bucket, Key=key, CopySource={"Bucket": hub.BUCKET, "Key": source[1]},
                               ContentType="video/mp4", MetadataDirective="REPLACE")
            else:
                put_design_file(s3, bucket, key, source[1], source[2])
            sent += 1
            print(f"uploaded {key}", flush=True)
        except Exception as error:  # an unavailable file is reported, never fatal
            failed.append({"key": key, "reason": f"{type(error).__name__}: {error}"[:300]})
    missing = {f["key"] for f in failed}
    index.manifest["assets"] = {p: k for p, k in index.manifest["assets"].items() if k not in missing}
    index.manifest["clips"] = {i: c for i, c in index.manifest["clips"].items() if c["url"][1:] not in missing}
    return {"files": len(index.files), "uploaded": sent, "failed": failed, "stale": sorted(present - set(index.files))}


def mirror_site(out: Path, bucket: str) -> dict:
    """Mirror the index and UI into site/; the manifest goes last, stale keys after it."""
    s3 = web_client()
    remote = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=SITE):
        for item in page.get("Contents", []):
            remote[item["Key"]] = item["ETag"].strip('"')
    local = {SITE + p.relative_to(out).as_posix(): p for p in out.rglob("*") if p.is_file()}
    manifest = SITE + "snapshot/manifest.json"
    sent = 0
    for key in sorted(local, key=lambda k: (k == manifest, k)):
        path = local[key]
        if remote.get(key) == hashlib.md5(path.read_bytes()).hexdigest():
            continue
        s3.upload_file(str(path), bucket, key, ExtraArgs={
            "ContentType": mimetypes.guess_type(path.name)[0] or "application/octet-stream"})
        sent += 1
    return {"uploaded": sent, "stale": sorted(set(remote) - set(local))}


def delete_keys(bucket: str, keys: list) -> int:
    """Last, once the new index is live. This storage refuses batch deletes without Content-MD5."""
    s3 = web_client()
    with ThreadPoolExecutor(8) as pool:
        return sum(pool.map(lambda key: s3.delete_object(Bucket=bucket, Key=key) and 1, keys))


def publish(out: Path = WORK / "site", bucket: str = WEB_BUCKET, upload: bool = True) -> dict:
    if upload and not bucket:
        raise RuntimeError("Set ROBOTLAB_WEB_BUCKET or web_bucket in data/private-archive.json to upload")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    index = Index(out)
    index.build()
    index.site()
    report = {"results": len(index.manifest["results"]), "files": len(index.files)}
    if upload:
        report["files_sync"] = sync_files(index, bucket)  # files first: the index only names what exists
    index.write_manifest()
    if upload:
        report["site_sync"] = mirror_site(out, bucket)
        stale = report["files_sync"].pop("stale") + report["site_sync"].pop("stale")
        report["deleted"] = delete_keys(bucket, stale)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "last-export-report.json").write_text(json.dumps(report, indent=1))
    return report


class SiteLock:
    """One export at a time writes the site and the bucket."""

    def __init__(self, blocking: bool):
        self.blocking = blocking

    def __enter__(self):
        import fcntl
        WORK.mkdir(parents=True, exist_ok=True)
        self.handle = open(WORK / "export.lock", "w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            self.handle.close()
            return False
        return True

    def __exit__(self, *exc):
        if not self.handle.closed:
            self.handle.close()  # closing releases the lock


def refresh():
    """Background refresh after a publish, an upload or a commit on main (results_hub and
    .githooks/post-commit start it).

    Requests arriving while a refresh runs leave a pending mark; the running refresh
    repeats once for all of them, so a burst of publishes costs one or two exports."""
    pending = WORK / "refresh.pending"
    WORK.mkdir(parents=True, exist_ok=True)
    pending.touch()
    while pending.exists():
        with SiteLock(blocking=False) as held:
            if not held:
                return  # the running refresh sees the mark
            while pending.exists():
                pending.unlink()
                started = datetime.now(timezone.utc).isoformat(timespec="seconds")
                report = publish()
                files = report["files_sync"]
                print(started, json.dumps({"results": report["results"], "files": files["files"], "uploaded": files["uploaded"],
                                           "deleted": report.get("deleted", 0), "failed": len(files["failed"])}), flush=True)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upload", action="store_true", help="Update the bucket (without it: build the index locally only)")
    parser.add_argument("--out", type=Path, default=WORK / "site")
    parser.add_argument("--bucket", default=WEB_BUCKET)
    parser.add_argument("--refresh", action="store_true", help="Coalesced update (run after publish, upload and commits on main)")
    args = parser.parse_args(argv)
    if args.refresh:
        return refresh()
    with SiteLock(blocking=True):
        report = publish(args.out, args.bucket, upload=args.upload)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
