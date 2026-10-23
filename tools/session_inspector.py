"""Serve one live, local index of retained robot experiment artifacts.

Run from the repository root: python tools/session_inspector.py serve
The catalog scans only artifact roots; source snapshots and CAS objects are not copied.
"""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FEATURED = ROOT / "ui" / "session-inspector" / "featured.json"
CURRENT_DESIGN = ROOT / "robot" / "ape-current-design.json"
MEDIA = {".mp4": "videos", ".webm": "videos", ".glb": "models", ".png": "images", ".jpg": "images", ".jpeg": "images", ".webp": "images"}
DATA = {".json", ".jsonl", ".csv", ".md"}


def title_from_name(name):
    words = name.replace("_", "-").split("-")
    return " ".join(w.upper() if w in {"cad", "rl", "g1", "ppo", "vla"} else w.capitalize() for w in words)


def kind_from_name(name):
    if any(w in name for w in ("g1", "microduck", "toddlerbot", "berkeley", "arena")):
        return "Reference"
    if any(w in name for w in ("policy", "learning", "motion", "step", "walk", "training", "balance", "perception", "control", "skill", "task", "gym", "teacher", "imitation", "rollout", "grasp", "pick-place", "teach-find", "belief")):
        return "Learning"
    if any(w in name for w in ("ape", "joint", "frame", "knee", "shoulder", "hip", "ankle", "shell", "actuator", "cad", "swarm", "robot")):
        return "Engineering"
    return "Other"


def artifact_url(folder, name):
    return "/artifacts/" + quote(folder, safe="") + "/" + quote(name, safe="/")


def catalog():
    from lab.blob_store import INDEX
    import sqlite3
    cold = set()
    cold_by_parent = {}
    if INDEX.exists():
        with sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True) as db:
            cold = {row[0] for row in db.execute("SELECT path FROM files")}
        for name in cold:
            if name.startswith("artifacts/"):
                cold_by_parent.setdefault(str(Path(name).parent), set()).add(ROOT / name)

    def available(path):
        if path.is_file():
            return True
        candidate = path.resolve() if path.is_symlink() else path
        return candidate.relative_to(ROOT).as_posix() in cold if candidate.is_relative_to(ROOT) else False

    featured = json.loads(FEATURED.read_text()) if FEATURED.exists() else {}
    current = json.loads(CURRENT_DESIGN.read_text()) if CURRENT_DESIGN.exists() else {}
    current_root = current.get("evidence", {}).get("artifact_root", "").removeprefix("artifacts/")
    design_rejected = current.get("status", "").startswith("rejected")
    sessions = []
    if not ARTIFACTS.exists():
        return sessions
    for folder in ARTIFACTS.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            candidates = set(folder.iterdir()) | cold_by_parent.get(str(folder.relative_to(ROOT)), set())
            files = [p for p in candidates if available(p)]
            names = {p.name for p in files}
            # Existing evidence may have its landing page one directory down.
            landing = "index.html" if "index.html" in names else None
            if not landing:
                for child in sorted(folder.iterdir()):
                    if child.is_dir() and child.name not in {"source", "artifacts", "node_modules"} and available(child / "index.html"):
                        landing = child.name + "/index.html"
                        break
            groups = {"videos": [], "models": [], "images": [], "data": []}
            for path in sorted(files, key=lambda p: p.name):
                ext = path.suffix.lower()
                group = MEDIA.get(ext)
                if group:
                    groups[group].append({"name": path.name, "url": artifact_url(folder.name, path.name)})
                elif ext in DATA and path.name not in {"inspector.json"}:
                    groups["data"].append({"name": path.name, "url": artifact_url(folder.name, path.name)})
            if not any(groups.values()) and not landing:
                continue
            metadata = featured.get(folder.name, {}).copy()
            if "inspector.json" in names:
                try:
                    local = json.loads((folder / "inspector.json").read_text())
                    metadata.update({k: v for k, v in local.items() if k in {"title", "summary", "kind", "featured"} and isinstance(v, (str, bool))})
                except (OSError, ValueError):
                    pass
            preferred = sorted(groups["images"], key=lambda x: (not any(t in x["name"] for t in ("first-frame", "preview", "thumbnail", "front", "robot")), x["name"]))
            is_rejected = design_rejected and folder.name == current_root
            is_current = folder.name == current_root and not design_rejected
            if is_rejected:
                metadata.update(title=f"{current.get('name', folder.name)} — rejected historical design",
                                summary="Rejected in the morphology-first reset. No mechanical candidate is currently selected.",
                                featured=False)
            session = {
                "id": folder.name,
                "title": current.get("name") if is_current else metadata.get("title") or title_from_name(folder.name),
                "summary": metadata.get("summary", ""),
                "kind": metadata.get("kind") or kind_from_name(folder.name),
                "featured": bool(metadata.get("featured")) or is_current,
                "current": is_current,
                "updated": int(folder.stat().st_mtime),
                "landing": artifact_url(folder.name, landing) if landing else None,
                "thumbnail": preferred[0]["url"] if preferred else None,
                **groups,
            }
            sessions.append(session)
        except (OSError, PermissionError):
            continue
    sessions.sort(key=lambda s: (-s["updated"], s["id"]))
    return sessions


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        if self.path == "/" or urlsplit(self.path).path.startswith("/ui/session-inspector/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        if path in {"/", "/ui/session-inspector/index.html"}:
            target = "http://127.0.0.1:8805/"
            query = []
            if params.get("session"):
                query.append("session=" + quote(params["session"][0], safe=""))
            elif params.get("skill"):
                query.append("result=" + quote(params["skill"][0], safe=""))
            if params.get("clip"):
                query.append("clip=" + quote(params["clip"][0], safe=""))
            if query:
                target += "?" + "&".join(query)
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return
        # Retired local training videos remain addressable through the results hub.
        if path.startswith("/artifacts/") and path.lower().endswith(".mp4"):
            logical = unquote(path.lstrip("/"))
            if not (ROOT / logical).is_file():
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:8805/video?path=" + quote(logical, safe=""))
                self.end_headers()
                return
        if path == "/api/sessions":
            raw = json.dumps({"sessions": catalog()}, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/api/preview":
            params = parse_qs(parsed.query)
            session = params.get("session", [""])[0]
            name = params.get("file", [""])[0]
            if (session in {"", ".", ".."} or name in {"", ".", ".."}
                    or Path(session).name != session or Path(name).name != name
                    or Path(name).suffix.lower() not in DATA):
                self.send_error(400, "Invalid artifact name")
                return
            file = ARTIFACTS / session / name
            if not file.is_file() or not file.resolve().is_relative_to(ROOT):
                self.send_error(404)
                return
            with file.open("rb") as stream:
                content = stream.read(120_001)
            truncated = len(content) > 120_000
            content = content[:120_000]
            payload = {"text": content.decode("utf-8", errors="replace"), "truncated": truncated}
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("serve", "list"), default="serve")
    parser.add_argument("--port", type=int, default=8804)
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps({"sessions": catalog()}, indent=2))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Inspector: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
