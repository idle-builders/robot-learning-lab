"""Owner design feedback from the design dashboard: an annotated CAD snapshot and what was on screen.

The dashboard posts the annotated PNG, a JSON context (study, version, camera, joint angles, display
state, numbered marks resolved to parts and 3D points) and its Markdown summary. Each feedback is one
folder of local process records under .bookkeeping/design-feedback/<time>-<study>/:

    feedback.md   status line, the snapshot, the owner's note, marks, view, provenance
    snapshot.png  the annotated view
    context.json  the full machine-readable context

Claude and Codex read feedback.md (and the PNG it links). After acting, the agent edits its
"Status:" line to "addressed: <commit or reason>".

The remote dashboard cannot write here; there the owner copies the image and text to a chat.
"""
from __future__ import annotations

import base64
import json
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEEDBACK = ROOT / ".bookkeeping" / "design-feedback"
MAX_BODY = 24 * 1024 * 1024
PNG = b"\x89PNG\r\n\x1a\n"


class FeedbackError(ValueError):
    pass


def _study(db: sqlite3.Connection, result_id: str | None) -> dict:
    if not result_id:
        return {}
    row = db.execute("SELECT id, study_id, title, source_commit, source_path, robot_sections_json FROM results WHERE id=?",
                     (result_id,)).fetchone()
    return {} if row is None else {"result_id": row[0], "study_id": row[1], "title": row[2], "source_commit": row[3],
                                   "source_path": row[4], "robot_sections": json.loads(row[5] or "null")}


def _head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save(db: sqlite3.Connection, value: dict) -> dict:
    image = value.get("image_png")
    if not isinstance(image, str) or not image.startswith("data:image/png;base64,"):
        raise FeedbackError("image_png must be a PNG data URL")
    return _write(db, base64.b64decode(image.split(",", 1)[1], validate=True), value.get("context"), value.get("markdown"))


def _write(db: sqlite3.Connection, raw: bytes, context, markdown) -> dict:
    if not raw.startswith(PNG):
        raise FeedbackError("image_png is not a PNG")
    if not isinstance(context, dict) or not isinstance(markdown, str) or len(markdown) > 200_000:
        raise FeedbackError("context (object) and markdown (text) are required")
    study = context.get("study") if isinstance(context.get("study"), dict) else {}
    record = _study(db, study.get("result_id"))
    context["study"] = {**study, **record}
    context["repository_head"] = _head()
    now = datetime.now().astimezone()
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", str(record.get("study_id") or study.get("result_id") or "cad")).strip("-")[:60]
    folder = FEEDBACK / f"{now:%Y-%m-%dT%H%M%S}-{slug or 'cad'}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "snapshot.png").write_bytes(raw)
    (folder / "context.json").write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n")
    provenance = (f"Version: result `{record.get('result_id') or study.get('result_id') or '?'}`, sources "
                  f"`{record.get('source_path') or '?'}` at commit `{(record.get('source_commit') or '?')[:9]}`; "
                  f"repository HEAD `{context['repository_head'] or '?'}`.")
    lines = markdown.splitlines()
    head, body = (lines[0], lines[1:]) if lines and lines[0].startswith("# ") else ("# Design feedback", lines)
    text = "\n".join([head, "", "Status: open", f"Created: {now:%Y-%m-%d %H:%M %Z}", provenance, "",
                      "![Annotated snapshot](snapshot.png)", *body, "",
                      "Full context (camera vectors, every joint angle, per-mark samples): `context.json`.",
                      "When this is acted on, change the Status line to `addressed: <commit or reason>`.", ""])
    (folder / "feedback.md").write_text(text)
    rel = folder.relative_to(ROOT)
    return {"folder": str(rel), "prompt": (f"Read my design feedback in {rel}/feedback.md and look at {rel}/snapshot.png "
                                           "(numbered marks name the parts in context.json). Then act on it.")}


def listing(only_open: bool = False, limit: int = 20) -> list[dict]:
    items = []
    for folder in sorted(FEEDBACK.glob("*/feedback.md"), reverse=True):
        text = folder.read_text()
        status = next((line[7:].strip() for line in text.splitlines() if line.startswith("Status:")), "?")
        if only_open and status != "open":
            continue
        note = text.split("## Owner note", 1)[1].strip().split("\n## ", 1)[0].split("\nFull context", 1)[0].strip() if "## Owner note" in text else ""
        items.append({"path": str(folder.parent.relative_to(ROOT)), "status": status, "title": text.splitlines()[0][2:],
                      "note": note[:200]})
        if len(items) >= limit:
            break
    return items
