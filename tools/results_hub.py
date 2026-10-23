"""Table-backed results viewer and verified video publisher.

Usage: python tools/results_hub.py publish RUN.json|upload|serve|design-template
The database is the presentation index. Original native reports remain scientific evidence.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from pathlib import Path
import sqlite3
import subprocess
import sys
import mimetypes
from urllib.parse import parse_qs, urlsplit
from session_inspector import catalog
import design_bom
from lab.archive_config import settings as archive_settings, client as archive_client

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.sqlite"
UI = ROOT / "ui" / "results-hub"
CATALOG = ROOT / "data" / "legacy-results-source" / "skill-results.json.gz"
QUALITATIVE = ROOT / "data" / "legacy-results-source" / "qualitative-assessments.json.gz"
BUCKET = archive_settings()["bucket"]
ENDPOINT = archive_settings()["endpoint"]
PREFIX = "robot-learning-lab/results/videos/sha256"
REVIEW_JUDGMENTS = {"unreviewed", "shortlist", "revise", "reject"}
MAX_REVIEW_BODY = 128 * 1024


def design_retirements():
    """Explicit user-directed cleanup, separate from engineering assessment."""
    path = ROOT / "data" / "design-study-retirements.json"
    return json.loads(path.read_text()) if path.exists() else {}


def dashboard_for(row):
    """Explicit routing wins; CAD catalogs and mechanical work default to design."""
    if row.get("dashboard") in {"learning", "design"}:
        return row["dashboard"]
    if row.get("workspace", "").lower() in {"body", "design", "engineering", "electronics"}:
        return "design"
    if any(isinstance(s.get("payload"), dict) and s["payload"].get("kind") == "design_catalog"
           for s in row.get("sections", [])):
        return "design"
    return "learning"


def design_sections(text):
    """Navigation labels, not design selection or engineering assessments."""
    patterns = {
        "Hip": r"hip", "Knee": r"knee", "Ankle / foot": r"ankle|foot|sole",
        "Shoulder": r"shoulder", "Elbow": r"elbow", "Neck / head": r"neck|head",
        "Hand / wrist": r"hand|wrist|grip", "Torso / shell": r"torso|shell|frame|waist|pelvis|hull",
        "Upper arm": r"upper[- ]arm", "Forearm": r"forearm",
        "Thigh": r"thigh", "Shin / lower leg": r"shin|lower[- ]leg",
        "Actuators": r"actuator|motor|transmission|drive|gear",
        "Electronics": r"electronic|battery|power|wiring|sensor|compute|controller|harness|heat|audio|vision|sensing",
    }
    labels = [label for label, pattern in patterns.items()
              if re.search(r"\b(?:" + pattern + r")\b", text, re.I)]
    if len(labels) > 1 and not re.search(r"torso|frame|waist|pelvis|hull", text, re.I):
        labels = [label for label in labels if label != "Torso / shell"]
    return labels or ["Whole robot"]


def annotate_design_groups(row):
    labels = []
    explicit = row.get("robot_sections")
    fallback = explicit or design_sections(row.get("area", "") + " " + row["title"])
    for section in row.get("sections", []):
        payload = section.get("payload")
        if not isinstance(payload, dict) or payload.get("kind") != "design_catalog":
            continue
        for group in payload.get("groups", []):
            group_labels = group.get("robot_sections") or explicit or design_sections(group.get("id", "") + " " + group.get("title", ""))
            group["design_sections"] = fallback if group_labels == ["Whole robot"] else group_labels
            labels.extend(group["design_sections"])
    row["design_sections"] = list(dict.fromkeys(explicit or labels or fallback))
    return row


def validate_design_registration(row):
    """Check registration links, never judge an engineering result or its metrics."""
    assets = {asset["path"] for asset in row.get("assets", [])}
    clip_ids = {clip["id"] for clip in row.get("clips", [])}

    def sections(value):
        if not isinstance(value, list) or not value or any(not isinstance(s, str) or not s.strip() for s in value):
            raise ValueError("robot_sections must be a nonempty list of part names, e.g. ['Knee']")

    def linked(path):
        if path and path not in assets:
            raise ValueError(f"Register this CAD/evidence path in the study's complete assets list: {path}")

    def evidence(items):
        if not isinstance(items, list):
            raise ValueError("design evidence must be a list")
        for item in items:
            if not isinstance(item, dict) or not item.get("id") or not item.get("title"):
                raise ValueError("each design evidence item needs a stable id and title")
            if item.get("kind") not in {"measurement", "simulation", "analysis", "physical_trial"}:
                raise ValueError("evidence kind must be measurement, simulation, analysis or physical_trial")
            for measurement in item.get("measurements", []):
                if not all(key in measurement for key in ("name", "value", "unit")):
                    raise ValueError("each measurement needs name, value and native unit")
            for ref in item.get("source_refs", []):
                linked(ref.get("path"))
            for clip_id in item.get("clip_ids", []):
                if clip_id not in clip_ids:
                    raise ValueError(f"Register evidence recording in this study's clips list: {clip_id}")

    if row.get("robot_sections") is not None:
        sections(row["robot_sections"])
    for section in row.get("sections", []):
        payload = section.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") == "design_evidence":
            evidence(payload.get("items", []))
        if payload.get("kind") != "design_catalog":
            continue
        for group in payload.get("groups", []):
            if group.get("robot_sections") is not None:
                sections(group["robot_sections"])
            for option in group.get("options", []):
                if "evidence" in option:
                    evidence(option["evidence"])
                # Explicit registrations use the current complete asset contract.
                # Retained historical catalogs remain readable without republishing.
                if row.get("robot_sections") is None:
                    continue
                cad = option.get("cad", {})
                for key in ("parts_path", "step_path", "report_path", "motion_contract_path"):
                    linked(cad.get(key))
                for view in cad.get("views", []):
                    linked(view.get("model_path"))
                linked(option.get("image_path"))


def archive_dashboard(row):
    name = row["id"]
    if re.search(r"policy|training|learning|rollout|perception|planning|walk|gait|imitation|teacher|balance|control", name, re.I):
        return "learning"
    if re.search(r"engineering|cad|actuator|morphology|mechanical|shell|skin|rated-leg|torso|deltoid", name, re.I):
        return "design"
    if row["kind"] == "Engineering":
        return "design"
    if row["kind"] in {"Learning", "Reference", "Perception and planning"}:
        return "learning"
    # Old unlabelled morphology/package studies belong with mechanical evidence.
    return "design" if re.search(
        r"morphology|torso|deltoid|shell|skin|rated-leg|compact-relocated|architecture|serial-scale|engineering|cad|actuator",
        row["id"], re.I) else "learning"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS results (
      id TEXT PRIMARY KEY, title TEXT NOT NULL, area TEXT NOT NULL DEFAULT '',
      status TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
      observation TEXT NOT NULL DEFAULT '', measurements TEXT NOT NULL DEFAULT '',
      limitations TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '',
      provenance TEXT NOT NULL DEFAULT '', matrix_json TEXT,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS clips (
      id TEXT PRIMARY KEY, result_id TEXT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
      title TEXT NOT NULL, role TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '',
      observation TEXT NOT NULL DEFAULT '', source_path TEXT NOT NULL,
      report_path TEXT, trace_path TEXT, sha256 TEXT NOT NULL,
      bytes INTEGER NOT NULL, bucket_key TEXT, verified_at TEXT
    );
    CREATE INDEX IF NOT EXISTS clips_result_idx ON clips(result_id);
    CREATE TABLE IF NOT EXISTS sections (
      result_id TEXT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
      position INTEGER NOT NULL, title TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
      payload_json TEXT NOT NULL, PRIMARY KEY(result_id,position)
    );
    CREATE TABLE IF NOT EXISTS source_payload (
      result_id TEXT PRIMARY KEY REFERENCES results(id) ON DELETE CASCADE,
      payload_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS assets (
      result_id TEXT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
      id TEXT NOT NULL, title TEXT NOT NULL, kind TEXT NOT NULL,
      path TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
      PRIMARY KEY(result_id,id)
    );
    CREATE TABLE IF NOT EXISTS design_reviews (
      result_id TEXT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
      design_id TEXT NOT NULL, judgment TEXT NOT NULL
        CHECK(judgment IN ('unreviewed','shortlist','revise','reject')),
      comment TEXT NOT NULL CHECK(length(comment)<=10000),
      version INTEGER NOT NULL CHECK(version>0), updated_at TEXT NOT NULL,
      PRIMARY KEY(result_id,design_id)
    );
    CREATE TABLE IF NOT EXISTS design_review_history (
      result_id TEXT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
      design_id TEXT NOT NULL, version INTEGER NOT NULL,
      judgment TEXT NOT NULL, comment TEXT NOT NULL, updated_at TEXT NOT NULL,
      PRIMARY KEY(result_id,design_id,version)
    );
    CREATE TABLE IF NOT EXISTS design_preferences (
      section TEXT PRIMARY KEY, study_id TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS design_study_clips (
      result_id TEXT NOT NULL REFERENCES results(id), clip_id TEXT NOT NULL REFERENCES clips(id),
      PRIMARY KEY(result_id,clip_id)
    );
    """)
    design_bom.init_schema(db)
    columns = {row[1] for row in db.execute("PRAGMA table_info(results)")}
    if "workspace" not in columns:
        db.execute("ALTER TABLE results ADD COLUMN workspace TEXT NOT NULL DEFAULT 'Control'")
    if "grades_json" not in columns:
        db.execute("ALTER TABLE results ADD COLUMN grades_json TEXT")
    if "dashboard" not in columns:
        db.execute("ALTER TABLE results ADD COLUMN dashboard TEXT NOT NULL DEFAULT 'learning'")
        for result in db.execute("SELECT id,workspace FROM results").fetchall():
            record = dict(result)
            record["sections"] = [{"payload": json.loads(s[0])} for s in db.execute(
                "SELECT payload_json FROM sections WHERE result_id=?", (result["id"],))]
            db.execute("UPDATE results SET dashboard=? WHERE id=?", (dashboard_for(record), result["id"]))
    if "robot_sections_json" not in columns:
        db.execute("ALTER TABLE results ADD COLUMN robot_sections_json TEXT")
    for name in ("study_id", "study_version", "collection_id"):
        if name not in columns:
            db.execute(f"ALTER TABLE results ADD COLUMN {name} TEXT")
    if "record_kind" not in columns:
        db.execute("ALTER TABLE results ADD COLUMN record_kind TEXT NOT NULL DEFAULT 'result'")
    if "archived_at" not in columns:
        db.execute("ALTER TABLE results ADD COLUMN archived_at TEXT")
    if "archive_reason" not in columns:
        db.execute("ALTER TABLE results ADD COLUMN archive_reason TEXT NOT NULL DEFAULT ''")
    db.commit()
    return db


def set_study_archived(db, study_id, reason=None):
    """Human-directed curation only. Retain every version, CAD asset, review and source."""
    rows = db.execute("SELECT id FROM results WHERE study_id=? AND dashboard='design'", (study_id,)).fetchall()
    if not rows:
        raise ValueError(f"Unknown design study: {study_id}")
    if reason is not None and not reason.strip():
        raise ValueError("Archiving needs a written reason")
    db.execute("UPDATE results SET archived_at=CASE WHEN ? IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END,archive_reason=? WHERE study_id=? AND dashboard='design'",
               (reason, reason or "", study_id))
    return len(rows)


def normalize_grades(value):
    """Graded assessment (docs/design/evaluation-principles.md): no pass/fail. Accepts a list of
    dimension rows or {"overall": text, "dimensions": [...], "requirement_challenges": [...]}.
    Each dimension row: dimension, grade 0-5 (0 = unknown), optional confidence, trend, reason."""
    if value is None:
        return None
    if isinstance(value, list):
        value = {"dimensions": value}
    if not isinstance(value, dict):
        raise ValueError("grades must be a list of dimension rows or an object")
    dims = value.get("dimensions", [])
    if not isinstance(dims, list):
        raise ValueError("grades.dimensions must be a list")
    for d in dims:
        if not isinstance(d, dict) or not isinstance(d.get("dimension"), str) or not d["dimension"]:
            raise ValueError("each grade needs a nonempty dimension name")
        g = d.get("grade")
        if not isinstance(g, (int, float)) or isinstance(g, bool) or not 0 <= g <= 5:
            raise ValueError(f"grade for {d['dimension']} must be a number from 0 (unknown) to 5 (viable)")
        if d.get("confidence") not in (None, "low", "medium", "high"):
            raise ValueError("confidence must be low, medium or high")
        if d.get("trend") not in (None, "improving", "flat", "regressing", "new"):
            raise ValueError("trend must be improving, flat, regressing or new")
    return {"overall": value.get("overall", ""), "dimensions": dims,
            "requirement_challenges": value.get("requirement_challenges", [])}


def safe_source(name: str, require_exists: bool = True) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name.startswith("artifacts/"):
        raise ValueError(f"Invalid clip path: {name}")
    absolute = ROOT / path
    if (require_exists and not absolute.is_file()) or not absolute.resolve().is_relative_to(ROOT):
        raise FileNotFoundError(absolute)
    return absolute


def design_ids(sections: list[dict]) -> list[str]:
    """IDs are scoped to a result, never inferred from a title or list position."""
    ids = []
    seen = set()
    for section in sections:
        payload = section.get("payload")
        if not isinstance(payload, dict) or payload.get("kind") != "design_catalog":
            continue
        groups = payload.get("groups")
        if not isinstance(groups, list):
            raise ValueError("Design catalog groups must be a list")
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("options"), list):
                raise ValueError("Design catalog groups need an options list")
            for option in group["options"]:
                key = option.get("id") if isinstance(option, dict) else None
                if (not isinstance(key, str) or not key or len(key) > 200
                        or any(c.isspace() or ord(c) < 32 for c in key) or key in seen):
                    raise ValueError("Design IDs must be unique nonempty strings, at most 200 characters, without whitespace")
                seen.add(key)
                ids.append(key)
    return ids


class ReviewError(ValueError):
    def __init__(self, status: int, message: str, review: dict | None = None):
        super().__init__(message)
        self.status = status
        self.payload = {"error": message}
        if review is not None:
            self.payload["review"] = review


def published_design_ids(db: sqlite3.Connection, result_id: str) -> list[str]:
    if not db.execute("SELECT 1 FROM results WHERE id=?", (result_id,)).fetchone():
        raise ReviewError(404, "Unknown result")
    sections = [{"payload": json.loads(row[0])} for row in db.execute(
        "SELECT payload_json FROM sections WHERE result_id=? ORDER BY position", (result_id,))]
    return design_ids(sections)


def current_review(db: sqlite3.Connection, result_id: str, design_id: str) -> dict:
    row = db.execute("""SELECT design_id,judgment,comment,version,updated_at
        FROM design_reviews WHERE result_id=? AND design_id=?""", (result_id, design_id)).fetchone()
    return dict(row) if row else {"design_id": design_id, "judgment": "unreviewed",
                                 "comment": "", "version": 0, "updated_at": None}


def save_design_review(db: sqlite3.Connection, value: dict) -> dict:
    if not isinstance(value, dict):
        raise ReviewError(400, "Review must be a JSON object")
    result_id, design_id = value.get("result_id"), value.get("design_id")
    judgment, comment, expected = value.get("judgment"), value.get("comment"), value.get("expected_version")
    if any(not isinstance(v, str) or not v or len(v) > 200 for v in (result_id, design_id)):
        raise ReviewError(400, "Nonempty result_id and design_id are required (maximum 200 characters)")
    if not isinstance(judgment, str) or judgment not in REVIEW_JUDGMENTS:
        raise ReviewError(400, "Unknown judgment")
    if not isinstance(comment, str) or len(comment) > 10000 or "\x00" in comment:
        raise ReviewError(400, "Comment must be text of at most 10000 characters without NUL")
    if type(expected) is not int or expected < 0:
        raise ReviewError(400, "expected_version must be a nonnegative integer")
    # Acquire the write lock before reading the version or current published catalog.
    with db:
        db.execute("BEGIN IMMEDIATE")
        if design_id not in published_design_ids(db, result_id):
            raise ReviewError(404, "Design is not in the published catalog")
        current = current_review(db, result_id, design_id)
        if expected != current["version"]:
            raise ReviewError(409, "Review changed; reload before saving", current)
        db.execute("""INSERT INTO design_reviews
            (result_id,design_id,judgment,comment,version,updated_at)
            VALUES (?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            ON CONFLICT(result_id,design_id) DO UPDATE SET judgment=excluded.judgment,
            comment=excluded.comment,version=excluded.version,updated_at=excluded.updated_at""",
            (result_id, design_id, judgment, comment, expected + 1))
        review = current_review(db, result_id, design_id)
        db.execute("INSERT INTO design_review_history VALUES (?,?,?,?,?,?)",
                   (result_id, design_id, review["version"], judgment, comment, review["updated_at"]))
    return review


def save_result(db: sqlite3.Connection, row: dict, clips: list[dict]) -> None:
    if row.get("id") in design_retirements().get("retired_result_ids", []):
        raise ValueError("This obsolete study was deleted during user-directed cleanup; publish new work under its own study identity")
    required = ("id", "title")
    if any(not isinstance(row.get(k), str) or not row[k] for k in required):
        raise ValueError("Result needs a nonempty id and title")
    if dashboard_for(row) == "design":
        if not row.get("study_id") or not row.get("version") or not isinstance(row.get("design"), dict):
            raise ValueError("A design study is ONE variant: provide study_id, version, robot_sections and one design object. Use design-template --robot-section Knee. Comparative legacy collections belong in Archive.")
        if not row.get("robot_sections"):
            raise ValueError("A design study needs explicit robot_sections")
        if not isinstance(row["study_id"], str) or not isinstance(row["version"], str):
            raise ValueError("study_id and version must be nonempty strings")
        if any(not isinstance(row["design"].get(k), str) or not row["design"][k].strip()
               for k in ("id", "title", "summary", "visual_kind", "maturity")):
            raise ValueError("design needs id, title, short summary, visual_kind and maturity")
        if row["design"].get("cad", {}).get("model_units", "m") not in {"m", "mm"}:
            raise ValueError("cad.model_units must be m or mm")
        prior = db.execute("SELECT study_id,study_version FROM results WHERE id=?", (row["id"],)).fetchone()
        if prior and (prior[0] != row["study_id"] or prior[1] != str(row["version"])):
            raise ValueError("A result id belongs to one study version; publish another version under a new result id")
        if any(isinstance(s.get("payload"), dict) and s["payload"].get("kind") == "design_catalog" for s in row.get("sections", [])):
            raise ValueError("Use the single design object for this study version; do not publish a multi-variant design_catalog")
        row = dict(row)
        row["sections"] = [{"title": "Design variant", "payload": {"kind": "design_catalog", "groups": [{
            "id": row["study_id"], "title": row["design"]["title"], "category": "Variant",
            "robot_sections": row["robot_sections"], "options": [row["design"]]}]}}] + row.get("sections", [])
        existing = db.execute("SELECT id FROM results WHERE study_id=? AND study_version=?", (row["study_id"], str(row["version"]))).fetchone()
        if existing and existing[0] != row["id"]:
            raise ValueError("This study version already has a result id; revise its text using that id, or publish a new version")
        if existing:
            cad = row["design"].get("cad", {})
            paths = [v["model_path"] for v in cad.get("views", [])]
            paths += [cad[k] for k in ("parts_path", "step_path") if cad.get(k)]
            source = db.execute("SELECT payload_json FROM source_payload WHERE result_id=?", (row["id"],)).fetchone()
            if source:
                old_cad = json.loads(source[0]).get("design", {}).get("cad", {})
                old_paths = [v["model_path"] for v in old_cad.get("views", [])]
                old_paths += [old_cad[k] for k in ("parts_path", "step_path") if old_cad.get(k)]
                if set(paths) != set(old_paths):
                    raise ValueError("CAD inventory changed for a retained version; publish a new version")
            for path in paths:
                old = db.execute("SELECT sha256 FROM assets WHERE result_id=? AND path=?", (row["id"], path)).fetchone()
                if old is None or old[0] != digest_file(source_asset(path)):
                    raise ValueError("CAD changed for a retained study version; publish a new version and result id")
    design_ids(row.get("sections", []))
    if isinstance(row.get("design"), dict) and "bom" in row["design"]:
        design_bom.validate_bom(row["design"]["bom"], {a["path"] for a in row.get("assets", [])})
    validate_design_registration({**row, "clips": clips})
    db.execute("""INSERT INTO results
      (id,title,area,status,summary,observation,measurements,limitations,body,provenance,matrix_json)
      VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
      title=excluded.title,area=excluded.area,status=excluded.status,
      summary=excluded.summary,observation=excluded.observation,
      measurements=excluded.measurements,limitations=excluded.limitations,
      body=excluded.body,provenance=excluded.provenance,matrix_json=excluded.matrix_json,
      updated_at=CURRENT_TIMESTAMP""",
      (row["id"], row["title"], row.get("area", ""), row.get("status", ""),
       row.get("summary", ""), row.get("observation", ""), row.get("measurements", ""),
       row.get("limitations", ""), row.get("body", ""),
       json.dumps(row.get("provenance", [])), json.dumps(row.get("matrix")) if row.get("matrix") is not None else None))
    for clip in clips:
        source = safe_source(clip["source_path"], require_exists=False)
        if source.is_file():
            size = source.stat().st_size
            digest = digest_file(source)
            if source.stat().st_size != size:
                raise ValueError(f"Clip changed during inventory: {source}")
        else:
            archived = db.execute("""SELECT sha256,bytes FROM clips
                WHERE id=? AND source_path=? AND verified_at IS NOT NULL""",
                (clip["id"], clip["source_path"])).fetchone()
            if archived is None:
                raise FileNotFoundError(source)
            digest, size = archived
        previous = db.execute("SELECT source_path,sha256,result_id FROM clips WHERE id=?", (clip["id"],)).fetchone()
        if previous and (previous["source_path"] != clip["source_path"] or previous["sha256"] != digest):
            raise ValueError(f"Clip ID is immutable; publish a new ID for {clip['id']}")
        db.execute("""INSERT INTO clips
          (id,result_id,title,role,body,observation,source_path,report_path,trace_path,sha256,bytes)
          VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
          title=excluded.title,role=excluded.role,
          body=excluded.body,observation=excluded.observation,
          source_path=excluded.source_path,report_path=excluded.report_path,
          trace_path=excluded.trace_path,sha256=excluded.sha256,bytes=excluded.bytes,
          bucket_key=CASE WHEN clips.sha256=excluded.sha256 THEN clips.bucket_key ELSE NULL END,
          verified_at=CASE WHEN clips.sha256=excluded.sha256 THEN clips.verified_at ELSE NULL END""",
          (clip["id"], row["id"], clip.get("title", clip["id"]), clip.get("role", ""),
           clip.get("body", ""), clip.get("observation", ""), clip["source_path"],
           clip.get("report_path"), clip.get("trace_path"), digest, size))
        # A retained recording can support another inspection revision without
        # taking it away from the version that originally published it.
        if previous and previous["result_id"] != row["id"]:
            db.execute("INSERT OR IGNORE INTO design_study_clips (result_id,clip_id) VALUES (?,?)",
                       (row["id"], clip["id"]))
        else:
            db.execute("DELETE FROM design_study_clips WHERE result_id=? AND clip_id=?",
                       (row["id"], clip["id"]))
    db.execute("DELETE FROM sections WHERE result_id=?", (row["id"],))
    for position, section in enumerate(row.get("sections", [])):
        db.execute("INSERT INTO sections VALUES (?,?,?,?,?)",
                   (row["id"], position, section["title"], section.get("note", ""),
                    json.dumps(section.get("payload"), ensure_ascii=False)))
    db.execute("INSERT OR REPLACE INTO source_payload VALUES (?,?)",
               (row["id"], json.dumps(row.get("source_payload", row), ensure_ascii=False)))
    grades = normalize_grades(row.get("grades"))
    db.execute("UPDATE results SET grades_json=? WHERE id=?",
               (json.dumps(grades, ensure_ascii=False) if grades is not None else None, row["id"]))
    workspace = row.get("workspace") or ("Perception" if "perception" in row.get("area", "").lower()
                                         else "Control")
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace must be a nonempty name")
    db.execute("UPDATE results SET workspace=? WHERE id=?", (workspace.strip(), row["id"]))
    if row.get("dashboard") is not None and row["dashboard"] not in {"learning", "design"}:
        raise ValueError("dashboard must be learning or design")
    db.execute("UPDATE results SET dashboard=? WHERE id=?", (dashboard_for(row), row["id"]))
    db.execute("UPDATE results SET robot_sections_json=? WHERE id=?",
               (json.dumps(row["robot_sections"]) if row.get("robot_sections") else None, row["id"]))
    db.execute("UPDATE results SET study_id=?,study_version=?,record_kind=? WHERE id=?",
               (row.get("study_id"), str(row["version"]) if row.get("version") is not None else None,
                "study" if dashboard_for(row) == "design" else "result", row["id"]))
    if row.get("study_id"):
        prior_archive = db.execute("SELECT archived_at,archive_reason FROM results WHERE study_id=? AND archived_at IS NOT NULL LIMIT 1",
                                   (row["study_id"],)).fetchone()
        if prior_archive:
            db.execute("UPDATE results SET archived_at=?,archive_reason=? WHERE id=?", (*prior_archive, row["id"]))
    assets = row.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("assets must be a list")
    seen = set()
    for asset in assets:
        asset_id = asset["id"]
        if not isinstance(asset_id, str) or not asset_id or asset_id in seen:
            raise ValueError("asset IDs must be distinct nonempty strings")
        seen.add(asset_id)
        path = source_asset(asset["path"])
        kind = asset.get("kind", "data")
        if kind not in {"image", "model", "data"}:
            raise ValueError(f"Unsupported asset kind: {kind}")
        db.execute("""INSERT INTO assets(result_id,id,title,kind,path,sha256,bytes)
            VALUES (?,?,?,?,?,?,?) ON CONFLICT(result_id,id) DO UPDATE SET
            title=excluded.title,kind=excluded.kind,path=excluded.path,
            sha256=excluded.sha256,bytes=excluded.bytes""",
            (row["id"], asset_id, asset.get("title", asset_id), kind,
             asset["path"], digest_file(path), path.stat().st_size))
    db.execute("DELETE FROM assets WHERE result_id=? AND id NOT IN (SELECT value FROM json_each(?))",
               (row["id"], json.dumps(list(seen))))
    if isinstance(row.get("design"), dict) and "bom" in row["design"]:
        design_bom.put_bom(db, row["id"], row["design"]["bom"])
    elif dashboard_for(row) == "design":
        # Text/CAD republication without a BOM preserves dashboard edits, but may
        # not silently remove a source asset used by the retained BOM.
        retained = design_bom.get_bom(db, row["id"])["bom"]
        if retained is not None:
            design_bom.validate_bom(retained, {a["path"] for a in assets})


def source_asset(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name.startswith(("artifacts/", "studies/")):
        raise ValueError(f"Invalid asset path: {name}")
    absolute = ROOT / path
    if not absolute.is_file():
        from lab.blob_store import ensure
        ensure(absolute)
    if not absolute.is_file() or not absolute.resolve().is_relative_to(ROOT):
        raise FileNotFoundError(absolute)
    return absolute


def import_catalog(db: sqlite3.Connection) -> None:
    with gzip.open(CATALOG, "rt", encoding="utf-8") as stream:
        catalog = json.load(stream)
    with gzip.open(QUALITATIVE, "rt", encoding="utf-8") as stream:
        behavior = {item["group_id"]: item for item in json.load(stream)["assessments"]}
    for group in catalog["groups"]:
        matrix = None
        if group.get("matrix", {}).get("path"):
            path = ROOT / group["matrix"]["path"]
            if path.is_file() and path.stat().st_size < 5_000_000:
                matrix = json.loads(path.read_text())
        row = {
            "id": group["id"], "title": group["title"], "area": group.get("area", ""),
            "status": group.get("current_status", ""), "summary": group.get("evaluation", ""),
            "observation": group.get("result", ""), "measurements": group.get("metric", ""),
            "limitations": group.get("limitation", ""),
            "body": group.get("current_policy", {}).get("body", "") if isinstance(group.get("current_policy"), dict) else "",
            "provenance": group.get("evidence", []), "matrix": matrix,
            "source_payload": group,
        }
        sections = []
        for item in group.get("extra_matrices", []):
            path = ROOT / item.get("path", "")
            if path.is_file() and path.stat().st_size < 5_000_000:
                sections.append({"title": item.get("title", path.name), "note": item.get("note", ""),
                                 "payload": json.loads(path.read_text())})
        for key, title in (("overview", "Assessment"), ("latest_training", "Latest training"),
                           ("current_policy", "Current policy"), ("policy_ids", "Policy IDs")):
            if group.get(key):
                sections.append({"title": title, "payload": group[key]})
        if group["id"] in behavior:
            sections.append({"title": "Behavior review", "payload": behavior[group["id"]]})
        row["sections"] = sections
        clips = [{"id": f"{group['id']}:{c['id']}", "title": c.get("title", c["id"]),
                  "role": c.get("role", ""), "body": c.get("body", ""),
                  "observation": c.get("note", ""), "source_path": c["path"],
                  "report_path": c.get("report"), "trace_path": c.get("trace")}
                 for c in group.get("clips", []) if c.get("path")]
        with db:
            save_result(db, row, clips)


@lru_cache(maxsize=1)
def s3_client():
    return archive_client()


def upload(db: sqlite3.Connection, limit: int | None = None) -> None:
    client = s3_client()
    rows = db.execute("SELECT * FROM clips WHERE verified_at IS NULL ORDER BY result_id,id").fetchall()
    verified = 0
    for row in rows[:limit]:
        source = safe_source(row["source_path"])
        if source.stat().st_size != row["bytes"] or digest_file(source) != row["sha256"]:
            raise ValueError(f"Source changed: {source}")
        key = f"{PREFIX}/{row['sha256'][:2]}/{row['sha256'][2:]}.mp4"
        try:
            head = client.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] != row["bytes"]:
                raise ValueError(f"Wrong size in bucket: {key}")
        except client.exceptions.ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) not in {"404", "NoSuchKey", "NotFound"}:
                raise
            client.upload_file(str(source), BUCKET, key,
                               ExtraArgs={"ContentType": "video/mp4", "Metadata": {"sha256": row["sha256"]}})
        response = client.get_object(Bucket=BUCKET, Key=key)
        digest = hashlib.sha256()
        count = 0
        try:
            for block in iter(lambda: response["Body"].read(1024 * 1024), b""):
                digest.update(block)
                count += len(block)
        finally:
            response["Body"].close()
        if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError(f"Bucket read-back mismatch: {key}")
        with db:
            db.execute("UPDATE clips SET bucket_key=?, verified_at=CURRENT_TIMESTAMP WHERE id=?", (key, row["id"]))
        verified += 1
        print(f"Verified {verified}/{min(len(rows), limit or len(rows))}: {row['id']}", flush=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI), **kwargs)

    def end_headers(self):
        if not urlsplit(self.path).path.startswith(("/asset", "/artifacts/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def json_response(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def local_review_write(self) -> bool:
        """Reject external browser writes and DNS-rebound Host names."""
        try:
            host = urlsplit("http://" + self.headers.get("Host", ""))
            if (host.hostname not in {"127.0.0.1", "localhost", "::1"}
                    or host.port != self.server.server_port or host.username
                    or host.password or host.path or host.query or host.fragment):
                return False
            origin_header = self.headers.get("Origin")
            if origin_header is None:
                return self.headers.get("Sec-Fetch-Site") not in {"cross-site", "same-site"}
            origin = urlsplit(origin_header)
            return (origin.scheme == "http" and origin.hostname == host.hostname
                    and origin.port == host.port and not origin.username
                    and not origin.password and not origin.path and not origin.query
                    and not origin.fragment)
        except ValueError:
            return False

    def do_POST(self):
        endpoint = urlsplit(self.path).path
        if endpoint not in {"/api/design-reviews", "/api/design-preferences", "/api/design-bom", "/api/robot-bom"}:
            return self.json_response({"error": "Unknown endpoint"}, 404)
        if not self.local_review_write():
            return self.json_response({"error": "Review writes require the local dashboard origin"}, 403)
        if self.headers.get_content_type() != "application/json":
            return self.json_response({"error": "Use application/json"}, 415)
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            return self.json_response({"error": "One Content-Length is required"}, 400)
        try:
            length = int(lengths[0])
        except ValueError:
            return self.json_response({"error": "Invalid Content-Length"}, 400)
        max_body = 2 * 1024 * 1024 if endpoint in {"/api/design-bom", "/api/robot-bom"} else MAX_REVIEW_BODY
        if length <= 0 or length > max_body:
            return self.json_response({"error": f"Dashboard body must be between 1 and {max_body} bytes"}, 413)
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Use a JSON object")
            # Escaped lone surrogates are JSON-readable but cannot be stored as UTF-8.
            json.dumps(value, ensure_ascii=False).encode("utf-8")
            if endpoint in {"/api/design-bom", "/api/robot-bom"}:
                with connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    if endpoint == "/api/design-bom":
                        if type(value.get("expected_revision")) is not int or not isinstance(value.get("result_id"), str):
                            raise design_bom.BomError("result_id and expected_revision are required")
                        design_bom.put_bom(db, value["result_id"], value.get("bom"), value["expected_revision"])
                        report = design_bom.study_report(db, value["result_id"])
                    else:
                        design_bom.put_plan(db, value.get("plan"), value.get("expected_revision"))
                        report = design_bom.robot_report(db)
                return self.json_response(report)
            with connect() as db:
                if endpoint == "/api/design-preferences":
                    section, study_id = value.get("section"), value.get("study_id")
                    if not isinstance(section, str) or not isinstance(study_id, str):
                        raise ValueError("section and study_id must be names")
                    rows = db.execute("SELECT robot_sections_json FROM results WHERE study_id=? AND archived_at IS NULL", (study_id,)).fetchall()
                    if not section or not rows or not any(section in json.loads(r[0] or "[]") for r in rows):
                        raise ValueError("Choose an existing study registered for this robot section")
                    db.execute("INSERT INTO design_preferences(section,study_id) VALUES (?,?) ON CONFLICT(section) DO UPDATE SET study_id=excluded.study_id,updated_at=CURRENT_TIMESTAMP", (section, study_id))
                    return self.json_response({"section": section, "study_id": study_id})
                review = save_design_review(db, value)
            return self.json_response({"review": review})
        except ReviewError as exc:
            return self.json_response(exc.payload, exc.status)
        except design_bom.BomError as exc:
            return self.json_response({"error": str(exc)}, exc.status)
        except (ValueError, UnicodeError):
            return self.json_response({"error": "Invalid dashboard JSON"}, 400)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path in {"/api/design-bom", "/api/robot-bom"}:
            query = parse_qs(parsed.query)
            try:
                with connect() as db:
                    db.execute("BEGIN")
                    revision = query.get("revision", [None])[0]
                    report = (design_bom.study_report(db, query.get("result", [""])[0], revision)
                              if parsed.path == "/api/design-bom" else design_bom.robot_report(db))
                if query.get("format") == ["csv"]:
                    import csv
                    import io
                    output = io.StringIO()
                    fields = ["assembly", "result_id", "study_id", "study_version", "bom_revision", "assembly_count",
                              "id", "name", "category", "part_number", "supplier", "material", "quantity", "unit",
                              "unit_cost", "currency", "extended_cost", "cost_basis", "price_date",
                              "unit_mass_g", "extended_mass_g", "lead_time_days", "notes", "source_refs"]
                    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    for line in report["lines"]:
                        item = {**line, "source_refs": json.dumps(line.get("source_refs", []), ensure_ascii=False)}
                        # Keep spreadsheet software from executing source text as formulas.
                        writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else v for k, v in item.items()})
                    raw = output.getvalue().encode("utf-8-sig")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="robot-design-bom.csv"')
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                return self.json_response(report)
            except design_bom.BomError as exc:
                return self.json_response({"error": str(exc)}, exc.status)
        if parsed.path == "/api/design-preferences":
            with connect() as db:
                return self.json_response([dict(r) for r in db.execute("SELECT * FROM design_preferences")])
        if parsed.path == "/api/design-reviews":
            result_id = parse_qs(parsed.query).get("result", [""])[0]
            if not result_id:
                return self.json_response({"error": "result is required"}, 400)
            try:
                with connect() as db:
                    db.execute("BEGIN")
                    ids = published_design_ids(db, result_id)
                    reviews = [current_review(db, result_id, key) for key in ids]
                return self.json_response({"result_id": result_id, "reviews": reviews})
            except ReviewError as exc:
                return self.json_response(exc.payload, exc.status)
        if parsed.path == "/api/sessions":
            dashboard = parse_qs(parsed.query).get("dashboard", ["learning"])[0]
            retirements = design_retirements()
            with connect() as db:
                published = {row[0] for row in db.execute("SELECT id FROM results")}
                collections = [dict(r) for r in db.execute("SELECT id,title,area,summary,study_id,study_version,robot_sections_json,archived_at,archive_reason FROM results WHERE (record_kind='collection' OR archived_at IS NOT NULL) AND dashboard=?", (dashboard,))]
                # Archived studies appear once; their version menu retains all historical CAD.
                version_key = lambda r: tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", r.get("study_version") or ""))
                latest_archived = {}
                for row in sorted(collections, key=version_key, reverse=True):
                    latest_archived.setdefault(row["study_id"] or row["id"], row)
                collections = list(latest_archived.values())
                for row in collections:
                    row.update(kind="Archived design study" if row["archived_at"] else "Historical comparison", videos=[], models=[], images=[], data=[], landing=None,
                               design_sections=json.loads(row["robot_sections_json"] or "null") or design_sections(row["area"] + " " + row["title"]))
                    if row["archived_at"]:
                        row["summary"] = row["archive_reason"]
                        row["result_id"] = row["id"]
                    for a in db.execute("SELECT title,kind,path FROM assets WHERE result_id=?", (row["id"],)):
                        from urllib.parse import quote
                        row[{"model": "models", "image": "images", "data": "data"}[a["kind"]]].append({"name": a["title"], "url": "/asset?path=" + quote(a["path"], safe="")})
                    for clip in db.execute("SELECT id,title FROM clips WHERE result_id=?", (row["id"],)):
                        row["videos"].append({"name": clip["title"], "clip_id": clip["id"]})
            rows = []
            for row in catalog():
                source_dashboard = retirements.get("artifact_dashboard_overrides", {}).get(row["id"], archive_dashboard(row))
                if row["id"] in published or row["id"] in retirements.get("retired_artifact_ids", []) or source_dashboard != dashboard:
                    continue
                row["design_sections"] = design_sections(row["id"] + " " + row["title"])
                excluded = retirements.get("artifact_section_exclusions", {}).get(row["id"], [])
                row["design_sections"] = [section for section in row["design_sections"] if section not in excluded]
                rows.append(row)
            return self.json_response({"sessions": collections + rows})
        if parsed.path == "/asset":
            name = parse_qs(parsed.query).get("path", [""])[0]
            if not name.startswith("artifacts/"):
                with connect() as db:
                    if not db.execute("SELECT 1 FROM assets WHERE path=?", (name,)).fetchone():
                        return self.send_error(404, "Asset unpublished")
            try:
                path = source_asset(name)
            except (ValueError, FileNotFoundError):
                return self.send_error(404, "Asset unavailable")
            except RuntimeError:
                return self.send_error(503, "Archived asset temporarily unavailable")
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    self.wfile.write(block)
            return
        if parsed.path == "/api/preview":
            name = parse_qs(parsed.query).get("path", [""])[0]
            if not name.startswith("artifacts/"):
                with connect() as db:
                    if not db.execute("SELECT 1 FROM assets WHERE path=?", (name,)).fetchone():
                        return self.json_response({"error": "Asset unpublished"}, 404)
            try:
                path = source_asset(name)
            except (ValueError, FileNotFoundError):
                return self.json_response({"error": "Asset unavailable"}, 404)
            except RuntimeError:
                return self.json_response({"error": "Archived asset temporarily unavailable"}, 503)
            if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".log"}:
                return self.json_response({"error": "Preview unavailable"}, 400)
            with path.open("rb") as stream:
                content = stream.read(120_001)
            return self.json_response({"text": content[:120_000].decode("utf-8", errors="replace"),
                                       "truncated": len(content) > 120_000})
        if parsed.path.startswith(("/artifacts/", "/ui/session-inspector/viewer3d.js")):
            from urllib.parse import unquote
            logical_name = unquote(parsed.path.lstrip("/"))
            if ".." in Path(logical_name).parts:
                return self.send_error(404, "Artifact unavailable")
            if parsed.path.startswith("/artifacts/"):
                from lab.blob_store import ensure
                logical = ROOT / logical_name
                if not logical.is_file() and logical.parent.resolve().is_relative_to(ROOT):
                    try:
                        ensure(logical)
                    except RuntimeError:
                        return self.send_error(503, "Archived asset temporarily unavailable")
            path = (ROOT / logical_name).resolve()
            if not path.is_file() or not path.is_relative_to(ROOT):
                return self.send_error(404, "Artifact unavailable")
            self.directory = str(ROOT)
            return super().do_GET()
        if parsed.path == "/api/design":
            pointer = json.loads((ROOT / "robot" / "ape-current-design.json").read_text())
            return self.json_response({"name": pointer.get("name"), "status": pointer.get("status"),
                                       "selected_design_id": pointer.get("selected_design_id"),
                                       "evidence": pointer.get("evidence", {}).get("report")})
        with connect() as db:
            if parsed.path in {"/", "/design/"}:
                key = parse_qs(parsed.query).get("result", [""])[0]
                target = db.execute("SELECT dashboard FROM results WHERE id=?", (key,)).fetchone()
                session_id = parse_qs(parsed.query).get("session", [""])[0]
                if session_id:
                    session = next((s for s in catalog() if s["id"] == session_id), None)
                    if session:
                        target = (archive_dashboard(session),)
                if target and (target[0] == "design") != (parsed.path == "/design/"):
                    self.send_response(302)
                    self.send_header("Location", ("/design/" if target[0] == "design" else "/") + "?" + parsed.query)
                    self.end_headers()
                    return
            if parsed.path == "/api/results":
                dashboard = parse_qs(parsed.query).get("dashboard", ["learning"])[0]
                rows = db.execute("""SELECT r.id,r.title,r.area,r.workspace,r.dashboard,r.robot_sections_json,r.study_id,r.study_version,r.record_kind,r.status,r.observation,r.measurements,
                    COUNT(c.id)+(SELECT COUNT(*) FROM design_study_clips sc WHERE sc.result_id=r.id) AS clip_count,
                    SUM(c.verified_at IS NOT NULL)+(SELECT COUNT(*) FROM design_study_clips sc JOIN clips cc ON cc.id=sc.clip_id WHERE sc.result_id=r.id AND cc.verified_at IS NOT NULL) AS uploaded_clips,
                    (SELECT COUNT(*) FROM assets a WHERE a.result_id=r.id) AS asset_count
                    ,(SELECT COUNT(*) FROM assets a WHERE a.result_id=r.id AND a.kind='model') AS model_count
                    FROM results r LEFT JOIN clips c ON c.result_id=r.id
                    WHERE r.dashboard=? AND r.record_kind!='collection' AND r.archived_at IS NULL GROUP BY r.id ORDER BY r.updated_at DESC,r.title""", (dashboard,)).fetchall()
                records = []
                for row in rows:
                    record = dict(row)
                    record["robot_sections"] = json.loads(record.pop("robot_sections_json") or "null")
                    if dashboard == "design":
                        record["sections"] = [{"payload": json.loads(s[0])} for s in db.execute(
                            "SELECT payload_json FROM sections WHERE result_id=?", (record["id"],))]
                        annotate_design_groups(record)
                        record.pop("sections")
                    records.append(record)
                return self.json_response(records)
            if parsed.path == "/api/result":
                key = parse_qs(parsed.query).get("id", [""])[0]
                row = db.execute("SELECT * FROM results WHERE id=?", (key,)).fetchone()
                if row is None:
                    if key in design_retirements().get("retired_result_ids", []):
                        return self.json_response({"error": "This obsolete hip study was deleted during cleanup.", "section": "Hip"}, 410)
                    return self.json_response({"error": "Unknown result"}, 404)
                result = dict(row)
                result["robot_sections"] = json.loads(result.pop("robot_sections_json") or "null")
                result["provenance"] = json.loads(result["provenance"])
                result["matrix"] = json.loads(result.pop("matrix_json")) if result["matrix_json"] else None
                grades_json = result.pop("grades_json", None)
                result["grades"] = json.loads(grades_json) if grades_json else None
                result["clips"] = [dict(c) for c in db.execute("SELECT * FROM clips WHERE result_id=? OR id IN (SELECT clip_id FROM design_study_clips WHERE result_id=?) ORDER BY id", (key, key))]
                result["assets"] = [dict(a) for a in db.execute("SELECT * FROM assets WHERE result_id=? ORDER BY kind,path", (key,))]
                result["sections"] = [{"title": s["title"], "note": s["note"],
                                       "payload": json.loads(s["payload_json"])} for s in
                                      db.execute("SELECT * FROM sections WHERE result_id=? ORDER BY position", (key,))]
                result["versions"] = [dict(v) for v in db.execute("SELECT id,study_version,title,updated_at FROM results WHERE study_id=?", (result["study_id"],))] if result["study_id"] else []
                if result["record_kind"] == "collection":
                    result["variants"] = [dict(v) for v in db.execute("SELECT id,study_id,study_version,title FROM results WHERE collection_id=?", (key,))]
                return self.json_response(annotate_design_groups(result))
            if parsed.path == "/api/video":
                key = parse_qs(parsed.query).get("id", [""])[0]
                row = db.execute("SELECT bucket_key FROM clips WHERE id=? AND verified_at IS NOT NULL", (key,)).fetchone()
                if row is None:
                    return self.json_response({"error": "Video not uploaded"}, 404)
                try:
                    url = s3_client().generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": row["bucket_key"]}, ExpiresIn=3600)
                except (OSError, RuntimeError, subprocess.CalledProcessError):
                    return self.json_response({"error": "Video credentials unavailable"}, 503)
                return self.json_response({"url": url})
            if parsed.path == "/video":
                source = parse_qs(parsed.query).get("path", [""])[0]
                row = db.execute("""SELECT bucket_key FROM clips WHERE source_path=? AND verified_at IS NOT NULL
                                    LIMIT 1""", (source,)).fetchone()
                if row is None:
                    return self.json_response({"error": "Video not uploaded"}, 404)
                try:
                    url = s3_client().generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": row["bucket_key"]}, ExpiresIn=3600)
                except (OSError, RuntimeError, subprocess.CalledProcessError):
                    return self.json_response({"error": "Video credentials unavailable"}, 503)
                self.send_response(302)
                self.send_header("Location", url)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
        if parsed.path == "/":
            self.path = "/index.html"
        elif parsed.path in {"/design", "/design/"}:
            self.path = "/design.html"
        super().do_GET()


def design_template(section):
    """An authoring template, never a generated engineering result."""
    slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
    root = f"artifacts/{slug}-study"
    template = {
        "id": f"{slug}-study-YYYYMMDD", "title": f"{section} mechanism study",
        "dashboard": "design", "workspace": "Body", "robot_sections": [section],
        "area": f"{section} design", "observation": "Replace with the main observed behavior or design finding.",
        "measurements": "Replace with native-unit evidence and the sampled conditions.",
        "limitations": "Replace with unresolved physical, manufacturing and simulation assumptions.",
        "provenance": [f"{root}/source-manifest.json"],
        "assets": [{"id": key, "title": title, "kind": kind, "path": f"{root}/{filename}"}
                   for key, title, kind, filename in [
                       ("neutral", "Neutral assembly", "model", "assembly.glb"),
                       ("step", "Editable native assembly", "data", "assembly.step"),
                       ("parts", "Named parts and joint metadata", "data", "parts.json"),
                       ("preview", "Source render", "image", "preview.png"),
                       ("report", "Native simulation report", "data", "report.json"),
                       ("trace", "Full simulation trace", "data", "trace.jsonl"),
                       ("config", "Exact simulation configuration", "data", "config.json")]],
        "sections": [{"title": "Mechanism alternatives", "payload": {
            "kind": "design_catalog", "title": f"{section} alternatives", "groups": [{
                "id": f"{slug}-round-1", "title": "Round 1", "category": "Mechanisms",
                "robot_sections": [section], "options": [{
                    "id": f"{slug}-A-r1", "title": "Replace with the mechanism name",
                    "summary": "One short sentence describing the trade-off.",
                    "description": "Mechanism, load path and unresolved engineering detail.",
                    "image_path": f"{root}/preview.png", "visual_kind": "native_CAD_assembly",
                    "maturity": "Exploratory CAD; physical behavior unmeasured",
                    "cad": {"model_units": "m", "views": [{"label": "Neutral assembly", "model_path": f"{root}/assembly.glb"}],
                            "parts_path": f"{root}/parts.json", "step_path": f"{root}/assembly.step",
                            "report_path": f"{root}/report.json", "scope": "State the exact source, modeled links and missing parts."},
                    "source_refs": [{"label": "Native simulation report", "path": f"{root}/report.json"}],
                    "evidence": [{"id": "simulation-r1", "kind": "simulation", "title": "Replace with the simulated condition",
                        "observation": "Replace with what actually happened, including degradation.",
                        "measurements": [{"name": "Peak joint effort", "value": None, "unit": "N m", "condition": "Replace with the actual load and pose"}],
                        "conditions": {"source_config": f"{root}/config.json", "sampling": "Record episodes, starts, horizon and solver settings where relevant"},
                        "limitations": "List unmodeled contact, flexibility, actuator and load assumptions.",
                        "source_refs": [{"label": label, "path": f"{root}/{filename}"} for label, filename in
                                        [("Report", "report.json"), ("Full trace", "trace.jsonl"), ("Configuration", "config.json")]],
                        "clip_ids": []}]}]}]}}], "clips": []}
    template["study_id"] = f"{slug}-variant-a"
    template["version"] = "v1"
    template["id"] = f"{slug}-variant-a-v1"
    template["title"] = f"{section} · variant A"
    template["design"] = template.pop("sections")[0]["payload"]["groups"][0]["options"][0]
    template["design"]["id"] = f"{slug}-variant-a-v1"
    template["design"]["bom"] = {
        "scope": "One assembly of this exact design version. Replace with actual scope.",
        "completeness": "partial", "notes": "Unknown costs and mass remain null. State tax, shipping and manufacturing assumptions.",
        "exclusions": ["List omitted hardware, electronics, labor and other costs."],
        "items": [{"id": "part-a", "name": "Replace with a real part", "quantity": None, "unit": "ea",
                   "unit_cost": None, "currency": "", "cost_basis": "unknown", "price_date": "",
                   "unit_mass_g": None, "lead_time_days": None, "supplier": "", "part_number": "",
                   "material": "", "category": "", "source_refs": [], "notes": ""}]}
    return template


def migrate_design_studies(db):
    """Retain legacy comparisons, index each source option as one variant/version."""
    created = 0
    sources = db.execute("SELECT * FROM results WHERE dashboard='design' AND study_id IS NULL AND record_kind='result'").fetchall()
    for source in sources:
        parent = dict(source)
        sections = [{"title": s["title"], "payload": json.loads(s["payload_json"])} for s in db.execute(
            "SELECT * FROM sections WHERE result_id=? ORDER BY position", (parent["id"],))]
        assets = [dict(a) for a in db.execute("SELECT * FROM assets WHERE result_id=?", (parent["id"],))]
        for section in sections:
            payload = section["payload"]
            if not isinstance(payload, dict) or payload.get("kind") != "design_catalog":
                continue
            for group in payload["groups"]:
                for original in group["options"]:
                    option = dict(original)
                    match = re.fullmatch(r"(.+?)[-_]r(\d+[a-z]?)", option["id"], re.I)
                    variant, version = (match[1], "r" + match[2]) if match else (option["id"], "v1")
                    key = parent["id"] + ":" + option["id"]
                    if db.execute("SELECT 1 FROM results WHERE id=?", (key,)).fetchone():
                        continue
                    refs = set()
                    def collect(value):
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if isinstance(v, str) and (k == "path" or k.endswith("_path")):
                                    refs.add(v)
                                else:
                                    collect(v)
                        elif isinstance(value, list):
                            for v in value:
                                collect(v)
                    collect(option)
                    manifests = [p for p in refs if p.endswith("/design.json")]
                    # Retained neutral exports carry an explicit competitor/revision name.
                    model_paths = set()
                    for path in manifests:
                        native = re.search(r"/([A-C])/([^/]+)/design.json$", path)
                        if native:
                            name = f"{native[1]}-{native[2]}-neutral-native.glb"
                            model_paths.update(a["path"] for a in assets if Path(a["path"]).name == name)
                        model_paths.update(a["path"] for a in assets if a["kind"] == "model" and Path(a["path"]).parent == Path(path).parent)
                    if not option.get("cad") and model_paths:
                        option["cad"] = {"model_units": "m", "views": [{"label": Path(p).stem, "model_path": p} for p in sorted(model_paths)],
                                         "parts_path": manifests[0] if manifests else None}
                    refs.update(model_paths)
                    if not option.get("cad") and option.get("image_path"):
                        lane = re.search(r"[-_]([ABC])$", option["id"])
                        if lane:
                            candidates = [a for a in assets if a["kind"] == "model" and Path(a["path"]).parent == Path(option["image_path"]).parent and Path(a["path"]).name.startswith(lane[1] + "-")]
                            if candidates:
                                option["cad"] = {"views": [{"label": a["title"], "model_path": a["path"]} for a in candidates]}
                                refs.update(a["path"] for a in candidates)
                    labels = design_sections(group.get("id", "") + " " + group.get("title", ""))
                    if labels == ["Whole robot"]:
                        labels = design_sections(parent["area"] + " " + parent["title"])
                    row = {**parent, "id": key, "title": option["title"], "study_id": parent["id"] + ":" + variant,
                           "study_version": version, "collection_id": parent["id"], "record_kind": "study",
                           "robot_sections_json": json.dumps(labels), "observation": option.get("summary", ""),
                           "summary": option.get("description", ""), "measurements": "See the source-matched native reports for this revision.",
                           "limitations": option.get("maturity", ""), "matrix_json": None, "grades_json": None}
                    columns = list(row)
                    db.execute(f"INSERT INTO results ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", list(row.values()))
                    group_payload = {"kind": "design_catalog", "groups": [{"id": group["id"], "title": option["title"], "category": "Variant", "robot_sections": labels, "options": [option]}]}
                    db.execute("INSERT INTO sections VALUES (?,?,?,?,?)", (key, 0, "Design variant", "Migrated source option; original comparison retained in Archive.", json.dumps(group_payload)))
                    db.execute("INSERT INTO assets SELECT ?,id,title,kind,path,sha256,bytes FROM assets WHERE result_id=? AND path IN (SELECT value FROM json_each(?))", (key, parent["id"], json.dumps(list(refs))))
                    for table in ("design_reviews", "design_review_history"):
                        db.execute(f"INSERT INTO {table} SELECT ?,design_id,judgment,comment,version,updated_at FROM {table} WHERE result_id=? AND design_id=?" if table == "design_reviews" else
                                   f"INSERT INTO {table} SELECT ?,design_id,version,judgment,comment,updated_at FROM {table} WHERE result_id=? AND design_id=?", (key, parent["id"], option["id"]))
                    db.execute("INSERT INTO source_payload VALUES (?,?)", (key, json.dumps({"study_id": row["study_id"], "version": version, "design": option, "legacy_collection": parent["id"], "legacy_group": group["id"]})))
                    created += 1
        db.execute("UPDATE results SET record_kind='collection' WHERE id=?", (parent["id"],))
    from design_study_migration import enrich
    enrich(db, sys.modules[__name__])
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Design registration: design-template --robot-section Knee emits an editable JSON example. "
        "A study is ONE variant, with a stable study_id, version, dashboard=design, robot_sections and one design object. "
        "Each version has a distinct result id. Register CAD, joint manifests, reports, configs and traces in assets; "
        "attach measurements/simulations in design.evidence or a design_evidence section. "
        "Read docs/operations/design-dashboard.md for units, motion, evidence and review identity."))
    parser.add_argument("command", choices=("init", "import-catalog", "publish", "upload", "serve", "design-template", "migrate-design-studies", "archive-study", "restore-study"))
    parser.add_argument("input", nargs="?", help="JSON file for publish")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--port", type=int, default=8805)
    parser.add_argument("--dashboard", choices=("learning", "design"), help="Publication destination; Body/CAD defaults to design")
    parser.add_argument("--robot-section", action="append", help="Explicit robot part for a design study, e.g. Knee (repeat for cross-section work)")
    parser.add_argument("--reason", help="Written reason for archiving a design study")
    args = parser.parse_args()
    if args.command == "design-template":
        print(json.dumps(design_template((args.robot_section or ["Knee"])[0]), indent=2, ensure_ascii=False))
        return
    with connect() as db:
        if args.command in {"archive-study", "restore-study"}:
            if not args.input or args.command == "archive-study" and not args.reason:
                parser.error("Supply the stable study_id, and --reason when archiving")
            with db:
                count = set_study_archived(db, args.input, args.reason if args.command == "archive-study" else None)
            print(f"{'Archived' if args.command == 'archive-study' else 'Restored'} {args.input}: {count} retained versions")
        elif args.command == "migrate-design-studies":
            created = migrate_design_studies(db)
            totals = db.execute("SELECT COUNT(*),COUNT(DISTINCT study_id) FROM results WHERE dashboard='design' AND record_kind!='collection'").fetchone()
            print(f"Registered {created} catalog options; design inventory: {totals[1]} single-variant studies, {totals[0]} retained versions. Original comparisons remain in Archive.")
        elif args.command == "import-catalog":
            import_catalog(db)
            print("Imported curated results and clip inventory")
        elif args.command == "publish":
            if not args.input:
                parser.error("publish requires a JSON file or '-' for stdin")
            value = json.loads(sys.stdin.read() if args.input == "-" else Path(args.input).read_text())
            if args.dashboard:
                value["dashboard"] = args.dashboard
            if args.robot_section:
                value["robot_sections"] = args.robot_section
            with db:
                save_result(db, value, value.get("clips", []))
            prefix = "/design/" if dashboard_for(value) == "design" else "/"
            print(f"Published {value['id']}: http://127.0.0.1:8805{prefix}?result={value['id']}")
        elif args.command == "upload":
            upload(db, args.limit)
        elif args.command == "serve":
            print(f"Results: http://127.0.0.1:{args.port}/", flush=True)
            print(f"Design: http://127.0.0.1:{args.port}/design/", flush=True)
            ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
