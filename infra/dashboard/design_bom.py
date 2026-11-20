"""Version-specific planning BOMs and explicit robot compositions for results_hub.

Amounts retain their native currency; unknown inputs never become zero-cost parts.
"""
from decimal import Decimal
import json
import math
import re


class BomError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def init_schema(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS design_boms (
      result_id TEXT PRIMARY KEY REFERENCES results(id) ON DELETE CASCADE,
      payload_json TEXT NOT NULL, revision INTEGER NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS robot_bom (
      id INTEGER PRIMARY KEY CHECK(id=1), payload_json TEXT NOT NULL,
      revision INTEGER NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS design_bom_history (
      result_id TEXT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
      revision INTEGER NOT NULL, payload_json TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY(result_id,revision)
    );
    """)


def number(value, label, positive=False):
    if value is None:
        return None
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0 and (not positive or value > 0))
    except OverflowError:
        valid = False
    if not valid:
        raise BomError(f"{label} must be a finite {'positive' if positive else 'nonnegative'} number or null")
    return value


def text(value, label):
    if not isinstance(value, str):
        raise BomError(f"{label} must be text")
    return value.strip()


def strings(value, label):
    if not isinstance(value, list):
        raise BomError(f"{label} must be a list of text")
    return [text(v, label) for v in value]


def validate_bom(bom, assets=None):
    if not isinstance(bom, dict) or not isinstance(bom.get("items"), list):
        raise BomError("BOM needs an items list")
    if bom.get("completeness", "partial") not in {"partial", "complete"}:
        raise BomError("BOM completeness must be partial or complete (author's scope declaration)")
    result = {"scope": text(bom.get("scope", ""), "BOM scope"),
              "completeness": bom.get("completeness", "partial"),
              "notes": text(bom.get("notes", ""), "BOM notes"),
              "exclusions": strings(bom.get("exclusions", []), "BOM exclusions"), "items": []}
    seen = set()
    for entry in bom["items"]:
        if not isinstance(entry, dict):
            raise BomError("Each BOM item must be an object")
        item = {k: text(entry.get(k, ""), k) for k in (
            "id", "name", "part_number", "category", "supplier", "material", "notes", "price_date")}
        if not item["id"] or item["id"] in seen or not item["name"]:
            raise BomError("BOM items need distinct nonempty IDs and names")
        seen.add(item["id"])
        item["unit"] = text(entry.get("unit", "ea"), "unit")
        if not item["unit"]:
            raise BomError("Quantity unit must be nonempty (e.g. ea, m, kg)")
        for key in ("quantity", "unit_cost", "unit_mass_g", "lead_time_days"):
            item[key] = number(entry.get(key), key)
        item["currency"] = text(entry.get("currency", ""), "currency").upper()
        if item["currency"] and not re.fullmatch(r"[A-Z]{3}", item["currency"]):
            raise BomError("Use a three-letter currency code")
        if item["unit_cost"] is not None and not item["currency"]:
            raise BomError("Every known unit cost needs its currency")
        if item["price_date"]:
            from datetime import date
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item["price_date"]):
                    raise ValueError("Date format")
                date.fromisoformat(item["price_date"])
            except ValueError as exc:
                raise BomError("Price date must be YYYY-MM-DD") from exc
        item["cost_basis"] = entry.get("cost_basis", "unknown")
        if item["cost_basis"] not in {"unknown", "estimate", "quote", "paid"}:
            raise BomError("cost_basis must be unknown, estimate, quote or paid")
        item["source_refs"] = []
        refs = entry.get("source_refs", [])
        if not isinstance(refs, list):
            raise BomError("source_refs must be a list")
        for ref in refs:
            if not isinstance(ref, dict):
                raise BomError("Each source reference must be an object")
            path, url = ref.get("path"), ref.get("url")
            if not path and not url:
                raise BomError("Source references need a path or URL")
            if path and (not isinstance(path, str) or assets is not None and path not in assets):
                raise BomError(f"Register BOM source in this version's assets: {path}")
            if url and (not isinstance(url, str) or not re.match(r"^https?://[^/\s]+", url)):
                raise BomError("Source URLs must use http or https")
            clean = {"label": text(ref.get("label", path or url), "source label")}
            if path:
                clean["path"] = path
            if url:
                clean["url"] = url
            item["source_refs"].append(clean)
        result["items"].append(item)
    return result


def study(db, result_id):
    row = db.execute("SELECT id,title,study_id,study_version,robot_sections_json FROM results WHERE id=? AND dashboard='design' AND record_kind='study'", (result_id,)).fetchone()
    if row is None:
        raise BomError("Choose a published design study version", 404)
    result = dict(row)
    result["robot_sections"] = json.loads(result.pop("robot_sections_json") or "[]")
    return result


def get_bom(db, result_id, revision=None):
    identity = study(db, result_id)
    if revision is None:
        row = db.execute("SELECT * FROM design_boms WHERE result_id=?", (result_id,)).fetchone()
    else:
        row = db.execute("SELECT * FROM design_bom_history WHERE result_id=? AND revision=?", (result_id, revision)).fetchone()
        if row is None:
            raise BomError("Unknown BOM revision", 404)
    return {**identity, "bom": json.loads(row["payload_json"]) if row else None,
            "revision": row["revision"] if row else 0, "updated_at": row["updated_at"] if row else None}


def put_bom(db, result_id, bom, expected_revision=None):
    current = get_bom(db, result_id)
    if expected_revision is not None and (type(expected_revision) is not int or current["revision"] != expected_revision):
        raise BomError("BOM changed while you were editing. Reload the saved BOM before saving again.", 409)
    assets = {r[0] for r in db.execute("SELECT path FROM assets WHERE result_id=?", (result_id,))}
    clean = validate_bom(bom, assets)
    totals([extended(item) for item in clean["items"]])
    if current["revision"]:
        db.execute("INSERT OR IGNORE INTO design_bom_history(result_id,revision,payload_json,updated_at) VALUES (?,?,?,?)",
                   (result_id, current["revision"], json.dumps(current["bom"], ensure_ascii=False), current["updated_at"]))
    db.execute("""INSERT INTO design_boms(result_id,payload_json,revision) VALUES (?,?,?)
        ON CONFLICT(result_id) DO UPDATE SET payload_json=excluded.payload_json,
        revision=excluded.revision,updated_at=CURRENT_TIMESTAMP""",
        (result_id, json.dumps(clean, ensure_ascii=False), current["revision"] + 1))
    db.execute("INSERT INTO design_bom_history(result_id,revision,payload_json) VALUES (?,?,?)",
               (result_id, current["revision"] + 1, json.dumps(clean, ensure_ascii=False)))
    return get_bom(db, result_id)


def finite_amount(value):
    result = float(value)
    if not math.isfinite(result):
        raise BomError("BOM amount is too large to represent")
    return result


def extended(item, assemblies=1):
    def mul(*values):
        if any(v is None for v in values):
            return None
        product = Decimal(1)
        for value in values:
            product *= Decimal(str(value))
        return finite_amount(product)
    quantity = mul(item.get("quantity"), assemblies)
    return {**item, "quantity": quantity,
            "extended_cost": mul(quantity, item.get("unit_cost")),
            "extended_mass_g": mul(quantity, item.get("unit_mass_g"))}


def totals(lines):
    costs = {}
    for line in lines:
        if line["extended_cost"] is not None:
            currency = line["currency"]
            costs[currency] = costs.get(currency, Decimal(0)) + Decimal(str(line["extended_cost"]))
    return {"known_cost_by_currency": {k: finite_amount(v) for k, v in sorted(costs.items())},
            "known_mass_g": finite_amount(sum((Decimal(str(l["extended_mass_g"])) for l in lines if l["extended_mass_g"] is not None), Decimal(0))),
            "line_count": len(lines),
            "missing_quantity_lines": sum(l["quantity"] is None for l in lines),
            "missing_cost_lines": sum(l["extended_cost"] is None for l in lines),
            "missing_mass_lines": sum(l["extended_mass_g"] is None for l in lines),
            "unreferenced_lines": sum(not l.get("source_refs") for l in lines)}


def study_report(db, result_id, revision=None):
    result = get_bom(db, result_id, revision)
    result["lines"] = [{**extended(item), "result_id": result_id, "study_id": result["study_id"],
                        "study_version": result["study_version"], "bom_revision": result["revision"],
                        "assembly": result["title"], "assembly_count": 1}
                       for item in (result["bom"] or {}).get("items", [])]
    result["totals"] = totals(result["lines"])
    return result


def validate_plan(db, value):
    if not isinstance(value, dict) or not isinstance(value.get("assemblies"), list):
        raise BomError("Robot BOM needs an assemblies list")
    clean = {"title": text(value.get("title", "Robot planning BOM"), "title"),
             "notes": text(value.get("notes", ""), "notes"),
             "exclusions": strings(value.get("exclusions", []), "exclusions"), "assemblies": []}
    seen, studies = set(), set()
    for assembly in value["assemblies"]:
        if not isinstance(assembly, dict) or not isinstance(assembly.get("result_id"), str):
            raise BomError("Each assembly needs a result_id")
        identity = study(db, assembly["result_id"])
        if identity["id"] in seen or identity["study_id"] in studies:
            raise BomError("Include a study once, at one version; use its count for repeated assemblies")
        count = number(assembly.get("count"), "Assembly count", positive=True)
        if count is None or int(count) != count:
            raise BomError("Assembly count must be a positive integer")
        seen.add(identity["id"])
        studies.add(identity["study_id"])
        clean["assemblies"].append({"result_id": identity["id"], "count": int(count)})
    return clean


def get_plan(db):
    row = db.execute("SELECT * FROM robot_bom WHERE id=1").fetchone()
    return {"plan": json.loads(row["payload_json"]) if row else {
        "title": "Robot planning BOM", "notes": "", "exclusions": [], "assemblies": []},
        "revision": row["revision"] if row else 0, "updated_at": row["updated_at"] if row else None}


def put_plan(db, plan, expected_revision):
    if type(expected_revision) is not int or expected_revision != get_plan(db)["revision"]:
        raise BomError("Robot BOM changed while you were editing. Reload before saving again.", 409)
    clean = validate_plan(db, plan)
    db.execute("""INSERT INTO robot_bom(id,payload_json,revision) VALUES (1,?,?)
        ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json,
        revision=excluded.revision,updated_at=CURRENT_TIMESTAMP""",
        (json.dumps(clean, ensure_ascii=False), expected_revision + 1))


def robot_report(db):
    result = get_plan(db)
    assemblies, lines, coverage = [], [], {}
    for assembly in result["plan"]["assemblies"]:
        report = study_report(db, assembly["result_id"])
        assemblies.append({**assembly, **{k: report[k] for k in (
            "title", "study_id", "study_version", "robot_sections", "bom", "revision", "updated_at")}})
        for section in report["robot_sections"]:
            coverage.setdefault(section, []).append(report["title"])
        for item in (report["bom"] or {}).get("items", []):
            lines.append({**extended(item, assembly["count"]), "result_id": report["id"],
                          "study_id": report["study_id"], "study_version": report["study_version"],
                          "assembly": report["title"], "assembly_count": assembly["count"],
                          "bom_revision": report["revision"]})
    sections = {s for row in db.execute("SELECT robot_sections_json FROM results WHERE dashboard='design' AND record_kind='study'")
                for s in json.loads(row[0] or "[]")}
    result.update(assemblies=assemblies, lines=lines, totals=totals(lines),
                  missing_bom_versions=[a["result_id"] for a in assemblies if a["bom"] is None],
                  empty_bom_versions=[a["result_id"] for a in assemblies if a["bom"] is not None and not a["bom"]["items"]],
                  uncovered_sections=sorted(sections - set(coverage)),
                  overlapping_sections={k: v for k, v in coverage.items() if len(v) > 1})
    return result
