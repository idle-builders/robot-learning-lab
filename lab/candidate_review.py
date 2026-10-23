"""Join retained local evidence without promoting it to whole-product validation."""

import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from typing import Literal

from pydantic import Field, model_validator
import yaml

from .artifact_store import snapshot_file, retain_dependency, verify_dependency
from .artifact_paths import checked_path
from .config import Strict, load_config


class Evidence(Strict):
    report: str
    binding: Literal["candidate", "shared", "reference"]
    configuration: str | None = None
    scope: str
    artifacts: list[str] = Field(default_factory=list)
    # Keys are the producer report's source_hashes keys; values are actual inputs.
    dependencies: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def configuration_required(self):
        if self.binding != "shared" and not self.configuration:
            raise ValueError(
                "candidate/reference evidence needs a retained configuration"
            )
        if self.binding == "shared" and self.configuration:
            raise ValueError("shared evidence must not claim a robot configuration")
        return self


class Review(Strict):
    schema_version: Literal[1]
    name: str
    variant: str
    evidence: dict[str, Evidence] = Field(min_length=1)
    open_work: list[str] = Field(min_length=1)


def differences(expected, actual, prefix=""):
    """Exact JSON value comparison, reporting fields instead of a vague mismatch."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                result.append(path)
            else:
                result.extend(differences(expected[key], actual[key], path))
        return result
    return [] if expected == actual else [prefix]


def run(specification, output, root=None):
    root = (root or Path.cwd()).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = {}
    dependency_manifests = {}

    def path(relative):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"input escapes workspace: {relative}")
        p = (root / relative).absolute()
        if not p.is_relative_to(root):
            raise ValueError(f"input escapes workspace: {relative}")
        if p.is_file():
            return checked_path(root, relative, repo_root=root)
        return p

    def digest(p):
        p = p.resolve()
        if p not in cache:
            with p.open("rb") as stream:
                cache[p] = hashlib.file_digest(stream, "sha256").hexdigest()
        return cache[p]

    def retain(p):
        relative = p.relative_to(root)
        if relative.parts[0] == "artifacts" and "source" in relative.parts:
            # The producer already retained this version. Keep its object manifest,
            # not another source/artifacts/.../source directory chain.
            key = str(p.parent.relative_to(root))
            if key not in dependency_manifests:
                dependency_manifests[key] = retain_dependency(p.parent, repo_root=root)
            return digest(p)
        dest = output / "source" / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(p, dest, repo_root=root)
        return digest(p)

    spec_path = path(str(specification))
    spec = Review.model_validate(yaml.safe_load(spec_path.read_text()))
    config = json.loads(
        load_config(path(spec.variant), path("robot/base.yaml")).model_dump_json()
    )
    (output / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    inputs = {str(spec_path.relative_to(root)): retain(spec_path)}
    for rel in (
        spec.variant,
        "robot/base.yaml",
        "lab/candidate_review.py",
        "lab/config.py",
    ):
        p = path(rel)
        if p.exists():
            inputs[rel] = retain(p)
    report = {
        "schema_version": 1,
        "name": spec.name,
        "variant": spec.variant,
        "scope": "Retained evidence integrity, exact configuration and declared dependency checks; no new engineering simulations",
        "product_readiness": "not_assessed",
        "resolved_config_sha256": digest(output / "resolved-config.json"),
        "evidence": {},
        "open_work": spec.open_work,
        "inputs": inputs,
        "dependency_manifests": dependency_manifests,
    }
    for name, entry in spec.evidence.items():
        item = entry.model_dump()
        item.update(integrity="pass", compatibility="not_applicable", errors=[])
        report["evidence"][name] = item
        try:
            p = path(entry.report)
            inputs[entry.report] = retain(p)
            source = json.loads(p.read_text())
            for ref in source.get("dependency_manifests", {}).values():
                if verify_dependency(ref, repo_root=root)["status"] != "pass":
                    raise ValueError("corrupted retained dependency")
            status = source.get("status")
            if status not in ("pass", "fail", "partial", "not_run"):
                raise ValueError(f"unsupported or missing source status: {status!r}")
            item["result"] = status
            item["reported_scope"] = source.get("scope", source.get("status_scope"))
            item["report_link"] = os.path.relpath(p, output)
            hashes = source.get("source_hashes", {})
            if entry.configuration:
                cfg_path = path(entry.configuration)
                inputs[entry.configuration] = retain(cfg_path)
                item["configuration_attestation"] = "not_recorded_by_producer"
                if (
                    cfg_path == p.parent / "resolved-config.json"
                    and "resolved_config_sha256" in source
                ):
                    if digest(cfg_path) != source["resolved_config_sha256"]:
                        raise ValueError(
                            "resolved configuration differs from producer digest"
                        )
                    item["configuration_attestation"] = "verified"
                if cfg_path != p.parent / "resolved-config.json":
                    snapshot_root = (p.parent / "source").absolute()
                    if not cfg_path.is_relative_to(snapshot_root):
                        raise ValueError(
                            "configuration must belong to report or its retained source snapshot"
                        )
                    key = str(cfg_path.relative_to(snapshot_root))
                    if digest(cfg_path) != hashes.get(key):
                        raise ValueError(
                            "configuration is not attested by source report"
                        )
                    item["configuration_attestation"] = "verified"
                changed = differences(config, json.loads(cfg_path.read_text()))
                item["configuration_differences"] = changed
                item["compatibility"] = (
                    "reference_only"
                    if entry.binding == "reference"
                    else "mismatch"
                    if changed
                    else "match"
                )
            if not hashes:
                raise ValueError("source report has no retained source hashes")
            drift = []
            for key, expected in hashes.items():
                # Snapshot keys must also stay inside the artifact's source folder.
                snapshot_root = p.parent / "source"
                snapshot = checked_path(snapshot_root, key, repo_root=root)
                if digest(snapshot) != expected:
                    raise ValueError(f"corrupted source snapshot: {key}")
                current = path(key)
                if current.is_file() and digest(current) != expected:
                    drift.append(key)
            item["verified_source_snapshots"] = len(hashes)
            item["changed_workspace_sources"] = drift
            item["dependency_checks"] = {}
            for key, relative in entry.dependencies.items():
                actual = digest(path(relative))
                inputs[relative] = actual
                if key not in hashes or actual != hashes[key]:
                    raise ValueError(
                        f"dependency differs from consumed input: {key} -> {relative}"
                    )
                item["dependency_checks"][key] = {"path": relative, "sha256": actual}
            item["artifacts_checked"] = {}
            for relative in entry.artifacts:
                artifact = path(relative)
                actual = digest(artifact)
                inputs[relative] = actual
                item["artifacts_checked"][relative] = {
                    "sha256": actual,
                    "bytes": artifact.stat().st_size,
                    "link": os.path.relpath(artifact, output),
                }
        except (ValueError, OSError, TypeError, KeyError) as exc:
            item["integrity"] = "fail"
            item["errors"].append(str(exc))
    items = report["evidence"].values()
    report["consistency"] = (
        "fail"
        if any(
            e["integrity"] == "fail" or e["compatibility"] == "mismatch" for e in items
        )
        else "pass"
    )
    # A reference is never promoted, and source failures remain visible even when
    # the original suite contains an intentional negative-control failure.
    report["status"] = (
        "fail"
        if report["consistency"] == "fail"
        or any(e.get("result") == "fail" for e in items)
        else "partial"
    )
    report["status_scope"] = (
        "Bundle evidence only; open work prevents any whole-product pass"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_review(report, output)
    return report


def write_review(report, output):
    esc = html.escape
    rows = []
    for name, item in report["evidence"].items():
        links = " ".join(
            f'<a href="{esc(v["link"], quote=True)}">{esc(Path(k).name)}</a>'
            for k, v in item.get("artifacts_checked", {}).items()
        )
        details = {
            k: item[k]
            for k in (
                "configuration_differences",
                "configuration_attestation",
                "verified_source_snapshots",
                "changed_workspace_sources",
                "dependency_checks",
                "errors",
            )
            if k in item
        }
        rows.append(f'''<tr><td><b>{esc(name)}</b><p>{esc(item["scope"])}</p></td>
<td>{esc(item.get("result", "unreadable"))}<br>{esc(item["binding"])}</td>
<td>{esc(item["compatibility"])}<br>Integrity: {esc(item["integrity"])}</td>
<td><a href="{esc(item.get("report_link", "#"), quote=True)}">Source report</a> {links}
<details><summary>Configuration, provenance and dependencies</summary><pre>{esc(json.dumps(details, indent=2))}</pre></details></td></tr>''')
    gaps = "".join(f"<li>{esc(x)}</li>" for x in report["open_work"])
    (
        output / "review.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{esc(report["name"])}</title><style>
body{{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#172c32;background:#f4f2eb}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:14px;border-bottom:1px solid #bdc9c5;vertical-align:top}}
td:first-child{{width:30%}}pre{{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}}a{{color:#007966}}
</style><h1>{esc(report["name"])}</h1>
<p>Evidence result: <b>{report["status"]}</b> · Configuration/dependency consistency: <b>{report["consistency"]}</b></p>
<p>{esc(report["scope"])}. Existing source failures remain failures. Reference evidence does not qualify this candidate.
Source drift is listed for review; matching configuration alone does not prove equivalent code or dynamics.</p>
<p><a href="report.json">Machine-readable review</a> · <a href="resolved-config.json">Candidate parameters</a></p>
<table><thead><tr><th>Experiment and scope</th><th>Source result / binding</th><th>Agreement</th><th>Evidence</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table><h2>Open product work</h2><ul>{gaps}</ul>
<p>Product readiness is not assessed. Output hashes record the files inspected now; source reports do not necessarily attest their original output hashes.
Large engineering artifacts remain in their original local run folders. This review is an index, not a standalone archive.</p></html>""")
