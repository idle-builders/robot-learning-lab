# Robot learning dashboard

Start the dashboards with `.venv/bin/python infra/dashboard/results_hub.py serve`.

- **[Policies](http://127.0.0.1:8805/)** (`/`): the control system: the whole-body controller and its analyses, and the skill policies until the controller supersedes them. Each opens its newest run; earlier runs are in the version menu, listed by date and commit.
- **[Perception](http://127.0.0.1:8805/perception)** and **[Planning](http://127.0.0.1:8805/planning)**: perception and planning results (`workspace: "Perception"` / `"Planning"`).
- **[Archive](http://127.0.0.1:8805/archive)** (`/archive`): history. Diagnostics, setup probes and superseded lines, each with the reason it left the current view (`archived_at`, `archive_reason`). See [Archive](#archive) below.
- **[Robot design](http://127.0.0.1:8805/design/)** (`/design/`): CAD-first inspection of the current body, by part.

Every published record carries `source_commit`, the repository commit it was published from. Name the code and configuration a result depends on in `source` (paths); if they have uncommitted changes the commit is marked `+dirty`. Commit before publishing a result you want to reproduce. Design studies take their version from the commit (see the design guide).

For design registration, CAD/motion metadata, measurements and engineering simulations, use the [design dashboard guide](design-dashboard.md) and `infra/dashboard/results_hub.py design-template --robot-section Knee`. Keep short findings and structured measurements in both dashboards; expand technical detail when needed. The mechanical body is the one the body definitions declare (`body:` in `robot/body/complete_body.bom.yaml`); presentation does not qualify a mechanical body or prove physical behavior.

## Publish a new result

Write one small JSON record, then run:

```sh
.venv/bin/python infra/dashboard/results_hub.py publish /path/to/result.json
.venv/bin/python infra/dashboard/results_hub.py upload
```

Use `upload --result-id RESULT_ID` to upload only one result while other work is active.

Pass `-` to `publish` to read the JSON from stdin and avoid saving a separate presentation file.

Example input:

```json
{
  "id": "forward-fresh-20261016",
  "title": "Forward walk, fresh starts",
  "workspace": "Control",
  "dashboard": "learning",
  "area": "Locomotion",
  "status": "experimental",
  "observation": "Alternating steps continued through the smooth starts; rough starts tipped at the right-side transition.",
  "measurements": "24 complete episodes; 8 independent start families; distance and duration distributions in matrix",
  "limitations": "Simulator body and motor model remain provisional.",
  "body": "cove50_description (robot/model/current-model.json)",
  "source": ["control/wbc/"],
  "matrix": [{"condition": "rough", "distance_m_p50": 0.42, "lifetime_s_p50": 1.8}],
  "grades": {"overall": "Workable on smooth starts; rough starts degrade at the right-side transition.",
             "dimensions": [{"dimension": "smooth starts", "grade": 4, "confidence": "medium", "trend": "improving", "reason": "24/24 episodes continue past 1 m"},
                            {"dimension": "rough starts", "grade": 2, "confidence": "medium", "trend": "flat", "reason": "tips after the third touchdown"}],
             "requirement_challenges": ["Is a 2 s horizon the right reference for rough starts?"]},
  "assets": [{"id": "native-report", "title": "Native report", "kind": "data", "path": "artifacts/forward-fresh-20261016/report.json"}],
  "clips": [{
    "id": "rough-tip-01",
    "title": "Rough start, right-side tip",
    "role": "Native original; severe degradation",
    "body": "cove50_description (robot/model/current-model.json)",
    "observation": "Tip began after the third touchdown.",
    "source_path": "artifacts/forward-fresh-20261016/rough-tip-01.mp4",
    "report_path": "artifacts/forward-fresh-20261016/rough-tip-01.json",
    "trace_path": "artifacts/forward-fresh-20261016/rough-tip-01.npz"
  }]
}
```

`grades` is optional but expected for new records. It is the graded assessment from the
[evaluation principles](../design/evaluation-principles.md):
- a written `overall` judgement;
- per-dimension `grade` from 0 (unknown) to 5 (viable), with `confidence` (low, medium, high), `trend`
  (improving, flat, regressing, new) and a `reason`;
- any `requirement_challenges`.

There is no pass/fail field. The viewer shows the grades as bars under the evidence cards.

The publisher hashes each local clip before inserting its row. `upload` writes videos under content-hash keys in the existing private `<archive-bucket>` bucket, downloads each object, checks its full SHA-256, and only then marks its row verified. It resumes safely. On this Mac, the app reads the existing Infomaniak `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` entries from Keychain when needed; environment variables also work. The server issues one-hour private playback URLs. Neither credentials nor signed URLs are stored in the table. The browser requests a URL only when a person presses **Watch**.

Clip IDs are immutable: a changed source video needs a new ID. Text and case sections can be revised on the same result row, while the original clip hash and bucket object stay tied to its episode.

Managed producers may publish clips directly from `.cache/work/<run-id>/`.
Keep the managed producer active through publication and verified private upload,
then release its scratch. Retain the rendering recipe and exact native inputs;
the verified private video remains playable after its local render is removed.

`workspace` accepts an agent area's name. `assets` accepts local image, model, and data files under `artifacts/`, `robot/` or `control/` (older records also name `studies/`, served from Git history); each source is hashed and shown in the shared image, 3D, or text widget. A publish call replaces that record's asset list, so include its full current list when revising. Keep native source evidence in its run folder; use the dashboard for presentation. New evaluators send JSON to `publish -` or call the publisher, and return the stable dashboard URL. They do not write an `index.html` or open a browser tab or Codex panel automatically.

Publishing a revision does not download cold assets. An unchanged asset may retain its identity when the same result, asset ID, path, SHA-256 and byte count are already registered and its exact private archive route belongs to a verified pack. This covers absent ordinary files and the existing direct missing CAS links; new or changed assets require local bytes. Asset viewing restores cold content only for the request.

Keep representative smooth, rough, recovering, and severe clips and describe where behavior degrades. For parallel runs, retain the full-episode source and show native-unit distributions and requested-horizon coverage by condition. Historical frozen gate counts remain provenance. The original native reports, traces, checkpoint bundles, and source hashes are the evidence; the table is a presentation index.

## Archive

**Archive** in [Robot learning](http://127.0.0.1:8805/archive) holds learning and reference runs; **Archive** in [Robot design](http://127.0.0.1:8805/design/) holds retained CAD, engineering and source comparisons. Both browse native files without copying them. Search narrows the retained runs; each run can show saved video, images, a standalone GLB, text evidence (previews capped at 120 KB), and an original interactive page inside the dashboard. The Archive omits folders that already have a published row, so a run appears only once. Port 8804 (`infra/dashboard/session_inspector.py`) is only a compatibility redirect for old `?view=results&skill=...` and `?view=archive&session=...` links and for previously linked local video paths.

## Robot design studies

Mechanical CAD belongs in the [design dashboard](http://127.0.0.1:8805/design/), not the learning table: one-variant studies per body section, versions from Git. See [design registration](design-dashboard.md). Comments and judgments use the conflict-aware `design_reviews` / `design_review_history` tables. Shortlists and section preferences are human choices, not mechanical qualification or whole-robot selection.

## Remote read-only view

The owner's remote host (named in the gitignored `.credentials/remote-dashboard.md`, with its HTTP basic-auth login) serves the current learning and design views, read-only. The login is never committed or kept in Keychain. After one login a device stays signed in through a 180-day session cookie, renewed on every visit; a new password signs every device out (owner, 2026-11-24). The local dashboard stays the only place that writes; selections, reviews, BOM edits, earlier versions and the Archive view are local only.

The remote view is the current state, about fifty files: the body's study (`body.study`, `body:` in `robot/body/complete_body.bom.yaml`) and the other declared studies with their GLB models, and the newest version of each learning skill with one clip. Its web-files bucket (separate from the private archive) is laid out as:

```
site/                                    index (the hub's API answers) and UI, copied into the pod every minute
files/design/<study>/<version>/<file>    GLB models (gltfpack web copies) of the selected studies
files/videos/<result>/<clip>.mp4         one clip per current run, copied inside object storage from the private archive
```

Every `publish`, `upload` and every commit or merge on `main` (`.githooks/post-commit`, `post-merge`) updates it in the background (`web_export.py --refresh`, log in `.cache/web-export/refresh.log`): it uploads only `files/` keys that are new (a key names one study version or clip, so it never changes), mirrors `site/`, then deletes keys nothing refers to. Refreshes run one at a time and requests that arrive meanwhile are folded into one follow-up run. Set `LAB_WEB_REFRESH=0` to skip it, for example during a batch of publishes, and refresh once by hand from the main checkout:

```sh
.venv/bin/python infra/dashboard/web_export.py --upload
```

The serving side lives in the owner's private infrastructure repository: nginx serves `site/` and streams `/files/` from the bucket through a read-only rclone gateway.

## Retention

Before removing a local video, require a `verified_at` row, the exact source SHA-256 and byte count, and a successful browser playback. Before removing a run directory or generated page, check live references from code, docs, the mechanical pointer, and retained source manifests. Keep any sole-copy checkpoint, trace, log, or report needed to reproduce a selected result. Do not prune `.artifact-store/objects` by directory size: its objects are shared by relative links and dependency manifests. Delete unreachable objects only after a reachability inventory.
