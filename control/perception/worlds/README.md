# iPhone video worlds

For the next owner recording, follow the [phone capture procedure](capture-guide.md).
The owner finds both old MOV reconstructions unsatisfactory. The first video
is retained comparison evidence. The current sensor-ZIP reconstruction is
[New phone scan · denser room, artifacts remain](http://127.0.0.1:8805/?result=spectacular-room-world002-20261027).
It uses only `3dgs_recording001.zip`:246 modeled views,211 fitted/35 heldout,
30000 updates and2.37M Gaussians. It improves much of the room over the53-view
sparse extraction, but turns, thin furniture and the late return remain weak.
Actual RGB consumer evidence includes two visible-chair misses and a candidate
question after four heldout relook views. It does not establish localization,
confirmed identity or metric navigation. Native evidence is under
`artifacts/spectacular-room-world002`; the sparse diagnostic is `world001`.

For a Spectacular sensor ZIP, audit the complete export before any splat fit:

```sh
.venv-learning/bin/python control/perception/worlds/audit_sensor_capture.py \
  --zip data/videos-for-3DGS/3dgs_recording001.zip \
  --output artifacts/spectacular-room-pilot001 --review-rotate-clockwise 90
```

This decodes the full video, checks sensor/frame timestamps, measures sampled
image quality and feature overlap, and creates an upright review sheet without
rotating the native data/calibration. It does not infer poses or train a world.
The original ZIP is the retained source; extracted payload copies regenerate
with the same command. `regeneration.json` verifies their hashes.

`sensor-runtime.Dockerfile` provides a local Linux x86-64 Spectacular SDK1.54.0
CPU runtime. On this Mac it uses Docker emulation, so full-resolution offline
replay is slow. `prepare_tracking_preview.py` creates a960px preview, preserves
sensor times/frame numbers and consistently scales all pixel intrinsics; its
frame count and byte-identical regeneration are recorded separately. The
`replay_sensor_capture.py --fast` option follows the upstream CLI fast preset;
it is a preliminary trajectory estimate, not final reconstruction qualification.
`replay-run.json` records the exact offline Docker command, resource limits and
outcome; `trajectory/configuration.json` records SDK/input/source identities.
The container must have networking disabled and a bounded foreground deadline.
Keep complete native pose outputs, final keyframes, sparse map and limitations.

Recreate the removed local runtime with:

```sh
docker build --platform linux/amd64 \
  -f control/perception/worlds/sensor-runtime.Dockerfile \
  -t rl-spectacular-audit:1.54.0-temp control/perception/worlds
```

After replay, assess and publish a written capture review in the shared dashboard:

```sh
.venv-learning/bin/python control/perception/worlds/assess_sensor_trajectory.py \
  --output artifacts/spectacular-room-pilot001
.venv/bin/python control/perception/worlds/publish_sensor_capture.py \
  --output artifacts/spectacular-room-pilot001 --id spectacular-room-pilot001-20261026
```

Publishing requires a human-authored `assessment.json`. A TRACKING status or
feature overlap alone does not establish accurate scale, a usable Gaussian
world, collision geometry or perception-task integration. No Runpod request
is needed for these local capture checks.

Work locally in `robot-learning-lab`. This pipeline reconstructs photographs and
camera motion for perception and relocalization. A monocular video supplies no
measured scale, floor alignment, collision surfaces or robot dynamics.

The historical first-video input is `data/videos-for-3DGS/IMG_3915.MOV`. Keep that
original recording. Its evidence folder is `artifacts/iphone-room-3915-v1`.
The shared learning dashboard is the presentation surface; no per-run website is
generated.

The owner clarified that this environment is limited to the recorded route and
viewing directions, and that the camera was roughly 0.50 m above ground. A
whole-flat capture or unrestricted roaming is outside this scope. Camera height
is an approximate owner report; it does not establish floor alignment.

`calibration-inputs.json` records the owner's nominal 0.80 m clear opening for
the left kitchen doorway. `audit_doorway.py` reproduces the annotated multiview
scale diagnostic:

```sh
.venv-learning/bin/python control/perception/worlds/audit_doorway.py
```

The three adjacent turning views give substantially different pairwise widths
(1.39–9.30 reconstruction units), so their endpoint triangulation does not
support adopting a scale. The nominal width remains available for later fitting;
no measured scale or metric pose is claimed. Recorded-view RGB perception and
localization in reconstruction coordinates can continue without metric scale.

## Reproduce preprocessing

The local Python environment used `pycolmap==3.13.0`,
`opencv-python-headless==4.13.0.92`, `imageio-ffmpeg==0.6.0`, NumPy and Pillow.
`inputs.json` freezes the actual source hash, FFmpeg version, tone mapping,
presentation timestamps, frame hashes and contiguous withheld blocks.

```sh
.venv/bin/python control/perception/worlds/prepare.py --exhaustive
.venv/bin/python control/perception/worlds/package.py
```

`prepare.py --frames-only` cheaply recreates source JPEGs from the original MOV
without rerunning the native camera solve. A fresh `--output` directory makes a
separate reconstruction. Do not overwrite a retained native solve to recreate
images. The first sequential attempt fragmented; exhaustive matches connected
revisited room views. Both native attempts and their limitations are retained.

The sparse map is built from training frames only. Withheld frame poses come
from SIFT-to-map correspondences and fixed-calibration PnP. Before/after temporal
subsets supply a consistency comparison; they share a scene and landmarks, and
are not independent pose ground truth. Eight matches during a fast turn are
weak evidence even though an estimator returns a pose. Small inlier residuals
do not establish reliable global localization. Query matching here considers
all mapped reference images; this is offline relocalization, not a timed online
retrieval/localization service.

`dataset/scene.npz` and `camera-path.npz` use OpenCV camera axes: x right, y down,
z forward. `c2ws` transform camera coordinates into the arbitrary COLMAP world.
The camera calibration is estimated from stabilized video. Rendered depth would
be the learned scene's expected depth in reconstruction units, not sensor depth.

## Historical movie-only GPU fit

These Flash commands document the earlier MOV workflow. For the new Spectacular
recording use the bounded queue controller below; the owner requires avoiding
background provider monitoring and excessive polling.

```sh
./tools/runpod-flash dev --port 8903 --no-reload
.venv-runpod/bin/python control/perception/worlds/submit.py --url http://127.0.0.1:8903
```

Read the actual Flash URL if its port changes. The worker permits one RTX 4090,
zero minimum workers, a five-second idle timeout and a 40-minute provider
deadline. Training itself has a shorter subprocess deadline. Temporary transfer
uses narrowly scoped signed URLs in the existing private Infomaniak bucket;
archive credentials stay on the Mac. Returned ZIP bytes and every extracted
file are hashed. Stop Flash and verify provider cleanup after downloading.

The worker freezes Python 3.10, Torch 2.4.1/cu124 and the upstream gsplat 1.5.3
binary. `train.py` follows the parameterization, Adam rates, densification and
L1/SSIM weighting of the upstream Apache-2.0
[simple trainer](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py).
It fits a single model, with 12,000 optimizer updates and intermediate heldout
appearance measurements. It does not train a robot policy. Native logs,
checkpoint, PLY, camera path, per-frame metrics and exact source hashes are the
evidence. This wrapper uses pytorch-msssim rather than upstream's fused SSIM
kernel. cuDNN TF32 must remain disabled for that loss on the NHWC-derived tensor
layout: `precision-layout-probe` reproduced a 2.7e8-fold gradient amplification
on one identical failed-model render; IEEE float32 agreed with float64 within
2.1e-6 SSIM across the three tested renders. The trainer records backend flags
and logs sampled frame names, loss and rendered value ranges every 100 updates.
This numerical diagnosis does not establish reconstruction quality.

The browser uses GaussianSplats3D 0.4.7 at SH degree 2; CUDA comparison
renders use the trained model's degree 3.

`world.json` identifies the selected model and exact fit counts. The original
12k bounded fit had median withheld PSNR 19.42 dB; continuous pruning degraded
it to 14.70 dB, and a 30k extended-refinement comparison degraded it further to
7.76 dB with invalid legacy SSIM values in the training trace. These failures
remain retained. A precision-corrected 30k fit uses the same data, poses, seed,
3M growth budget and refinement configuration as that last comparison.
`train.py` still defaults to the historical 12k bounded configuration; use
explicit flags for controlled comparisons. These are Gaussian fits, with zero
robot policy updates or simulator transitions.

The selected precision-corrected first fit has median withheld PSNR 20.85 dB;
its initial view improves from 9.96 to 24.48 dB and loses the severe purple haze.
Floor/doorway tearing remains. All 24 native evaluation pairs and all 227 modeled
route frames were reviewed; off-route movement remains substantially weaker.

The longer `IMG_3921.MOV` capture has a separate evidence folder
`artifacts/iphone-room-3921-v1`. Its second-only camera solve fragmented. The
retained joint solve uses first-session geometric anchors with a separate second
camera calibration, while the fitted RGB and initial colors use the second
recording only. `joint-coverage.json` records every missing interval; neither
camera coverage nor lighting consistency is inferred from video duration.
Only 257/469 training frames and 6/12 withheld query poses connect in its selected
map. Precision correction improves six-query median PSNR from 10.51 to 18.66 dB,
but the first 17 modeled frames and several turns remain severely occluded.
The owner finds both reconstructions unsatisfactory; the earlier relative
preference does not establish a useful improved world. A 2,846-splat camera-clearance
comparison made negligible median difference and was not selected.

For a future sensor capture, the skill's reconstructed-world reference links
the official Spectacular Rec recording format. Check a short complete export
(video, IMU, calibration and metadata) before requesting a full capture. A
shared start point alone is weaker than overlapping textured views along a
connected route. The existing MOVs contain no synchronized IMU export.

## Exercise an existing perception consumer

The room was subsequently exercised through the native-furniture experiment's
RGB-only mode. Set `NODE_PATH` to the directory containing your installed
`playwright` package (for example the `node_modules` directory returned when
locating its local `package.json`). The adapter defaults to the local Chrome
app; `CHROME_EXECUTABLE` overrides that path. Keep the shared dashboard running
so it can serve the private model bytes.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-learning/bin/python control/perception/worlds/run_native_object_probe.py \
  --rgb-world artifacts/iphone-room-3915-v1/world.json \
  --output artifacts/NEW-RGB-CONSUMER-RUN
```

This uses unchanged cached Grounding DINO Tiny and DINOv2 Small weights. The
selected first world's fixture compares six source/model pairs, follows a
supplied recorded-view sequence and checks missing-camera/metric-routing
behavior. `artifacts/iphone-room-3915-v1/consumer/report.json` retains all 17
observations: five of six paired question decisions agree; the legs-only render
remains weak, and the sequence asks about a candidate on its fourth image.
This is candidate resemblance, not confirmed identity or autonomous pose planning.
The longer capture's 18-observation run reaches a candidate only on its teaching
image, so that sequence supplies no independent find evidence.

Historical baseline evidence at `iphone-room-perception-20261026` retains the
five-view same-center and six-view shifted-center episodes. Its report refers
to the original PLY, not today's selected model. The branch still lacks measured
depth/range, scale, floor frame and collisions for the full metric route.
Existing Open3D rendering returned black images in the tested local Metal
runtime; `diagnose_open3d.py` reproduces that evidence.

`publish_perception.py` rebuilds the review image and publishes native evidence
for this consumer run; it writes no HTML page. The observation adapter exposes
camera transforms only to the renderer/evaluator and keeps metric capabilities
unavailable to the actor.

## Retention

Keep the original MOV, COLMAP database and solved maps, selected checkpoint and
PLY, complete localization records, native failures, source/config hashes and
selected scene videos. Keep the final scene active locally and archive selected
large native evidence with the existing blob-store workflow. Active inputs can
stay materialized while the scene is being improved.

Unused extracted JPEGs, duplicate transport ZIPs and temporary remote transport
objects are disposable. Verify frame regeneration against `inputs.json` before
deleting unused frame expansions; record the recipe and hashes. Remove only
this run's explicit temporary object keys, after checking the returned local
archive. `cleanup.py` records the precise selection and verification. Never
delete other runs to make room on Runpod's temporary volume.

Recreate a deleted GPU transport without rerunning PnP or changing scene inputs:

```sh
.venv/bin/python control/perception/worlds/package.py \
  --output artifacts/iphone-room-3915-v1 --transport-only
```

`probe_dataset.py` recreates the numerical diagnostic transport from its retained
failed checkpoint, scene, three source images and `probe-input.json`; the exact
command and member hashes are in `precision-probe-attempt1/dataset-regeneration.json`.

Optional map text tables can be recreated with
`.venv/bin/python control/perception/worlds/map_text.py`.
Their numeric round trip was checked before removing unused text expansions;
the exact native binary solves remain retained. `archive-verification.json`
records all 77 selected native files in the private blob index. No active scene
or original video was evicted. The original recording's artifact path is a hard
link so it can be registered in the dashboard and private archive without a
second physical local copy; `source-restore.json` records recovery.

Dashboard names and the written recommendation live in each capture's retained
`display.json`. Its `model` identifies the selected fit; older attempts remain
available in a collapsed list. When changing the selected fit, update this
assessment as well. Republish labels with the existing immutable video IDs:

```sh
.venv/bin/python control/perception/worlds/publish.py --clip-version precision-v4
.venv/bin/python control/perception/worlds/publish.py \
  --output artifacts/iphone-room-3921-v1 \
  --record-id iphone-room-3921-20261026 --clip-version precision-v2
```

## New Spectacular reconstruction (recording001)

The owner authorized a fresh GPU fit from `3dgs_recording001.zip`, and requires
sparse foreground Runpod requests. The active fit uses `spectacular_pod.py`:
one standard GPU pod, direct SSH output, a remaining foreground wall budget,
and termination on completion/deadline. `execution-plan.json` records its
actual GPU, price and limits; this is not a provider-side expiry. Setup failures
may be retained for immediate foreground repair with `--keep-on-error`; the
same pod can then be adopted using its saved verified SSH mapping. Do not
leave a retained repair pod unattended.

The explicit task public key is supplied through `PUBLIC_KEY`; the private key
stays local. In this recorded CLI3.0.1 attempt, RESTv2 `ssh info` incorrectly
reported initializing while container logs showed SSH ready. One direct
GraphQL `runtime.ports` read supplied the working TCP mapping. The controller
uses at most two spaced connection reads and no status polling loop. SSH did
not inherit the image's pip override, so the disposable uv bootstrap permits
the system-package install, then uses an isolated Python3.10 environment.

The earlier queued attempt never reached its handler and was deleted with its
template. Its exact configuration and startup observations are retained.
`spectacular_control.py` implements that original queue lane and the shared
storage transfer/verification actions, with no Flash development process:

```sh
.venv-runpod/bin/python control/perception/worlds/spectacular_control.py prepare
.venv-runpod/bin/python control/perception/worlds/spectacular_control.py launch
```

`prepare` needs the pinned SDK wheel and identity in the output directory;
`plan.json` freezes its complete inputs/configuration. Do not repeat either
command for an existing submitted job. `refresh` updates changed small scripts
before launch without re-uploading the original recording. The worker uses the
official PyTorch image and an isolated Python3.10 environment for SDK1.54.0,
Torch2.4.1/cu124 and gsplat1.5.3. Its provider execution timeout is3300s, queue
TTL4500s, minimum workers0, maximum1, idle timeout5s. Template/endpoint/job IDs
are saved after each operation in `control.json`.

The worker pushes stage progress and the final native ZIP/receipt into scoped
private storage. `progress` and `download` read that storage, making no Runpod
requests. Use them at spaced foreground checkpoints, never inside a loop.
`status` makes exactly one explicit provider read if storage progress cannot
resolve whether the job is queued. Any429 records a stop marker and prevents
further provider calls; there are no automatic rate-limit retries. `cleanup`
deletes only the saved task-owned endpoint and template after evidence recovery.

The SDK export retains full native poses and sparse observations. It caches
keyframe pixels on mapping updates because final-map image buffers are not
reliable. Raw stored pixels are sideways in this specific recording. Training
rotates them90deg clockwise and rotates the OpenCV camera axes and intrinsic
matrix consistently, then resizes to720x960. Native raw poses/K/feature pixels
remain in the retained solve. Train colors use training keyframes only; the
joint sensor pose solve sees the full capture, so interleaved RGB holdouts are
pose-conditioned appearance evidence, not independent localization accuracy.
Handheld height may have changed; estimated height variation is not proven drift.

After reviewing the native render pairs and complete modeled route, write a
qualitative `assessment.json`, then publish with `publish_spectacular.py`. This
creates `world.json` and the saved interactive camera without opening a tab.
The RGB consumer uses a480x640 viewport and aspect-preserving resizing for this portrait capture,
with matching intrinsics and masks outside the captured field of view. Legacy
worlds retain their original640x360 resize. Select new teaching/paired/relook
fixtures from this capture's actual images; do not inherit old movie frame IDs.
Retain the full consumer report and episodes, and match the PLY hash before
claiming integration. SDK estimated metres and Z-up are not surveyed scale,
floor height, measured depth or collision geometry.
