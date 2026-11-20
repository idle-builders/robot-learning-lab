# Robot model

`current-model.json` names the current simulation model: the selected body and actuators as MJCF and URDF (`cove50_description`). It is built in stages; each stage's outputs are in `artifacts/robot/model/<stage>/`.

| Stage | What it does |
|---|---|
| `build/` | CAD → native MJCF with sensors, actuators and timing |
| `collisions/` | Collision corrections on aesthetic reference shapes |
| `timestep/` | 0.25 ms numerical resolution of the servo law; the native model the controllers use (`cove50-rigid.xml`) |
| `urdf/` | Portable URDF and ROS package with complete motor-case visuals; `select.py` writes `current-model.json` |
| `dynamics/` | Actuator dynamics sidecar |
| `motor-map/` | Tier-3 motor map (which servo drives which joint) |

`physics-ledger.yaml` records the mass and physics assumptions. The text below is the model's history as built.


`current-model.json` identifies the provisional physics and sensor model for the
selected V24 sensing / R23 compute / R33 power composition and upper V5 / lower
V16 mechanics. This does not replace the historical experimental control body,
change the selected Electronics study, or claim a qualified physical robot.

The current model has 22 actuated axes. Its portable URDF is the rigid reference;
MuJoCo provides both that reference and a separate model with 16 coupled FEM
interfaces. Two additional wrist compliance matrices are exported without
inventing a selected hand mechanism or its mass distribution.

The current URDF visual revision restores all 22 complete motor-case meshes
from selected CAD; the earlier 4,000-face reductions lost most case surfaces.
`current-model.json` points to the corrected URDF and its optional
`shell_hidden_URDF`. The latter hides 45 appearance visuals while retaining
the same 3.8303 kg mass, inertias, collisions and joints. R3 native MJCF physics
and actuator/sensor timing remain unchanged. The portable URDF package and
front/rear render are in `artifacts/robot/model/urdf/`;
verification and the fresh independent review are under
`data/urdf-inspection-r1-20261026/`.

Current artifacts are in `artifacts/robot/model/timestep/`. The downloadable
`cove50-description-r3-20261026.zip` contains the URDF, ROS package description,
SRDF, native MJCF variants, every mesh dependency, parameter sidecars and runtime.
The package inventory seals each file. Supplier documents, original CAD/FEM
sources and native diagnostics remain in the private evidence archive.

R2 corrects hard collision on 44 unfitted, explicitly non-BOM aesthetic reference
shapes. It retains their visuals and conditional mass allowance, and regenerates
affected physical link hulls while preserving represented hardware contacts.
Native CAD confirmed a roughly 111 mm³ overlap between a real battery bridge
and an unfitted thigh reference. The corrected neutral body holds for the full
five-second diagnostic; old walking transfers still fall near one second.
Production shell fit and useful forward walking remain unresolved. See
[`r2-20261026/README.md`](r2-20261026/README.md) for the revision and consumer limits.
The original R1 export and its failed-control observations remain frozen.

R3 retains those physical contracts and corrects the numerical resolution of the
assumed external servo law. At 1 ms, an unloaded knee/ankle developed a growing
alternating mode. An exact-start/target 0.25 ms comparison removed that chatter,
but still fell during load transfer. R3 uses 0.25 ms for rigid physics and 4 kHz
evaluation of the continuous servo assumption, with the same motor gains/caps,
50 Hz commands and delays. This does not identify the real digital servo cadence,
reflected inertia or thermal capability, or establish general convergence. See
[`r3-20261026/README.md`](r3-20261026/README.md).

## Use the model

From this checkout:

```sh
.venv/bin/python - <<'PY'
import mujoco
m = mujoco.MjModel.from_xml_path(
    'artifacts/robot/model/timestep/cove50-rigid.xml')
print(m.nq, m.nv, m.nu)
PY
.venv-control/bin/python robot/model/timestep/mjlab_adapter.py
```

The rigid model has a free root and 22 motors: `nq=29`, `nv=28`, `nu=22`. The
compliant model additionally has 96 passive scalar coordinates: `nq=125`,
`nv=124`, `nu=22`. Bind policy channels by the ordered names in `actuators.json`,
not by the first 22 `qpos` or `qvel` entries.

`Runtime` accepts named joint targets in output radians and applies delayed,
catalog-bounded assumed position control. Its new actor observation has 75
values: 22 encoder positions, 22 velocities, body gyro, an IMU-derived gravity
estimate, three command values, and 22 previous accepted targets. Capture times,
sample ages and health are separate fields. Simulator root pose, velocity and
object truth are not silently included in that actor vector. The MJLab adapter
constructs a new 22-axis entity with the upstream DC actuator approximation;
it does not substitute this body into a sealed 21-axis task.
The MJLab adapter expresses the 20 ms delay as 80 physics lags at 0.25 ms;
pass the actual task `physics_dt_s` when changing resolution. Nominal sensor
rates do not imply identical quantized capture times across physics steps.
Encoder positions use the declared 12-bit quantization; joint velocities are
captured native velocities in the synthetic packet. Actual velocity-register
quantization, filtering and noise are not identified, so this is an optimistic
velocity measurement assumption for conditional learning.

For an extracted portable package, set `COVE50_MODEL_DIR` to its absolute
directory and add its `runtime/` folder to `PYTHONPATH`. Native XML/URDF loading
requires only the package's relative mesh paths. `cove50.ros.urdf` uses
`package://cove50_description/meshes/…`; install the included ROS description
package when using RViz or robot_state_publisher. Use the SRDF collision
exclusions alongside the URDF for planning. An importer that ignores the SRDF
will report intended joint/mount overlaps as self-contact.
MuJoCo's direct URDF import also creates a fixed-base 22-coordinate body with no
motors, rather than the free-root 29/28-coordinate MJCF. A new dynamics importer
must supply the floating base, exclusions, actuator law and timing, then compare
matched native motion. File compilation and articulated FK alone do not prove
equivalent dynamics. Spawn the level neutral torso near Z=302.2 mm for 0.2 mm
clearance above the added TPU pads; derive the height from the current-pose foot
colliders rather than starting in the default CAD pose's floor overlap.

## What is known, and what is estimated

All 853 selected scene entries are retained. Valid native solids supply
homogeneous volume centres and full inertia tensors. Supplier-invalid cases and
the unmeasured internals of motors and electronics use explicit approximations.
The conditional represented mass is **3.6803 kg**. Adding an explicit **150 g**
unfinished-installation allowance gives **3.8303 kg** for the default simulation.
The engineering sensitivity sum is **2.95–4.73 kg**, not a confidence interval.
About 0.814 kg is a conditional reference-skin mass/inertia allowance. Its
unfitted reference surfaces are visual context, not R2 rigid collision. Weighing the actual links
and completed robot remains necessary.

The selected CAD has 20 FEETECH ST-3215-C047 drives and two ROBOTIS XM335-T323-T
neck drives. Supplier curves, gear ratios, currents, encoder resolution and
catalog backlash are retained in `actuators.json`. No installed gains, friction,
reflected rotor inertia, continuous thermal law or bus delay has been identified.
The XM335 runtime uses a symmetric absolute-speed cap from its digitized
positive-quadrant curve; it supplies zero torque beyond the curve's no-load
speed. Braking and overspeed recovery are not identified by that curve. This
approximation is distinct from the C047's directional motoring/braking envelope;
do not interpret simulated overspeed loss of neck authority as measured hardware.
The XM335 supplier inertia document lists 30 g, while its e-manual and the
selected case/horn split use 27 g. That reference tensor is preserved without
silently assigning a whole assembled motor tensor to the case alone. A full
3S charge is 12.6 V; the neck's documented 12 V maximum requires the intended
regulated rail. Remaining power installation is tracked separately.

The FEM variant preserves full 6×6 coupled tangent stiffness through six modal
tendon springs per active interface. Matrices are shifted to the actual model
joint origins and rotated with proper polar/axial conventions. Source ideal
clamps, material assumptions, source mesh limits and elastic breakdown remain
visible. Static FEM does not identify damping: modal damping is explicitly an
assumed sensitivity. Wrist matrices remain sidecars, and replaced spine/pelvis/
head parts do not inherit old stiffness. The connector meshes do not deform;
their receiving interfaces and downstream bodies move through the reduced model.
Receiving case/horn mass and visuals move, while their decomposed contact hulls
remain with the source parent links. This is a small-offset reduced-contact
approximation, unsuitable for connector-gap contact or large flexible motion.
The flexible default uses a 2 microsecond step. A 5 microsecond probe changed
whole-episode tracking error materially. The final 1 and 5 microsecond probes
change RMS tracking by about 14% and 29% against 2 microseconds; numerical
convergence is not established.
Use the rigid model for the initial learning campaign and the flexible branch
for bounded CPU sensitivity experiments.

Camera barrel-axis proxies, the exact IMU board pose, six foot sensor centres,
ADC and microphone port are exported. Camera intrinsics/pixel roll, IMU die
registration, force calibration, clock alignment and installed compute delays
remain unmeasured. Synthetic camera/depth packets identify their calibration and
capture epoch. `metric_point` requires a capture-time pose and caller-declared
age envelope; it leaves coordinates unavailable after missing calibration,
clock changes, missing signal or stale capture. No walking SLAM is claimed.

The native source describes forward as CAD −Y and places source `R` at positive
CAD X. The proper rotation is model `(x,y,z)=(-CAD_Y,CAD_X,CAD_Z)`. Source `R` is
therefore at positive model y. R/L remain source identifiers; no reflection or
silent anatomical rename is applied. `frames.json` gives the explicit contract.

Six hip ranges are unspecified in the selected CAD. Their URDF type is continuous
and their native limit is unspecified, which does **not** establish unrestricted
physical travel. Runtime diagnostics use an explicit ±0.35 rad temporary command
envelope. Electrical signs/zeros remain uncalibrated.

## Evidence and interpretation

Native URDF/MJCF forward kinematics agree within floating-point precision, and
independent MuJoCo coupons reproduce all 18 exported coupled stiffness matrices.
V-HACD collision decomposition improves on a single filled torso hull, but still
has voxel/thin-wall approximation. R2 removes the source-declared unfitted
reference collision responsible for R1's right-leg neutral loading; it preserves
actual hardware contacts, including opposing-foot contacts during bad transfer.
Contact proxies are not continuous CAD clearance.
The 23 initial single-hull meshes remain solely for the retained coarse-collision
model under `development/`, which records the filled-torso artifact. Current
URDF/MJCF descriptions use the 592 decomposed hulls.

The original R1 export's six complete 0.8 s native episodes compare two common starts on rigid nominal,
lower-voltage/delayed, and flexible nominal bodies. Every 1 ms observation station
and early numerical end is retained, while physics microsteps are counted
separately. These are closed-loop assumed servos following a lossy mapping of
an older attained-position trajectory. They are **not** old actor inference,
new training, or successful walking: all variants lose balance. The source actor
has 72 inputs/21 outputs; this model's interface has 75/22. The mapping adds neck
pitch and both ankle rolls, removes two old grippers and explicitly handles
spatial sides and axis signs. It is a diagnostic, not a policy compatibility claim.

The later source-bound walking diagnosis runs the actual retained actor on its
source CPU body, then explicitly adapted inference and recorded targets on R1
and R2. The source walks about 1.08 m over four seconds. R2 neutral holds, but
same-angle crouch/actor transfer still falls. Current-geometry step references
expose high stance hip-roll demand and incomplete touchdown/recentering. A
pointwise IK branch jump was repaired in a separate continuous path; matched
numerical replay then isolated and removed the external-servo alternating mode.
Actual floating-base balance still degrades after unloading. These
are model/control development observations, not tuning readiness. Full episodes,
source contracts, load/contact distributions and selected videos are in the
existing learning dashboard under `cove50-current-body-walk-diagnosis-r2-20261026`.

The perception exercise renders native RGB/depth and checks projection,
capture-time transforms and missing/stale packet handling over 61 prescribed
head poses. At 100 ms arrival-pose substitution, the prescribed scan has about
82 mm median point error. Known simulated capture pose is supplied for this
transform check; it does not establish estimator accuracy or physical calibration.

## Reproduce the workflow

Use the repository `.venv` for native CAD/MuJoCo and `.venv-control` for MJLab.
Restore exact source paths and dependencies named in `source-lock.json` and
`compliance.json` through the existing blob store before running collection.
Never treat an evicted source as an absent part.

For the current numerical revision, follow the three regeneration commands in
`r3-20261026/README.md`. R2's collision recipe remains in its own README. The
original full R1 export workflow is retained below as provenance; do not
overwrite frozen evidence to regenerate a derivative.

The original scripts in `r1-20261026/` ran in this order: `collect.py`, `compliance.py`,
`contracts.py`, `build.py`, `collisions.py`, `build.py`, `geometry_identity.py`,
`validate.py`, `perception.py`, `replay.py`, `numerical.py`, `render.py`, `report.py`,
`package.py`, `portable.py`, `seal.py`, `publish.py`. The initial
build creates visual meshes for decomposition. Reuse the sealed geometry caches
when source identities are unchanged. Run `mjlab_adapter.py` with `.venv-control`.
The current local physics variants and meshes remain materialized. Large evidence
and the portable package are privately preserved; generated bulk is not Git source.

**Known gap.** The validation replay in `build/` (`replay.py`, `validate.py`) used a September policy on the rejected Frame50 body and two Frame50 probe files; those inputs no longer exist in the bucket, so that step cannot run. The model itself is built and checked by the later stages.
