# Selected APE0 model: collision correction R2

R2 corrects the rigid collision treatment of 44 source shapes explicitly marked
as non-BOM, unfitted aesthetic references. Their appearance remains visible.
The actual frame, battery bridges, power and sensor mounts, selected head/neck
panels and other represented hardware retain collision. Affected link hulls
are regenerated from the retained physical geometry; mixed hulls are not
arbitrarily disabled. No new body-pair exclusion is introduced.

The nominal 3.8303 kg estimate is unchanged to isolate this correction. It
includes **0.8138 kg of conditional reference-skin mass/inertia**, plus 0.150 kg
installation allowance. Neither is measured completed-robot mass. Removing
unfitted reference collision does not solve production shell fit: the original
right-thigh reference overlaps a real battery bridge by about 111 mm³ in native
CAD. The eventual fitted panels need their own geometry, mass and clearance
evaluation before replacing this allowance.

Use `cove50-rigid.xml` with the named 22-channel `Runtime` for the native MuJoCo
learning experiments. `cove50-compliant.xml` also retains the provisional FEM
connector reductions, their numerical stiffness limitations and their separate
very small timestep; it is not a calibrated flexible robot.

**Consume URDF and SRDF together.** `cove50.urdf`/`cove50.ros.urdf` carry the tree,
meshes, inertia and frames; `cove50.srdf` carries the existing assembly-mating
collision exclusions. A bare URDF imported into another simulator can generate
forces between intended mating parts. MuJoCo's direct URDF import does not
automatically apply SRDF. Compilation or FK agreement alone does not prove
equivalent dynamics. The supplied MJCF includes the exclusions explicitly. A
new simulator importer must apply the exclusion/actuation/timing contracts and
check matched native motion before using a policy.

The importer must also create the intended floating base: direct MuJoCo URDF
import produces a fixed-base 22-coordinate body, while the supplied rigid MJCF
has 29 position / 28 velocity coordinates including its free root. These are
different simulation entry conditions despite matching articulated FK.

The CAD torso origin is at 300 mm and the added TPU sensor pads reach 2 mm below
the old CAD ground datum. Spawn/reset a level neutral body with torso Z around
302.2 mm for 0.2 mm sole clearance; derive the exact height from the compiled
foot colliders for a changed pose. Do not train from the default source pose's
floor overlap or anchor the base. The current native episode recipes retain
their exact reset state and the sensor bootstrap.

R1 artifacts and recipes remain immutable historical evidence. Their saved
policies bind the R1 model, source and observation/action contracts. Reproduce
that run from its recorded source commit; a same-shaped checkpoint is not
automatically an R2 policy.

Regenerate R2 locally from the repository root:

```sh
.venv/bin/python -B robot/model/collisions/prepare.py
.venv/bin/python -B robot/model/collisions/collisions.py
.venv/bin/python -B robot/model/collisions/build.py
.venv/bin/python -B robot/model/collisions/verify_revision.py
```

Preparation verifies the parent seal and refuses to replace prepared inputs.
For a partial regeneration, resume collision/build/verification. The R2 native
validation compares unchanged mass, inertia, joint tree, axes/ranges, actuator
and sensor arrays, and neutral FK against R1. Fresh whole-motion diagnosis is
reported separately through the learning dashboard. This revision does not
claim stable walking, calibrated motors, fitted shells or tuning readiness.
