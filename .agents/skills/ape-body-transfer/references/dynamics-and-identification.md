# Dynamics assumptions and identification

Use for scale transfer, actuator selection, simulator/contact changes or planning a physical measurement. Read `body-contract.md` for the current local source ledger and task entrypoints.

## Scaling is a conditional calculation

For geometrically similar bodies at unchanged density, scaling every length by `s` gives mass proportional to `s³`, inertia to `s⁵`, and quasistatic gravity torque to `s⁴` (mass times lever arm). For 35→50 cm, `s=10/7`, giving approximately 2.92× mass, 5.95× inertia and 4.16× gravity torque. These follow directly from geometry; they are **not predictions for our assembled ape**, whose purchased motors, structure, proportions and mass distribution are not uniformly scaled. Calculate the actual per-link mass/COM/inertia and joint demand from the selected candidate instead of scaling the old torque limit by height. When demand exceeds a catalog point, record the gap as evidence and grade the drive dimension 0–5 ([evaluation principles](../../../../docs/design/evaluation-principles.md)); ask whether the actuator choice or the requirement (speed, payload, mass budget) is wrong, and put any requirement change to the user.

Distal mass changes both static moment and acceleration demand. A small fraction of total body mass at a hand/foot can matter disproportionately. Keep fixed-hand and opening-hand mechanisms distinct. Joint names may reflect a source convention rather than the anatomical function in the current pose; inspect world/local axes, signs and representative displacements. Matching a functional axis does not justify deleting another motor's supporting structure.

## A simulator is a specified approximation

Collision geometry, friction/contact parameters, constraint solver and integrator/timestep affect the result. A more detailed visual mesh or photorealistic reconstruction does not identify contact compliance, friction, mass or drive delay. Refitting appearance can improve image realism without improving force realism. MuJoCo contact is a modeled soft-constraint system with configurable solver parameters; a successful rollout is evidence about that model and configuration only. [MuJoCo modeling](https://mujoco.readthedocs.io/en/3.3.6/modeling.html#solver-parameters).

Before attributing a change to learning, check reset/contact initialization, timestep/decimation, joint-limit and actuator saturation, mass/inertia plausibility and the control adapter. A kinematic IK path or support-polygon/ZMP screen is a feasibility clue under assumptions, not a general guarantee against dynamic falls, slip, torque infeasibility or contact loss. Inspect the actual contact and acceleration behavior.

## Measure only what the experiment can identify

Source/catalog parameters, assumed sensitivity ranges and physical measurements need distinct provenance. Stall torque, no-load speed and rated/continuous operation describe different conditions. A nominal electrical current signal is not automatically motor torque, output torque or fingertip force; locate the selected actuator's documented measurement point, gearing, efficiency and temperature/load dependence.

Before fitting many parameters, list available observations, excitation, units and the combinations they can distinguish. Multiple friction, damping, delay and gain settings may explain one trajectory. Compare fitted behavior on held-out motions and loads; disclose non-identifiable combinations rather than reporting precise-looking parameter estimates. Keep physical measurement prerequisites explicit in `T-cgx9gk`/`T-j8fmq8`; no amount of simulated fitting supplies missing real traces.

Randomization is a training distribution, not a robustness certificate. A tested parameter being inside a declared range does not prove sufficient sampling, coverage of correlations, or successful learning there. Compare nominal and shifted conditions, choose plausible correlated variations, and separate uncalibrated stress assumptions from measured uncertainty. Do not tune ranges on the reserved evaluation outcomes and then call those outcomes held out.

Differentiable simulation may support some identification/optimization problems. Contact mode changes, available solver derivatives, model mismatch and partial observability still matter; verify the exact engine/version/solver's differentiation support. A new engine announcement does not establish end-to-end identification of this ape. Use a simple replay/sensitivity baseline before proposing a new simulation stack.

## Preserve the selected robot

ToddlerBot and other references can inform joints, transmission and control. The ToddlerBot-shaped `functional_50` attempt was ruled out and removed at repository commit `c1fbe88`; its earlier replay results are historical, not a selected body or a current runnable path. Do not resurrect that exterior through a “source-backed” recommendation. Read the current user brief and Idle state before transferring any mechanism into the ape silhouette.

Current hypotheses belong in `T-t28835` (explicit provisional control candidate), `T-t7wb6j` (one consequential axis/transmission comparison) and the existing calibration tasks. A source-part envelope, pose clearance screen, connected assembly and measured duty test answer different questions; report exactly which one was run.

The active replacement is the bare ape-frame tree `T-59kag5`, with sketch/Cove anatomical proportions retained and cosmetic skin deferred under `D-3ejdcp`. Reuse its `ape_frame50/frame_contract.yaml` and eventual `T-xmnhsx` export; do not fork another nominal body. A declared mass budget, reserved sensor frame or catalog point in that manifest remains an assumption until the relevant build/measurement exists.
