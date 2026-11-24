# Perception and sensing

Status 2026-11-20. Perception work is paused. Its Idle tree (`T-tr3ec8`) is parked for re-planning on the selected printed body (`robot/design.json`). The reusable modules are in `control/perception/` (README there); the runner chain behind the whole-task numbers is in history (`git checkout history/perception-runners`, `experiments/ape_high_level/`, commit `6fbb278e4`).

Scope: cameras, depth, range, IMU and proprioception, audio, touch, captured worlds and localization, as measurements. How the behaviour uses them (belief states, whole-task room results, stop and response policy, owner identity) is in [high-level behaviour](high-level-behaviour.md). Sensor parts, connectors, masses and prices are in [actuators, power and electronics](actuators-power-and-electronics.md).

## How to read this file

- **Bodies.** Almost every result ran on the APE0 Direct Frame (a 21-axis simulation body, rejected 2026-10-10, removed 2026-11-12), on a kinematic 8 cm or 0.4 s step surrogate, or on a 17 cm collision proxy. No sensor is mounted, powered or calibrated on the selected body. Read each number as evidence about a sensing role, not as a part spec.
- **Idealisations that recur.** Rendered depth, virtual 72-beam range planes at 4 / 8 / 11 / 30 cm, ideal yaw, exact joint state, scripted owner and props, kinematic stops. A result that depends on one says so.
- **Evidence ids.** `V<n>` are run ids of the closed-loop chain (`artifacts/ape-closed-loop-cove50-v<n>/`), kept once per finding.

## What carries over to the selected body

1. **Sensing is a set of separate jobs.** Social and object RGB, floor and near-hand views, metric depth for the held item, room-scale pose, near-body protection, and body motion and contact health each succeeded or failed on their own. No single module covered more than two jobs in any test.
2. **A separately monitored range role carries the whole task.** In 30 recorded rooms a two-height range route completed 28/30 owner tasks and a two-view RGB-D route 18/30 (detail in the behaviour file).
3. **Near-floor obstacles need a measurement near the floor.** A 4 cm range plane caught crate edges that 8 cm, 11 cm and head depth missed. No tested mountable layout closed the near-body gap.
4. **Localization needs a geometric correction beside motion packets.** Joint, contact and IMU increments alone drift 14-20 cm over 2-3 m walks; scan registration brings it to 1-4 cm but can report healthy while wrong.
5. **Freshness, capture pose and health travel with every packet.** Stale views, 3 cm extrinsic error, mistimed head yaw and shared sector faults each defeated a decision that looked fine on ideal data.
6. **Real-image identity is weak.** A generic pipeline chose the right class in 26/61 present real-image trials; at the thresholds tried, confident claims were wrong more often than right.
7. **Millimetres of camera placement matter** near the muzzle and the paw. Placement must be checked in the posture that uses it, on the real head geometry.
8. **No hardware sensor data exists** for the ape: no mounted-camera footage, microphone or motor-noise recording, IMU or encoder trace, touch telemetry.

## Cameras: coverage and placement

At the provisional 44.6 cm eye height, one yaw-only 67 deg vertical view cannot hold an adult face at 1 m and an 8 cm floor object at 0.4 m in one upright frame (2026-10-06, `T-pjrh73`). A face at 1.5 m height and an object at 40 cm range are about 92 deg apart in elevation. Screened views (5 sample points each, standing pose):

| Candidate view | Shown object, 35 cm | Adult face, 1 m | 8 cm floor object at 25 / 40 cm |
|---|---:|---:|---:|
| Yaw-only eye, horizontal | 5/5 | 0/5 | 0/5 / 0/5 |
| Pitchable eye: neutral / 45 deg up / 45 deg down | 5/5 | 5/5 | 5/5 / 5/5 |
| Two fixed eye views, 25 deg up and 40 deg down | 5/5 | 5/5 | 1/5 / 5/5 |
| Same, down lens 5 mm forward | 5/5 | 5/5 | 5/5 / 5/5 |
| Fixed 25 deg up eye plus 40 deg down chest view | 5/5 | 5/5 | 5/5 / 5/5 |

- No layout sees an 8 cm object 12 cm ahead. A 3 cm item at 80 cm spans about 13 x 12 px in the downward views.
- With 40 deg head yaw the fixed down view sees the right item 5/5 but its own muzzle hides the centre item (0/5); moving the optical centre 5 mm forward gives 2/5, 10 mm 5/5.
- Walking view (forward-policy trace, 8 cm marker 12-40 cm ahead): a head-down and a low chest view see it in every sampled frame; below 12 cm the head view 0/4, the chest 3/4. Marker motion in the chest view was a median 9.6 px (p95 15.7) per 20 ms. The chest view intersected the older torso shell.
- Near-paw view (3 cm object 25 mm ahead of the paw): at 60 deg forward reach the fixed down eye sees 5/5, a pitched eye 3/5; at 90 deg reach 4/5, 0/5 and the chest view 0/5. In a saved side reach neither head camera nor a forward torso view saw a 2 cm toy; an outside-forearm view saw it in 90/90 samples (median 1,779 px), a rear torso view 90/90 at 168 px (2026-10-12, `T-5ytcbv`).
- Camera-goal reach: a goal locked at the 7.5 s sighting gave contact in 7/8 and 11/12 fresh scenes; updating it from every capture failed (0/4 against 4/4 entry-locked), because a moving camera shifts the perceived goal 12-20 mm while the cube stays put (2026-10-10 to 10-15).
- Pitch at an owner standoff: a fixed 70 deg view framed head and held can in 0/12 close postures, three ideal 90 deg pitches in 10-11/12. Per-object depth quality was a better test than whole-frame validity: 90 deg met a central-depth check yet found no person; 70 deg found two people with 100% valid depth in each box.

What would change this: mounted footage at the real optical centre (show, face, floor search, near hand) with a pitchable camera and two fixed candidates on the same scenes, measuring exposure, blur, focus distance and occlusion during head, arm and body motion.

## Depth roles

Depth did four jobs and no candidate did all of them:

| Job | Best evidence | Limit |
|---|---|---|
| Metric position of a shown item | RGB-D gave a can centre with 2.6 cm error; RGB alone only a visual find (V471-V472) | One can, ideal renderer depth |
| Local obstacle marking | Head depth with predicted self-pixel masking restored a cube near the arm that the range planes missed (V92-V96) | Does not replace low range |
| Egomotion | 70 deg RGB-D ended 0.3-0.7 cm from the camera path in three Isaac approaches with 20% short odometry; without depth 6-8 cm short | Ideal calibration; ORB+PnP worse than scan matching in a textured room |
| Owner arm and item surface | A before/after depth view gave the changed surface a range track missed (V511-V523) | Ideal 90 deg view |

- **The two fixed Pi camera views (+30 / -40 deg) give no stereo.** 0 of 152,149 sampled lower-view pixels projected into the upper image in a full 102 deg render (V211, V227).
- **Range windows** (ideal masks, not devices): a D405-class 7-50 cm window had can pixels in 2/11 to 6/12 visible views and none at the forward owner views; a D430-class 0.3-3 m window covered all 11 and gave the same routes as unlimited depth; a VL53L8CX 8x8 zone hit the can in 6/11 views and located it in 4/9 (V212-V215).
- **RGB floor ray.** A box-bottom ray to the known floor located a floor can within 6 cm in 9/9 accepted boxes, but put a table can 7.42 m away. It needs support-plane evidence (V212-V213).
- **The window must cover the decision boundary, and age matters.** For an object about 62 cm away a 7-50 cm window asked for help; 30 cm-1 m and 30 cm-3 m windows gave an owner-confirmed find. A 400 ms-old depth view lost the benefit (2026-10-13).
- **Head depth sees what a chest view cannot.** A moving 8 cm side prop gave 0 upper-forward pixels but 994-4,257 pixels in a lower head view yawed +1.2 rad; four noisy shin views gave 2-11 points (2026-10-10).
- **Synthetic depth defects changed nothing.** 6% missing tiles, 10% missing pixels, 2 mm + 1% noise and 3 mm bias left every action sequence unchanged (V266). The failures were in view reconfirmation and scan/motion consistency.
- **Self pixels.** A predicted-self mask from CAD and joint state removed 14,010 self pixels and reopened the local graph from 1 to 74-91 cells; a blanket 12 cm near-camera exclusion hid a cube 4 cm from the arm (V92-V96). In a side look about 40,700 predicted self pixels sat 12-21 cm from the eye. Calibration error and fast near obstacles are untested.

Open: the mounted, self-occluding head placement of an OAK-D SR or D435i class device; stereo holes on dark, reflective and transparent items; exposure and processing latency.

## Near-floor and low-obstacle range

Virtual, central, ideal rings; these say which heights and views carry information, not which part to buy.

- **The 4 cm plane carries information the others lack.** On a 312-action varied-crate bank (289 safe at a 5 cm clearance label): 11 cm range plus any head or chest depth passed 11-17 unsafe moves; adding the 4 cm plane passed 0 and refused 11 safe ones. The head saw the 9 cm crate's top while the limiting contact was the sole near the floor (V56-V62).
- **Heights.** On 592 new-layout actions (81 unsafe): flat 11 cm passed 55 unsafe / refused 36 safe; 4+11 cm 7 / 23; 4+8+11 cm 0 / 27 (V70). On a single 4.63 cm edge, every noisy 4 cm and three-plane replay rejected it and no 8+11 cm replay did (V128).
- **Field of view.** A forward 90 deg sector matched a full ring only while unsafe moves were forward; on a rotated cohort it passed 20 unsafe against 5 for the full ring (V63-V64).
- **Representation.** Sparse points leave unsafe passes; footprint projection or height-limited columns remove them but refuse 20-45 safe moves (V65-V68).

Mountable layouts tried on the rejected body (unsafe passed / safe refused):

| Candidate | Result |
|---|---|
| Ideal body-centre 4/8/11 cm + head depth (reference) | 0 / 27 of 81 / 511 |
| Two front shin 8x8 ToF views | 81 / 0 (V190) |
| Four forward shin views | 37 / 8, all 37 outside the forward 90 deg |
| Eight shin views, front and rear | 26 / 16, the rest are side sole-to-crate edges (V216) |
| C1-class single-plane lidar, five placements | torso-centred plane fully self-occluded; head-top plane passed all 81; forward 11 / 4 cm planes passed 62 / 23 (V237) |
| Paired shoulder 8x8 ToF at table blind spots | 206 false-safe against 232 for low range alone, over 1,424 poses (V39) |
| Front and flank 51x31-ray fans | side unknowns 108 to 92, forward 45 to 43; no 8 cm step certified (V319) |

Limits: static after-the-fact screens with kinematic stops and ideal rays; mount, leg occlusion, reflectance, latency and the stop tail are not modelled. The native forward-policy stop tail (up to 11.5 cm after a zero command, V210) exceeds the observed live-path clearances of 9.4-13.9 cm.

## Moving obstacles and channel independence

Sensor-side results; the stop and response policy is in the behaviour file.

- **Mount height decides what a stream sees.** With four saved native stop traces: a low 15 Hz stream gave no overlap with an approaching person (gaps 8.6-16.9 cm) and so did mid-height; for a low pet the mid stream never stopped (four overlaps) while low 15 Hz had none but two gaps under 5 cm (2026-10-12). A 5 cm plane missed an approaching owner that a 30 cm plane and a fused 5+30 cm stream flagged at 3.70 s; mid and fused kept 100/100 alarms under 30% beam or 20% scan loss, but five consecutive blind sector scans removed the cue (V547-V549, V550-V567).
- **Cadence and latency.** At 10 Hz a two-packet motion prompt came 0.56-0.70 s before a 5 cm crossing; at 5 Hz 0.04 s late, at 2 Hz 0.54 s late (V500-V504). A worker thread kept 89 scans during an 8.89 s detector call, longest interval 212 ms against 100 ms; RGB+depth packets had median 23-36 ms and worst start gaps 161-315 ms; Isaac wall gaps reached 0.38-0.60 s. The released person detector took a median 3.10 s per image; one RGB-D capture 18 ms median, 56 ms p95.
- **Independence needs sector-health provenance.** A 30 cm range sector frozen at plausible values let the actor pass 2.6 cm from the person; a second 45 cm plane or a rendered owner-view depth stream kept 26.4-28.5 cm. Freezing the range sector and the tracked depth sector together brought the 2.6 cm gap back, with fresh delivery timestamps and plausible return counts (2026-10-12).
- **RGB image-difference cue.** It rescued a case where depth and range were both stale, but a 25% gain pulse false-alarmed and a 5 deg pitch change moved 18.8% of pixels. Resetting the reference after 1 mm or 0.5 deg of camera motion removed the false alarms (thresholds assumed).
- **Two-view velocity is fragile.** A 3.36 cm overshoot made a stationary prop look like it was leaving; 6.36 cm approved one unsafe action. A lower bound 10 cm/s below the estimate approved 0/7 unsafe but blocked 2/3 safe (V307-V316).
- **Capture age is its own health term.** Registering a 400 ms-old low scan at its capture pose removed a false 0.17 m/s prop velocity but did not make it safe for motion; a 3 cm registration error flipped help requests either way (2026-10-13).
- **Discrete zones overstate clearance.** Multizone ToF models overestimated clearance by about 5 cm; a two-packet closing cue false-alarmed in 20/20 stationary drift trials; flush side modules self-occluded in all 74 poses (Isaac, V217-V227).
- **Map ghosts.** A permanent map update left three ghost points after a stool left; a temporary layer expiring after three unseen scans removed them (V251-V259).

## Localization and body-motion sensing

- **Command odometry drifts, scan registration repairs it.** Under alternating slip: final XY error 13.9 cm (command), 1.1 cm (scan registration), 3.4 cm (RGB-D odometry) (2026-10-12, `T-5ytcbv`). With a 20% motion undercount in 30 rooms, adjacent two-height scan correction cut the median maximum pose error from 14.6 to 4.3 cm.
- **Registration can be confidently wrong.** In repeated cabinets a 3.0 m range reported "registered" on about 137 wrong frames per run with 2.15-2.49 m final error; a 5.0 m range anchored the aisle within 0.2-4.4 cm (V400-V406). A scan alias proposed -3 cm against a +3.7 cm step while reporting 94-96% inliers (V141-V142). Inlier quality does not reveal an alias.
- **Heading.** A 5 deg abrupt beam-bearing fault gave 20-28 cm false arrivals; an innovation gate stopped abrupt faults (3.4-4.0 cm) but missed a gradual 0-5 deg drift (15.5-18.3 cm) (V417-V422). With 1 deg/s heading drift a translation-only registrar let 75 of 395 frames through with over 10 cm error; adding heading fixed the varied walk but left 77 frames over 5 deg on a curved walk (`T-mftnhd`, `T-g3w6zn`). No tested gate catches both abrupt and gradual faults.
- **Proprioception alone is not enough.** With exact joints, ideal orientation and contact bits, foot-anchored odometry ended 2.3-20.3 cm off after 1.9-2.6 m walks; yaw drift or contact delay alone gave 7-26 cm. Sole-centre travel during contact was 19-30 cm per foot, foot roll and patch motion rather than slip.
- **Stop detection from body packets** (20 packets of 50 Hz, lossless simulator data): IMU yaw rate declared stop 0.06-0.38 s early with under 0.8 mm path left, encoders similar, sole force 0.36-0.86 s early with up to 11.3 mm left; fused 0.06-0.30 s early (2026-10-11).
- **Maintained maps.** A fixed-origin map supported 28/60 returns in a moving room, a maintained map 50/60 (V251).

Interface rules that follow: pose is reconstructed at capture time; late, old and duplicate frames are tagged; scan quality, translation and heading uncertainty stay separate; independent yaw cues are cross-checked and persistent disagreement stops metric motion.

## Object and person perception on images

- **Real-image search (HOPE, 2026-10-06).** Generic proposals + DINOv2, 12 taught classes, 108 searches: the right class in 26/61 present trials (a matching proposal existed in 46/61); something was always proposed, including all 47 absent trials. A 0.60 score gate makes 36 claims, 11 right; 0.65 makes 12, 8 right; 0.70 makes 4, all right, but misses 22 of 26 correct choices.
- **Teaching crops.** Grounding DINO crops overlapped the clicked object in 4/12, SAM 2 point masks in 12/12; present choices 10/61 (generic click), 18/61 (SAM 2 click), 26/61 (evaluator crop). Owner correction of a contradicted crop raised present choices from 12/36 to 17/36.
- **A second view does not confirm.** Top choices in adjacent HOPE-Video frames landed on the same world point in 19/51 pairs; 8 were the true target in both.
- **People and pointing.** Released detectors fire on hands, garments and clutter (person detector 27/40 false on no-person frames; face detector 0/40 false but misses 12 of 40 faces). A finger ray chose the right object in 59/90 natural points at 640 px, even with true object centres. A hand was visible in six of nine no-pointing frames. 129 images had an auditory cue but no visible person, so `heard_cue`, `visible_person` and `associated_speaker` stay separate.
- **Short-range RGB bearing** tracked one shown can at 10 Hz; losing two frames left a 200 ms-old track (V374-V378).

Limits: tabletop RGB with evaluator-projected boxes; no ape camera, real owner or moving body.

## Captured worlds

- **Room sources for the whole-task loop.** 30 development-exposed rooms: 27 ScanSceneVerse scans, one Hallwyl capture, two synthetic Habitat rooms; several have holes and smeared textures, and inserted people and items float. Replica PTex re-renders reproduced saved poses pixel-identically (at least 71.4 dB PSNR); its floor height varies about 8.8 cm along one route, so a global floor plane rejects valid poses. Of 65 Habitat-GS scenes, about ten look indoor and none is certified photoreal. In 12 varied furnished rooms 5 were route-valid for a moving stool. RoboCasa kitchens give a textured upstream loop but need a scene-validity check.
- **Gaussian worlds from phone captures (2026-10-26 to 10-27, `T-deez3j`, `T-yzd56r`).** One indoor room, captured twice as plain video and once with Spectacular Rec (video and IMU).
  - The first 30k-update refinement fit filled the views with floaters (held-out median 7.76 dB against a 19.42 dB baseline fit). Cause: cuDNN TF32 in the SSIM path of training. On one frame TF32 gave SSIM 0.454 and a gradient norm of 2.17 M; IEEE float32 gave 0.650 and 0.008, matching float64 within 2.1e-6. `control/perception/worlds/train.py` now disables TF32.
  - The corrected fit raised the first recorded view from 9.96 to 24.48 dB PSNR and the held-out median from 19.42 to 20.85 dB.
  - The sensor recording (100 s, 3,001 frames) gave world002: 211 fitted and 35 held-out views, 2.37 M Gaussians, held-out mean 23.18 dB. The sparse 53-view world001 used a different held-out set, so this is not a matched comparison. Turns, thin chair edges and the late return still blur or tear, and sideways views show clouds and missing surfaces. An RGB consumer run on it (17 observations) missed two visible chairs.
  - Use: inspecting a recording and limited RGB perception tests. A splat gives images, not contact: scale, floor, collision geometry and localization are not established. Capture procedure: `control/perception/worlds/capture-guide.md`.
  - The 2026-09-21 design route (NVIDIA 3DGRUT exported to Isaac Sim) was never executed.

## IMU, hearing and touch

- **IMU.** A torso reserve for a Pololu MinIMU-9 v6 (20 x 13 x 3 mm, 0.7 g) and a separately designed LSM6DSO board exist (part details in the electronics file). No estimator, noise, magnetometer-near-motor or calibration result exists. A head-camera IMU does not replace body IMU and joint feedback.
- **Hearing** (2026-10-08, `T-kvs73w`). No microphone, motor-noise, echo or two-speaker recording exists. The interface is a timestamped speech interval with channel health, VAD/ASR status, playback reference and motor state, not a final transcript; an unclear name asks for repetition. A recording plan compares one head-area microphone with an external directional array on missed names, self-triggering, wrong-person address and barge-in latency.
- **Touch** (2026-10-08 to 10-12, `T-xvcza3`). Petting and possession need different evidence. Cameras and motor current cannot see a quiet hand on the shell, and the XL330 reports supply current, not phase current. A one- or two-zone capacitive electrode on a temporary cover is the first comparison; nothing is selected.

## Open questions

1. **Camera layout on the selected head:** one pitchable view or separate fixed views, from mounted footage with arm and body in motion.
2. **A mountable low-range layout** that closes the near-body gap on the selected body (field, mount height, leg occlusion, scan rate and stop tail together).
3. **Whether close metric depth earns its mass, power and calibration.** It gave a 2.6 cm item location where RGB gave a visual find; held-out approach, hand and dark or transparent objects are untested.
4. **Pose source mix on the walking body:** a time-aligned trace with realistic calibration and a look-away-and-return task.
5. **Identity on ape-view images** with a real owner, wrong clicks and absent targets.
6. **Speech** with motor noise and echo; **touch and possession** with a real contact channel.
7. **Capture-disjoint, appearance-qualified rooms with moving people** for the transfer test; the 30 exposed rooms are the baseline, and a better phone-captured world needs a continuous, overlapping capture.

## Evidence index

- Code: `control/perception/` (identity, people, range, localization, sensor models, worlds). Runner chain: tag `history/perception-runners`.
- Idle: `T-pjrh73` (sensor decision), `T-5ytcbv` (ablations, pose slip, reach view), `T-mftnhd` (heading), `T-g3w6zn` (gait range fusion), `T-kvs73w` (hearing), `T-xvcza3` (touch), `T-gqgzz4`, `T-rsjape` (localization), `T-deez3j`, `T-yzd56r` (phone worlds). Decision `D-sznrsa` (Direct Frame rejected).
- Dashboard records: `spectacular-room-world002-20261027`, `iphone-room-3915-20261026`, `ape-perception-20261005T163242Z-5ad34f`.
- Artifacts: `artifacts/ape-closed-loop-cove50-v<n>/`, `artifacts/ape-sensors-50cm/`, `artifacts/ape-depth-guard-faults/`, `artifacts/ape-rgb-photometric-guard/`, `artifacts/ape-real-room-next-captures-20261014/`, `artifacts/iphone-room-3915-v1/`, `artifacts/spectacular-room-world002/`.
