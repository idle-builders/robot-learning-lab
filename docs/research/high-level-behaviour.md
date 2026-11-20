# High-level behaviour

Owner interaction, search, routes, belief and the near-body stop policy. Status 2026-11-20: paused; the perception and behaviour Idle tree (`T-tr3ec8`) is parked for re-planning on the selected printed body (`robot/design.json`). Code: `control/behaviour/` (frozen option head and viewpoint routing) on top of `control/perception/`; the whole-task harness is in history (`git checkout history/perception-runners`). Sensor measurements are in [perception and sensing](perception-and-sensing.md).

## What these results are

- All simulation, 2026-10-05 to 10-16. The body was a 17 cm kinematic proxy or the Cove-50 Direct Frame (rejected 2026-10-10, removed 2026-11-12), so no mount, stop distance or clearance transfers to a built robot. What transfers is the **software contract**: which states the behaviour keeps apart, which evidence moves it between them, and which failures appear.
- Owner, bystanders and props are scripted. Owner replies come from evaluator truth. The 5 cm margin is a development proxy around modelled bodies, not a human-safe distance. Near misses are counted next to the continuous gaps.
- Most room results use rooms already exposed during development: they show consistency, not generalisation.
- The target stays: a 50 cm ape that learns a shown object from its owner, finds it again after it moved, approaches only when the route is currently safe, and says honestly when it does not know. The first interactions are notice me, teach me, find it again, play with me, help me, show me what you learned; teach-and-find is the first connected probe. Fetch, cleanup and play with animals add possession and contact checks later.

## 1. The loop and its belief states

```
owner cue + camera + joint/IMU/range health
  -> teach/identify: owner-confirmed crop, proposals, appearance + depth
  -> belief: time-stamped views, robot pose, 3-D occupied/free/unknown scene
  -> choose: look, information viewpoint, short approach, ask, or stop
  -> skill request: bounded motion with source frame, map epoch, deadline, cancel id
  -> independent proximity stop running during moves, waits and looks
  -> new image/range/pose/contact evidence -> verify, recover, or ask
```

Item states are never merged into "found":

| State | Evidence needed | Allowed next action |
|---|---|---|
| Taught | owner confirmed one image crop and name | remember the view, not a location |
| Candidate | fresh RGB proposal resembles the crop | look again or ask; no "found" claim |
| Located candidate | fresh image + support/range evidence + healthy pose agree | plan a viewpoint or target-area route |
| Owner-confirmed | owner yes (or a distinctive marker or continuous track), then a fresh image | report a visual find with its source image |
| Occluded/uncertain | old view useful, current identity or pose invalid | head look, safe viewpoint, or ask; no stale approach |
| Route clear | fresh known-free evidence with body margin | one bounded motion request, stop monitor active |
| Blocked/unknown | sensors reject the edge or the map is disconnected | reobserve, change side or posture, or ask for help |
| Arrived/held | new image plus observed stop | tell the owner only what was observed |

Rules that held in every run:

- A request is not a completion. Wait for the observed stop and a post-stop image before upgrading identity or reporting. Body packets give a useful provisional "body quiet" trigger but not how far the base moved.
- One owner of gaze and motion per turn: listening may orient to the person, teaching looks at the object, verification looks again at the candidate.
- Three kinds of learning stay apart: owner-taught names and places (inspectable memory), demonstrations and corrections (dataset), a changed model or skill (only after training and a fresh closed-loop score).
- A blocked or uncertain route produces another look or an honest request for help, never an invented find or a move toward a stale coordinate. Per-packet `missing` or `stale` is never read as free space. Room-specific thresholds or exceptions are not allowed in the actor.

## 2. Teach, identify and find

- **The image gate is a candidate generator.** An appearance gate accepted a different red-labelled prop in 15/18 target-absent views and an exact-replica swap in 15/18. What worked: ask the owner at the safe stop, then take a fresh same-region image. Present items reached `owner_confirmed_visual_find`, lookalikes `owner_rejected_candidate`, absent rooms asked nothing (V247-V250). A narrowly missed gate (hue 0.683-0.698 against 0.70) is recovered by a guarded head look plus one owner yes/no plus a recheck (4/4), not by lowering the gate.
- **Keep four negative outcomes apart:** `not visible`, `visible but no proposal`, `identity ambiguous`, `owner rejected`.
- **Teaching aids enter the evidence path, not the answer.** A SAM 2 click mask rescued all three failed teaching crops but produced one wrong candidate in an absent trial; used only conditionally it kept 38/38 correct present offers. A slower generic package pass recovered two visible misses but doubled image-worker time; run it only after an unresolved search (39 of 76 conditions).
- **The owner's yes/no binds to the displayed frame and crop.** A rejection blocks re-offering that frame. Silence or a late reply leaves motion unavailable.
- **Object choice matters as much as room.** Over 20 recorded rooms one item pair was found 20/20, another 13/20 (visible misses with no proposal), a third never cleared the teaching pixel gate. Changed item plus dim light in 7 Replica rooms: 0/7 completions, no wrong approach.
- **Raised targets.** A can moved from floor to table made floor-only loops stop on expired memory; a guarded upper look plus owner confirmation and a fresh image found it in 3/3 present rooms with no false question in 3/3 absent ones (V217-V219).
- **Pose health decides whether a remembered coordinate may be used** (HOPE-Video replay, 365 frames, 10 cm gate, `T-j6rhvd`):

| Pose input | Pose error | Correct / wrong visible accepts |
|---|---|---|
| Published pose (upper bound) | none | 27 / 0 |
| Continuous RGB-D odometry | final 6.0 cm, 5.8 deg | 27 / 0 |
| 10-frame dropout, held pose | 8.9 cm | 14 / 5 |
| 41-frame dropout, held pose | 37.3 cm | 16 / 12, plus 2 out of frame |

  Mark pose unavailable at the first skipped update and withhold coordinate-based approach.

## 3. Whole-task results over recorded rooms (2026-10-15 to 10-16)

One frozen state machine (approach owner, owner-confirmed teaching crop, present/absent search, fresh RGB-D relook before motion, bounded approach, owner review) from a fresh start in ScanSceneVerse, Replica, Hallwyl and Habitat rooms. Ideal aligned RGB-D, yaw and virtual range; motion from historical joint/contact/IMU-shaped increments.

| Run | Rooms | Completed |
|---|---:|---|
| Shared policy, development | 20 | 19/20 |
| Same policy, capture-disjoint transfer | 10 | 5/10; 9/10 after generic recovery (five-heading sweep, owner point, show-again) and one owner re-show rule |
| Unified policy, healthy reference | 30 | 28/30, 2 safely unresolved |
| Search lighting flipped | 30 | 29/30 |
| Range stream lost 1 s after the first step, stop only | 30 | 4/30 (all 25 observed gaps stopped) |
| Same outage, bounded recovery (hold, 3 fresh packets, new relook, new health epoch) | 30 | 27/30 |
| 20% motion undercount, no range correction | 30 | 19/30, 12 inside the 0.45-0.75 m target band |
| 20% undercount with two-height range correction | 30 | 26/30, all in band |
| Light flip + undercount + correction + outage | 30 | 27/30, all in band; 25/25 outages recovered; 1 wrong candidate question, 0 wrong final claims |
| Route evidence from two-view RGB-D instead of separate range | 30 | 18/30 against 28/30 (six losses on unknown forward clearance) |
| Replica photographic rooms, after floor-supported placement and guarded search | 7 | 6/7 healthy; a 1 s range gap at the end withheld the claim in all six |

- The loop breaks first at recovery states: an owner station not reached, a failed teaching click that raised instead of re-asking, a relook that returned unknown, a final view that no longer supported identity.
- Source health flows into claim validity: a missing range stream suppresses both the claim and the relook that would have produced it.
- **Learned option head.** V4 imitates the state machine's option choices: 18/20 development, 5/10 transfer direct, 7/10 with recovery, 0 wrong confident questions. V5, retrained on 122 learner-visited events instead of 40, made identical outcomes, so more examples from the same authored boundary add volume, not robustness. Frozen V6 on five mixed whole-task faults: 5/5 owner-confirmed reviews, 118 option decisions with 0 corrections against the teacher and 0 safety overrides.
- **Angular coverage of the monitor decides rear entries.** A rear entrant at 0.65 m: an all-around monitor issued `HOLD` 24 ms after the first alarm and discarded the pending image; a forward 140 deg monitor saw nothing and made one wrong final claim.

## 4. Owner interaction

All with a scripted owner in RoboCasa or proxy scenes (2026-10-10 to 10-13).

- **The unit of design is the whole interaction:** request, owner motion, view, claim, return. A view that sees the can can put the ape inside the owner's reach. Asking the owner to hold the can toward the camera raised visible can pixels from 0 to 2,626-2,905 and gave an owner-confirmed find and refind with at least 71.5 cm prop clearance (V778-V780). Change the presentation before chasing the view with the body.
- **Hand-off geometry.** A close owner forearm intersected a raised fingertip by 2.08 mm; a farther offer left 14.79 cm. Near articulated hand paths fall to about 2 cm modelled clearance, far ones keep at least 14.8 cm. "Person confirmed" by appearance spans 1.9 cm overlap to 12.2 cm clearance and accepted two static coat stands, so it cannot choose the hand-off zone (V335-V347).
- **Cued owner response.** Ask for a wiggle, wait for unique observed motion, ask for a still hold, require stillness, two fresh samples and an owner yes. One retry after 1.3 s recovered delayed owners (2/2); ignored owners stayed unknown (V391-V395). Keeping every appearance candidate and choosing only a uniquely moving 3-D track picked the held can over an identical one (9 correct, 0 wrong per seed, V379-V386).
- **Slow inference is an action window.** While a detector call ran (8.9-12.6 s to the first output), another 80 cm of owner approach brought the person under 5 cm. A concurrent 10 Hz range stream flagged it at 5.1 s, the ape backed off 50 cm and discarded the stale image (28.6 cm gap); the old owner association was then invalid and the show was withheld (V568-V578).
- **Staging.** A compact torso-like range cluster does not bound an articulated owner: a 30 cm pre-show retreat left 0.86 cm to the can under a 60 cm owner move. Observed arm and can volume from before/after RGB-D plus a 60 cm motion bound fixed the staging choice (V505-V523). A 50 cm meeting distance kept the margin under a late response; 40 cm did not.
- **Who is the owner.** Choosing the largest score-qualified person box with local depth cut a 64 cm head-to-body depth jump to 2.8 cm. After the owner left, a bystander 68.6 cm from the old track was bound as the owner. The fix: after a lost track, two consistent person-depth views make an **anonymous person**; the ape asks for reintroduction and never re-links to the old owner. Fresh teaching after identity loss is tagged `unattributed_new_session`. With two people, a location rule binds the wrong one; proposal correspondence across views is needed, not a looser bound (V582-V590).
- **Moving worlds.** With rigged people in Isaac Sim, warm image age (0.22-0.27 s) against a 0.20 s translation freshness bound blocked 10 cm steps, and a NavMesh walk reported `RUNNING` then `FAILED` next frame: the planner consumes observed completion, not inferred reachability (V82-V118).

## 5. Routes, viewpoints and return

- **The planning unit is a reachable known-free observation pose at a standoff from unknown space,** not a frontier cell. Score a viewpoint by what it shows ahead along the remaining route. A 2-D map cannot predict the next target view; keeping the owner-confirmed crop as a bundle of 3-D view rays supports more viewpoints (V22-V81).
- **Validate the room before scoring.** The route scorer once omitted three small props the sensors could see; with them, neither start had a 5 or 6.5 cm path, so the corridor was disconnected, not unreachable (V114-V135). Include floor props, walls and furniture in the feasibility witness, and keep evaluator route witnesses out of the actor.
- **Goals must satisfy the stop condition.** A 40 cm planner annulus against a 37 cm stop band ended 1.39 cm outside it (V181-V183).
- **Independent splits** (frozen before exposure): 0/4, 0/3, 1/3, 0/3. Failures were rejected turns, scan ambiguity, partial-view identity loss, and a 16.5 cm route margin that left 1.15 mm of slack over the 11.5 cm stop tail (V189, V210).
- **Return after walking away** (RoboCasa, V599-V600):

| Item at return | Pose source | Final XY error | Result |
|---|---|---:|---|
| present | command only | 5.30 cm | fresh visual candidate, metric location withheld |
| present | range scan | 1.11 cm | metric re-find (item error 2.23 cm) |
| present | RGB-D | 4.39 cm | metric re-find (item error 3.63 cm) |
| removed | any | 1.8-5.4 cm | not refound; ask the owner |

  A remembered point says where to look, not where the item is.
- **Station choice.** A person-grounded station 0.8 m from the observed person found the can in 8 route steps; a fixed route took 58 steps and saw no can pixels (V694-V706). A fixed 30 cm retreat kept the find but put the ape in a prop's crossing corridor (modelled overlaps up to 32 cm); choosing station, refuge and duration together over the whole interaction is the open design (V680-V693).

## 6. Near-body stopping and moving hazards

What a software monitor can and cannot do; none of it qualifies a body.

- **The guard must run across moves, waits and looks.** A decision-step range alarm acted too late (clearance 7.8 to 0.9 cm within 0.4 s), and a 2.81 cm near contact happened while the planner waited in place (V15-V20).
- **Stopping is not safety.** A 20 ms range stop halted at 11.8 cm from an intruding prop, yet **all five held stop poses were later struck** by the still-moving prop. Of 48 ideal straight escapes up to 1.0 m/s, one kept the margin. There is no native escape skill (V423-V430).
- **The stop tail sets the margin.** The native forward policy travelled 9.6-11.5 cm after a zero command, 1.14-1.68 s from command to observed stop. That dwarfs a 9 cm point rule.
- **A scalar near-range threshold cannot represent path and motion.** At 65 cm all pets cleared but a crossing person caused four needless stops. A constant-velocity closest-approach tracker missed three of four slow side entries. A full-scan guard with a prior room map gave 0/4 overlaps and 0/4 needless stops across crossings, pauses and speed changes, but only with coverage and registration (2026-10-12).
- **Track every anonymous surface, not only the nearest.** Tracking all contiguous clusters moved an alarm 0.30 s earlier and kept 8.5 cm to the prop (V61-V65). Persistent ids staged 0.8 s earlier with two props; a reused id made a new prop look slow and sent the ape into its corridor.
- **An asynchronous image calculation must not own the lifetime of a motion response.** When the image worker finished after the south exit had started, the held pose was hit (-30.8 cm modelled overlap); a raw stop needs a fresh route or hold decision (V727-V743). Better-timed depth with an unsupported route made it worse (0.81 cm). Admit the owner station and a reachable continuation together before the show starts, and re-evaluate the reached pose after every guard stop (V763-V770).
- **Replanning needs hysteresis.** Fresh 10 Hz replanning gave 13.3 cm but changed goals six times; three successive risk observations gave 6.5 cm and no wander, at the cost of one packet of reaction (V744-V762).
- **Creeping obstacles break rules silently.** A cube moving 2 cm per capture entered the eye proxy and ended 4.3 cm away with no rule violated; ego-compensated two-frame closure triggered retreats and kept 17-18 cm, but ended 1.7 m from the owner. A time-indexed occupancy forecast and a recovery objective are needed (V142-V147).

Architecture consequences: two independently health-checked range roles (timely scene and motion tracking for decisions; a fast local stop monitor in every phase); forecast through the held footprint and response corridors at the sensor, command and stop latency; state when no refuge is supportable; a held pose is not safe because the ape stopped.

## 7. Sensor roles for behaviour

| Function | Current allocation |
|---|---|
| Owner attention, taught identity | RGB plus owner confirmation; ask at the safe stop, then a fresh image |
| 3-D furniture and upper body | head or chest RGB-D with calibrated self-masking; unseen space never free |
| Floor object location | RGB ray to a known floor with support-plane evidence |
| Room-scale navigation | 5 m-class scan with a growing map; a low plane (about 5 cm) for floor clutter, a mid plane (about 30 cm) for structure and owner motion; fused 5+30 cm best |
| Fast near-body stop | a separate 10-20 ms stream during moves, waits and looks; a typical spinning lidar scans at about 10 Hz, so it cannot claim a 20 ms guard |
| Localization | gyro and encoders plus range registration, motion-checked RGB-D fusion, heading-innovation gate, pose health in every claim |
| Touch, force, audio | reserved interfaces; no behaviour result |

## 8. Open questions

1. One body-independent policy trained on joined whole-task recovery states across new capture-disjoint, appearance-qualified rooms with sensor and timing faults, against frozen V6, the authored machine and an always-hold baseline.
2. Joint choice of owner station, reachable refuge and interaction duration from the real sensor and skill response envelope, before the show starts.
3. A continuation when no refuge is supportable: physical continuation or an early owner pause or reposition request. The honest help state works; moving-scene teach-and-refind has not been shown.
4. Real timing on the selected body: native stop path from today's controller (`control/wbc/`), cancel latency, sensor age and pose error.
5. A real owner: speech, arm motion, timing, a camera at the real aperture, distractors, other people and lighting.
6. Multi-person identity from persistent anonymous tracks with owner-specific evidence, not spatial association.
7. Candidate-local depth quality in place of whole-frame central-depth checks.

## Evidence index

- Code: `control/behaviour/` (`options/`, `route/`); harness `live_option_episode` in tag `history/perception-runners`; option-head models under `artifacts/runs/ape-high-level-task-policy-20261016`.
- Idle: `T-rsjape`, `T-gqgzz4` (loop design), `T-pjrh73` (first-question scope), `T-j6rhvd` (belief under motion), `T-h7pn8x`, `T-5ytcbv` (social cues, ablations). Decision `D-sznrsa`.
- Artifacts: `artifacts/ape-closed-loop-cove50-v<n>/` (V-ids above), `artifacts/ape-real-room-next-captures-20261014/`, `assets/real_indoor_scenes/learner_option_pilot_20261016/`, `artifacts/ape-active-find-50cm-health-20261006/`.
