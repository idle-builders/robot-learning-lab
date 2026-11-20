# Evidence and evaluation for ape experiments

Use when designing a comparison, interpreting a claimed improvement, or deciding whether to retain a policy. This reference contains the decisions that recur in this repository. The project rule is in [evaluation principles](../../../../docs/design/evaluation-principles.md): no pass/fail; clean metrics, a written judgement and 0–5 distance-to-viable grades per dimension, with confidence and trend.

## Separate four kinds of statement

| Kind | Destination | Example |
|---|---|---|
| Established method, with its assumptions | Relevant skill reference | Paired comparisons preserve episode pairing; a transform needs a declared direction. |
| Observed repository result | Dated evidence doc and policy/run record | An actor maintained contact on one named body and start distribution. |
| Unproven benefit for this ape | Existing Idle task, or a proposed experiment | A recurrent actor might tolerate timing variation better on the selected body. |
| Unsupported, obsolete or commercially reported claim | Audit correction/context, with provenance | A vendor's deployment total is not independently measured autonomous uptime. |

A paper can establish a method without proving that it improves this robot. Conversely, a local failed trial does not disprove a method in general. Record source version/date, tested embodiment, task and input assumptions. Preparation notes are a finding aid, not a specification. Verify fast-moving model/version claims in official papers, model cards and code before adoption; “replaced,” “standard everywhere” and “physics-correct” are usually too broad.

## Count the unit that can fail independently

Define an attempt before looking at results: an independently reset episode, held-out scene, object instance, owner session or continuous mission. Hundreds of frames from one rollout are not hundreds of independent successes. Report per-condition numerator/denominator, seeds, supported command/start distribution and failures. Keep training seeds, development cases used to tune decision thresholds or reference values, and final evaluation separate. A changed camera, correction policy or tuned decision threshold can leak evaluation data just as training can.

For event counts such as falls or drops, use an interval that behaves at small counts and near zero/one; the symmetric normal/Wald interval can be misleading. Wilson or exact binomial intervals are useful choices with stated conventions. With zero failures in `n` independent trials, the one-sided 95% exact upper failure bound solves `(1-p)^n = .05`: `p = 1 - .05**(1/n)`. Thus 0/5 failures still permits about 45% failure, and 0/100 about 3%. This derivation assumes exchangeable independent trials; it is not a bound for arbitrary unseen environments. [NIST confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

For matched policy or sensor comparisons, estimate the *difference* on matched episodes and preserve pairing. In SciPy bootstrap, `paired=True` resamples a shared index; it does not infer which observations belong to an independent episode. Aggregate or resample complete scene/session/trajectory units where that is the independent unit. Degenerate all-identical samples can make BCa intervals undefined. Report this rather than substituting a zero-width certainty claim. [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

Declare the improvement or regression size you would consider meaningful before the comparison, as a reference point to grade against rather than a cutoff. Failure to detect a difference is not evidence of equivalence. Report uncertainty and any changed sampling/stopping rule. More GPU environments improve data throughput; they do not make a fixed narrow scene distribution representative of the intended robot experience.

## Match metrics to the claim

- **Control:** commanded displacement/direction, turns and stopping; falls, slip, saturation and contact outcomes. Survival alone can reward standing still or walking backwards. Reward is a diagnostic, not the success definition.
- **Manipulation:** acquisition from a genuinely unheld state, supported lift/hold, placement error, release and settling. Track contact/possession throughout, including failed attempts. Pose imitation alone can miss the object.
- **Perception and teaching:** useful correct claims, wrong confident claims, abstention, unnecessary questions, correction retention and time to resolution. Retrieval similarity is not a calibrated probability; AP/IoU is not instance identity or possession.
- **State estimation:** pose/error convention, alignment and scale, drift, resets and time without usable pose. Low aligned trajectory error can hide an unavailable metric scale or discontinuities that invalidate motion targets.
- **Runtime:** capture-to-application age, inference time, queue wait, deadline misses, stale-sample drops and recovery, with sample counts. A p99 estimated from a handful of samples is weak evidence. Paused-physics inference is no evidence for an uninterrupted controller.
- **Composition:** complete episode success, conditional transition failures, interventions and recovery cost. Multiplying isolated success rates assumes conditions that skill handoffs often violate; directly measure the actual sequence.

## Grade, and challenge the requirement

After the metrics, describe the behaviour in words (what it attempted, how and where it degraded, the severe cases) and grade each dimension separately from 0 (unknown) to 5 (viable, only physical tests left), with confidence (low: estimate, medium: model or simulation, high: reproduced or measured) and trend. Reference values from earlier gates, papers or reference robots are points to grade against, never cutoffs. When evidence shows a gap, ask both questions: is the solution wrong, or is the requirement wrong? Record where the number came from, what it buys and what it costs, and put proposed requirement changes to the user. Reviewers grade and explain; they do not issue verdicts.

## Retain the evidence needed to disagree

Use the existing artifact store and manifests: exact code/body/config/checkpoint/data identities, declared actor inputs, independent outcome scoring, synchronized video and traces, and failed as well as successful episodes. Keep the original run immutable; corrections name the prior record. Export/import success establishes packaging only. A simulator, world model or learned reward may be useful for triage, but its ranking of candidates needs independent outcome validation before it is trusted as evidence.

Prefer a bounded existing harness over building a fleet data platform for this POC. Dataset mixtures, model families, recurrent state, distillation, tactile sensing and latency compensation remain choices to test when an observed failure gives them a purpose. Do not add one task for every interview term.
