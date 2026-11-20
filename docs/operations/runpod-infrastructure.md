# Runpod infrastructure

The new account is managed from `infra/compute/runpod/main.tf` using the official
`runpod/runpod` Terraform provider, pinned to `2.0.0-rc.3` with its checksum lock
file. The bootstrap owns a 10 GB STANDARD transport volume in EU-RO-1 and a
private 100 GB training template. A template reserves no GPU or container disk.
The image is pinned to the Linux amd64 manifest of
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`; it is a base image, not the
verified R3 simulator dependency closure.

Use the isolated client environment from the repository root:

```sh
.venv-runpod/bin/python infra/compute/runpod_infra.py init
.venv-runpod/bin/python infra/compute/runpod_infra.py validate
.venv-runpod/bin/python infra/compute/runpod_infra.py plan
# Read the complete saved plan before applying it.
.venv-runpod/bin/python infra/compute/runpod_infra.py apply
.venv-runpod/bin/python infra/compute/runpod_infra.py output
```

Plan and apply verify the authenticated account against `infra/compute/runpod/account.json`
(Git-ignored; create it from `account.example.json`).
The launcher passes the existing CLI credential through the child environment;
keys must never enter Terraform configuration, tfvars, Git or command arguments.
Authenticate with `.venv-runpod/bin/flash login` when needed. Flash saves a
`[default].api_key`; runpodctl also needs the same key as the top-level `apikey`.

Terraform state and saved plans are local, ignored by Git. Retain state: it is
the ownership record, and losing it can lead to duplicate resources. Copy it
securely when moving machines; do not commit it. The volume has `prevent_destroy`
because it can hold original episode evidence. Review retained evidence before
deliberately removing that protection or destroying storage.

Apply and output regenerate the Git-ignored `infra/compute/serverless/storage_config.py` from Terraform
outputs; apply also removes its consumed saved plan. The generated configuration is an
active worker input, included by Flash's native Python module bundler. The worker's
default S3 profile is `runpod-robot-lab-new`; configure the **new account's S3
keys** locally with `aws configure --profile runpod-robot-lab-new`. S3 keys are
separate from the account API key. The old `runpod` S3 profile is not selected
by current submitters. Worker S3 operations make one attempt, with no SDK retries.

The bootstrap creates no GPU Pod or serverless endpoint by default. The optional
`walking-trial.tf` creates a temporary endpoint with a digest-pinned Flash image,
zero minimum / one maximum worker, five-second idle tail and a 30-minute execution
deadline by default. A substantive continuation can explicitly select the bounded
one-hour deadline with `TF_VAR_walking_trial_timeout_ms=3600000` for plan and apply.
A longer reviewed job sets its own deadline, for example 75 minutes with
`TF_VAR_walking_trial_timeout_ms=4500000`, and a separate job TTL. It stays one
explicit job with no automatic extension.
When 4090 capacity is unavailable, a reviewed trial may also select
`TF_VAR_walking_trial_ampere24_fallback=true`, allowing the 24 GB L4/A5000/3090
pool alongside the 4090 pool. This changes eligible hardware, not the one-worker
cap or job budget. Record the actual GPU and measured throughput; a slower worker
may yield a partial fit under the same deadline. A trial that needs more
simulated worlds than 24 GB holds (4,096 R3 worlds at 0.25 ms physics did not
fit), or whose 24 GB pool has no capacity, may select
`TF_VAR_walking_trial_large_gpu=true`: the H100 80 GB pool with an A100 80 GB
fallback, still one worker. A job that stays `IN_QUEUE` with no worker in any
state (endpoint `health`) may mean an empty account balance rather than GPU
stock: check `myself { clientBalance }` first; topping up is the account
owner's action. Keep all selected variables for
subsequent plans while that trial is active. Review an in-place plan before applying;
do not replace an endpoint with an outstanding job.
Enable it explicitly for the foreground trial:

```sh
TF_VAR_walking_trial_enabled=true .venv-runpod/bin/python infra/compute/runpod_infra.py plan
TF_VAR_walking_trial_enabled=true .venv-runpod/bin/python infra/compute/runpod_infra.py apply
```

Keep that variable true for any plan while a trial is active. After the job is
terminal and its evidence is restored, run the ordinary plan/apply commands with
the variable unset to remove only the temporary endpoint. Read the deletion plan;
the persistent volume and template remain. No Flash process manages this endpoint.
`control/common/compute/launch.py` (and `control/wbc/compute/launch.py`) submits its
native Flash FunctionRequest exactly once; status is an explicit single request, with
at least five minutes between invocations. Do not resubmit a queued or running job.
The pinned worker needs cloudpickle-encoded arguments; the client uses the SDK's
native serialization helper, not JSON arguments.

`control/common/compute/launch_fit.py` submits the actor-and-value continuation. It shares a
five-minute minimum interval between explicit status and storage-progress reads,
streams staged native episode packs to disk, checks their SHA-256 identities,
and restores complete traces and compatible actor/critic/Adam checkpoints.
Workers push progress at completed stages; no process polls the provider in the
background.

For a different runtime, use bounded serverless jobs, or a Pod with a verified
provider-side expiry. Do not start a persistent Pod without that cost bound.
Do not mix Flash lifecycle management and Terraform ownership of the same endpoint.

Never install or restart background Runpod watchdogs, launchd services, cron
jobs, detached provider-polling loops or indefinite retries. The launcher uses
one foreground operation, a local exclusion lock, serial Terraform resource
operations and a five-minute command deadline. It does not retry a failed
operation. Stop on 429 and honor the reset window before another invocation.
The pinned provider's upstream SDK can internally attempt an HTTP request up
to four times; this is bounded, but its retry behavior is not configurable by
this provider. Do not describe it as a strict one-request transport. Do not
automate repeated plan/apply invocations. A timeout can follow a completed
mutation: inspect state before another apply.

The installed `runpodctl 3.0.1` uses `pod delete`. A `pod stop` releases
compute and retains the pod volume disk; ephemeral container-disk contents
do not survive stopping. The obsolete `pod terminate` invocation can print group help
with exit code zero; that is not evidence of deletion. Require a structured
response matching the requested pod ID and terminal desired status; reject
empty/error JSON. Record deletion as requested until a separate foreground
provider observation establishes completion. Do not allocate through a direct
SSH launcher: it has no provider-side expiry. New work uses the bounded
infrastructure above.

Sources: [official provider](https://registry.terraform.io/providers/runpod/runpod/2.0.0-rc.3/docs),
[agent guide](https://registry.terraform.io/providers/runpod/runpod/2.0.0-rc.3/docs/guides/agent-setup),
[SDK retry implementation](https://github.com/runpod/go-sdk/blob/v2.0.0/transport/retry.go).
