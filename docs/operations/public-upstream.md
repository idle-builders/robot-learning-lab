# Public repository and private evidence

This checkout is the public repository `idle-builders/robot-learning-lab` (remote `origin`). Everything tracked in Git is public. Private material stays in Git-ignored local files and the private archive bucket.

## How it was connected (2026-11-20)

Until 2026-11-20 the lab kept a private Git history (850 commits) and published a separately exported snapshot (`.public-upstream/`, 2026-10-23, now obsolete). On 2026-11-20 the cleaned current tree became a fresh root commit, this checkout's `main` was moved to it, and it was force-pushed over the public `main`, replacing the old snapshot. From then on there is one history and an ordinary `git push`.

The private history before that is not public and is never pushed:

- it is kept locally as the tags `archive/private-main-20261120`, `design/*` and `history/*`, and as the bundle `../robot-learning-lab-private-history-20261120.bundle` (outside the repository; restore with `git fetch <bundle> 'refs/*:refs/private/*'`);
- `.githooks/pre-push` refuses any ref whose history reaches the private root commit `9bf48951f`, so `git push --tags` or an old branch cannot leak it.

Gitleaks over the whole private history (850 commits, 1.47 GB of diffs, 2026-11-20) found 1,433 generic-api-key candidates and no provider-specific credential. Every one inspected is an identifier: artifact `output_key` names, stage keys, run names, hashes and public reCAPTCHA site keys in captured web pages. The narrow allowlist for those identifier fields is `.gitleaks.toml`.

## What goes where

| Material | Destination |
| --- | --- |
| Authored code, training definitions, robot configuration and CAD sources, electronics, dashboard UI, agent skills, operations and research docs | Git (public) |
| `data/` (results and blob catalogs, the catalog pointer, spend ledger, run and build records), Runpod/Infomaniak transfer journals and manifests, personal study notes (`docs/learning materials/`), the Runpod account id (`infra/compute/runpod/account.json`) | Git-ignored local files; `data/` is backed up to the private bucket by `infra/storage/catalog_backup.py` |
| Archive bucket, endpoint and Keychain prefix | Git-ignored `data/private-archive.json` (or `ROBOTLAB_ARCHIVE_*` environment); docs write `<archive-bucket>` / `<archive-endpoint>` |
| Remote dashboard host and its login | Git-ignored `.credentials/remote-dashboard.md` |
| Selected episodes, checkpoints, CAD exports, native reports and traces | The private evidence archive, by path and hash |
| API keys, passwords, AWS/session credentials, filled-in environment files | Keychain, the provider CLI profile or the shell environment; never Git, artifacts or logs |
| Expiring signed URLs | Generated on demand; never committed |

Logical asset paths and SHA-256 values are provenance, not access credentials. Public users configure their own archive with local AWS environment credentials or a profile; copy `.env.example` to `.env` only when needed and load it into the shell explicitly.

## Guards

- `infra/config/publication/public-source-policy.json` lists what may be tracked; `python3 infra/tools/public_source.py check` refuses a tracked file outside it (private roots, payload types, unreviewed binaries). Binaries are allowed only by exact path in `exact_assets`.
- `.githooks/pre-push` runs that check, the private-history check and Gitleaks over exactly the commits being pushed. Enable the hooks once per checkout: `git config core.hooksPath .githooks`. Install Gitleaks with `brew install gitleaks`.
- `.github/workflows/public-source.yml` repeats the check and a pinned Gitleaks scan of the whole public history on every push and pull request. GitHub secret scanning and push protection are enabled on the repository.

Before adding a new kind of file, ask whether it would be fine on the public internet. If a real credential ever reaches a commit, rotate or revoke it first; editing a later commit does not remove the earlier copy (see [GitHub's sensitive-data removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)).

## Archive configuration

`infra/storage/archive_config.py` is shared by the results hub and cold blob storage. The public clone has no bucket configured. The lab's destination and Keychain service prefix live in ignored, owner-readable `data/private-archive.json` (configuration names, not credential values). `ROBOTLAB_ARCHIVE_BUCKET`, `ROBOTLAB_ARCHIVE_ENDPOINT` and `ROBOTLAB_ARCHIVE_REGION` override them. Both AWS access and secret environment values must be supplied together; temporary sessions also support `AWS_SESSION_TOKEN`. The local dashboard can mint signed URLs and restore private bytes when configured, so keep it bound to `127.0.0.1`.

The legacy `infra/compute/serverless/s3_artifacts.py` transport uses a Runpod volume and short-lived job transport. Its `storage_auth` must not enter manifests, reports or retained job logs.
