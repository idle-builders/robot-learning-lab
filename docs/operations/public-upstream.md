# Public upstream and private evidence

The public destination is `idle-builders/robot-learning-lab`, administered through `tobiagru`. Publish a reviewed source snapshot with fresh Git history. Keep the original lab branches, ongoing staged changes, private catalogs and recovery records in this workspace. A public source checkout is not a full lab backup or a replay bundle.

## What goes where

| Material | Destination |
| --- | --- |
| Authored reusable code, training definitions, robot configuration, firmware/electronics source, dashboard UI and setup docs | Public Git, from an exact reviewed inventory |
| Compact scientific records chosen for public release | Explicit later source-policy addition after content review |
| Selected full episodes, sensor data, checkpoints, CAD exports, native reports/traces and decisive failures | Existing private evidence archive; retain native paths, hashes and source/config provenance |
| Results/blob SQLite catalogs, full recovery manifests, working patches, original Git bundles, local preferences and storage inventories | Private local state and selected verified private backups |
| API keys, passwords, AWS/session credentials, filled-in environment files, Keychain entries | Local environment, credential store or scoped provider secret configuration; never Git or artifact payloads |
| Expiring signed playback/download URLs | Generated on demand; never source, receipts, logs or committed catalogs |
| Caches, duplicate renders and abandoned variants | Selective expiry under the existing retention workflow, after required remote hash verification |

Logical asset paths and SHA-256 values are provenance, not access credentials. A bucket name or provider endpoint is not itself a secret. Still keep the original lab's destination in ignored local configuration so a public clone has no implicit archive connection. Downloaded reference material and source mirrors are omitted from the first publication; their provenance and reuse rights need their own review.

## Archive configuration

`lab/archive_config.py` is shared by the results hub and cold blob storage. The public clone has no bucket configured. The local lab's existing destination and explicit Keychain service prefix live in ignored, owner-readable `data/private-archive.json`. It contains configuration names, not credential values. `ROBOTLAB_ARCHIVE_BUCKET`, `ROBOTLAB_ARCHIVE_ENDPOINT` and `ROBOTLAB_ARCHIVE_REGION` override the corresponding settings. Existing object prefixes and hashes are unchanged.

Public users configure their own archive and use local AWS environment credentials or a profile. Copy `.env.example` to `.env` only when needed and explicitly load it into the shell; the application does not parse dotenv files automatically. Never put a credential into either JSON configuration or the example file. Both AWS access and secret environment values must be supplied together; temporary sessions also support `AWS_SESSION_TOKEN`.

The legacy `workers/serverless/s3_artifacts.py` transport is separate: it uses a Runpod volume and short-lived job transport, with a historical destination identifier still in its reference configuration. Its `storage_auth` must not enter manifests, reports or retained job logs. It is not used by the results hub or cold evidence archive. Review that reference integration before deploying it for another account.

The existing archival workflow remains authoritative: publish compact results into the local database, explicitly upload selected clips, read back and verify full content hashes, archive selected stable native evidence, and restore exact inputs before reanalysis. This publication does not change retention or delete/upload evidence. The local dashboard can mint signed URLs and restore private bytes when configured, so keep it bound to `127.0.0.1`; it is not a public hosting service.

## Prepare a publication

The source policy is `publication/public-source-policy.json`. Inclusion globs select source candidates; additional exclusions omit private roots, payload types, captured source trees and generated output. The exact inventory is the review surface. New untracked source can be included, but staged changes are never consumed or altered: exports read current working-tree bytes and seal their hashes. Deleted files, symlinks, oversized source candidates and captured references are reported as omissions. Do not treat the policy as an automatic license or privacy review.

```sh
python3 tools/public_source.py plan .cache/publication-audit/public-inventory.json
# Review every destination/source path and hash in that inventory.
python3 tools/public_source.py export \
  .cache/publication-audit/public-inventory.json \
  .cache/publication-audit/public-repo
```

Use new paths on each attempt. The export refuses changed source bytes, private roots, arbitrary overlays and symlinks. It copies no `.git` directory and runs Gitleaks with fully redacted output before reporting completion. Reports stay beside the private inventory. Install Gitleaks locally (`brew install gitleaks` on this Mac). Review any findings in private; never publish raw secret reports or broad scanner suppressions. Secret scanning reduces exposure risk but cannot prove that every arbitrary credential or private fact is absent.

Inspect the resulting tree, verify its setup and check for active credential values and signed URLs before creating the first commit. Use a new repository with exactly one root commit. Do not copy old refs, use `git push --all`, mirror the private repository, or make the original history public. Old recovery bundles can contain historical sensitive material even when the current source is clean.

On GitHub, enable secret scanning and push protection. In the public checkout, run `git config core.hooksPath .githooks` to enable its pre-push source-boundary and Gitleaks checks over all refs. `.github/workflows/public-source.yml` repeats those checks with a pinned scanner and verified download hash, without archive credentials. Future publications must update the reviewed file inventory and repeat the source scan. Develop public changes in that clean history or copy reviewed updates into it; the private branch is not its push source. Keep the public export's remote separate from this mixed private working repository.

## Audit evidence, 2026-10-23

The original lab had no configured Git remote. `tobiagru` authentication and active administrator membership in `idle-builders` were verified without printing credentials. The current history contains 348 commits. There were over 30,000 tracked/untracked nonignored files, with substantial ongoing source and staged preservation work; none of those existing changes were reset, stashed or bulk-staged for publication.

Gitleaks 8.30.1 scanned all reachable local refs (348 commits, approximately 1.02 GB of diff content) and produced 457 generic API-key candidates. Inspected examples were source hashes, part identifiers and captured website form tokens. This count is a scanner inventory, not 457 confirmed credentials. All original history is excluded from the public snapshot regardless of those classifications.

A separate targeted working-tree scan examined 30,550 text files; 324 binary, linked or large files were outside that scan. It found four token-bearing GitBook image URLs in downloaded reference captures and a token-shaped substring in a SanDisk product URL. These captures are excluded from the public selection. No active provider credential was confirmed by that targeted scan. Generated payloads and ignored private state are outside the source publication audit; no claim is made that they are sanitized for public release.

Redacted native reports and exact inventories remain in ignored `.cache/publication-audit/`. Final export measurements and verification belong in the private publication receipt. If an actual credential is discovered, rotate or revoke it before distribution; editing a current file does not remove earlier copies. See [GitHub's sensitive-data removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

The first source candidate contained 4,034 files and approximately 35.45 MB. Its complete directory scan produced no Gitleaks findings; a separate byte-for-byte search for five locally available active credential values found no matches. Without AWS/GitHub/Runpod environment credentials or local archive configuration, its learning/design pages, empty results API, session API, design preferences and UI assets served successfully from a fresh database. The shared archive configuration also retrieved the existing private clip's metadata with the expected byte count; the bucket ACL had no public-group grants. This ACL observation is not a full policy audit. Training, native CAD replay and hardware operation were not exercised by the publication checks.
