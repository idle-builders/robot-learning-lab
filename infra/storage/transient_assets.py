"""Read cold evidence in request-scoped temporary storage, without warming runs.

With a SHA-256, the exact registered content is returned: the local file when it
matches, else that version from the blob index (current or earlier), else from Git
history. Paths are edited in place, so an older dashboard version needs this.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _from_git(name, sha256, destination, limit=200):
    """Write the committed blob of NAME whose SHA-256 matches; False if none does."""
    try:
        commits = subprocess.run(["git", "log", f"-n{limit}", "--format=%H", "--", name], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return False
    for commit in commits:
        blob = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT, capture_output=True)
        if blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == sha256:
            Path(destination).write_bytes(blob.stdout)
            return True
    return False


@contextmanager
def materialized_input(path, sha256=None, registered=()):
    """REGISTERED: SHA-256 values a catalog records for this path; tried in Git history
    when the path is neither on disk nor in the blob index (history out of the tree)."""
    from infra.storage import blob_store as bs
    from infra.storage.regenerable_artifacts import lookup, rebuild
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.parent.resolve().is_relative_to(ROOT):
        raise ValueError('Asset escapes workspace')
    if path.is_file() and (sha256 is None or _digest(path) == sha256):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Asset link escapes workspace')
        yield path
        return
    name = path.relative_to(ROOT).as_posix()
    record = lookup(path) if sha256 is None else None
    row = bs.lookup(path, sha256)
    if not record and not row and bs.moved(path) is not None:
        row = bs.lookup(bs.moved(path), sha256)
    if not record and not row:
        # the same bytes stored under another path (content-addressed copy or an earlier name)
        row = next((found for found in map(bs.lookup_sha, [sha256] if sha256 else registered) if found), None)
    cache = ROOT / '.cache/asset-requests'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='asset-', dir=cache) as scratch:
        destination = Path(scratch) / path.name
        try:
            if record:
                rebuild(record, destination)
            elif row:
                bs.restore_entry(row, destination=destination)
            elif sha256 is not None:
                if not _from_git(name, sha256, destination):
                    raise FileNotFoundError(path)
            elif not any(_from_git(name, known, destination) for known in registered):
                raise FileNotFoundError(path)
        except FileNotFoundError:
            raise
        except Exception as error:
            raise bs.BlobUnavailable('Cold asset could not be prepared: ' + name) from error
        yield destination
