"""Retain logical artifact names while checking regular files or verified CAS links."""
from pathlib import Path
from .artifact_store import validate_file


def checked_path(folder, relative, *, repo_root=None):
    """Constrain a report key to its tree; allow a verified store link at the leaf."""
    folder = Path(folder).absolute()
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts or not name.parts or '\\' in str(relative):
        raise ValueError(f'unsafe artifact path: {relative}')
    path = folder / name
    # Directory links cannot redirect an otherwise innocuous relative key.
    if not path.parent.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f'artifact parent escapes tree: {relative}')
    if path.is_symlink():
        validate_file(path, repo_root=repo_root)
    elif not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f'artifact escapes tree: {relative}')
    if not path.is_file():
        raise ValueError(f'missing artifact: {path}')
    return path
