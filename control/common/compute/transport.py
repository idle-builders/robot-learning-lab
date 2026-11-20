"""Deterministic uncompressed transport; native NPZ already compresses its bytes."""
import hashlib,tarfile
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def pack(root,paths,target):
    root=Path(root).resolve();files={}
    for path in paths:
        path=Path(path)
        for entry in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if entry.is_symlink():raise ValueError('Symlink in evidence pack')
            if entry.is_file():files[str(entry.resolve().relative_to(root))]=entry
    rows=[]
    with tarfile.open(target,'w',format=tarfile.USTAR_FORMAT) as archive:
        for name,path in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o644
            with path.open('rb') as stream:archive.addfile(info,stream)
            rows.append(dict(path=name,bytes=info.size,sha256=digest(path)))
    return dict(sha256=digest(target),bytes=Path(target).stat().st_size,files=rows)
