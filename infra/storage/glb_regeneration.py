"""Exact viewer GLB composition from retained meshes and original layout metadata.

No CAD, remeshing or physical simulation. Every binary chunk and complete output
must reproduce its recorded SHA-256 before publication.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def rebuild(record, destination):
    from infra.storage import blob_store
    from infra.storage.transient_assets import materialized_input
    import trimesh

    if blob_store.digest(Path(__file__)) != record['regenerator_sha256']:
        raise ValueError('GLB regeneration source changed')
    template = ROOT / record['layout_template']
    if not template.resolve().is_relative_to(ROOT) or blob_store.digest(template) != record['layout_template_sha256']:
        raise ValueError('GLB layout template changed')
    with gzip.open(template, 'rb') as stream:
        layout = stream.read()
    metadata = json.loads(layout)
    meshes = {}
    inputs = []
    for name, expected in record['mesh_inputs'].items():
        original = blob_store.safe_path(name)
        with materialized_input(original) as path:
            before = blob_store.identity(path.stat())
            if blob_store.digest(path) != expected:
                raise ValueError('GLB mesh input changed: ' + name)
            scene = trimesh.load(path, force='scene', process=False)
            meshes.update(scene.geometry)
            if blob_store.identity(path.stat()) != before:
                raise ValueError('GLB mesh input changed while reading: ' + name)
            if path == original:
                inputs.append((path, before))
    output = trimesh.Scene()
    for mesh in metadata['meshes']:
        name = mesh['name']
        if name not in meshes:
            raise ValueError('Missing retained mesh: ' + name)
        output.add_geometry(meshes[name], node_name=name, geom_name=name)
    encoded = output.export(file_type='glb')
    length, kind = struct.unpack_from('<II', encoded, 12)
    if kind != 0x4e4f534a:
        raise ValueError('Unexpected GLB layout')
    binary = encoded[20 + length:]
    if hashlib.sha256(binary).hexdigest() != record['binary_chunk_sha256']:
        raise ValueError('Regenerated GLB geometry bytes differ')
    size = 20 + len(layout) + len(binary)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.glb', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(struct.pack('<III', 0x46546c67, 2, size))
            stream.write(struct.pack('<II', len(layout), 0x4e4f534a))
            stream.write(layout)
            stream.write(binary)
        if (size != record['bytes'] or blob_store.digest(temporary) != record['sha256']
                or any(blob_store.identity(path.stat()) != before for path, before in inputs)):
            raise ValueError('GLB output or retained input differs')
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
