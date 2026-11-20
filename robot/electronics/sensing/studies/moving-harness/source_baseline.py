"""Retain camera smooth-shank occupancy already present in the exact source screws."""
from pathlib import Path
import hashlib,importlib.util,json
from build123d import import_step
ROOT=Path(__file__).resolve().parents[5];OUT=ROOT/'artifacts/cove50-sensor-integration/v7'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 spec=importlib.util.spec_from_file_location('mount',ROOT/'robot/electronics/sensing/studies/moving-harness/build_mounts.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);context=json.loads((ROOT/'artifacts/cove50-sensor-integration/v6/assembly-parts.json').read_text());index={p['name']:p for p in context['parts']};camera=ROOT/index['upper__OAK_D_Lite_native']['native_step'];cam=import_step(camera);rows=[]
 for name in ['upper__camera_M4x8_0','upper__camera_M4x8_1']:
  path=ROOT/index[name]['native_step'];s=import_step(path);rows.append(dict(source_part=name,native_step=str(path.relative_to(ROOT)),native_step_sha256=sha(path),camera_native_step=str(camera.relative_to(ROOT)),camera_native_step_sha256=sha(camera),source_screw_camera_overlap_mm3=m.local_common(s,cam),source_bounds_mm=m.native.bounds(s).tolist(),interpretation='Inherited nominal smooth-shank/source-camera material relation. No actual thread engagement, wall thickness, torque or physical interference qualification.'))
 (OUT/'source-fastener-baseline.json').write_text(json.dumps(dict(observations=rows,producer=str(Path(__file__).relative_to(ROOT)),producer_sha256=sha(Path(__file__))),indent=2)+'\n');print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
