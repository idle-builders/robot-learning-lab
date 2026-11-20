"""Retain the small sharp-hex proxy/spine-bend material witness without an engineering verdict."""
from pathlib import Path
import hashlib,importlib.util,json
from build123d import Box,Compound,import_step,export_step,export_stl
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.TopTools import TopTools_ListOfShape
ROOT=Path(__file__).resolve().parents[5];OUT=ROOT/'artifacts/cove50-sensor-integration/v7'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def intersect(a,b):
 aa,bb=TopTools_ListOfShape(),TopTools_ListOfShape();aa.Append(a.wrapped);bb.Append(b.wrapped);op=BRepAlgoAPI_Common();op.SetArguments(aa);op.SetTools(bb);op.SetNonDestructive(True);op.Build()
 if not op.IsDone():raise RuntimeError('Witness Common incomplete')
 return Compound.cast(op.Shape())
def main():
 spec=importlib.util.spec_from_file_location('native',ROOT/'robot/electronics/sensing/studies/imu-mount/build.py');n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n);np=OUT/'mount-parts/harness_fixed_mount_nut_1.step';sp=OUT/'mount-parts/harness_four_hole_spine.step';nut=import_step(np);spine=import_step(sp);b=n.bounds(nut);room=Box(*(b[1]-b[0])).translate((b[0]+b[1])/2);w=intersect(nut,intersect(spine,room));folder=OUT/'material-witnesses';folder.mkdir(exist_ok=True);p=folder/'fixed_nut_spine_corner.step';q=p.with_suffix('.stl');export_step(Compound(children=list(w.solids()),label='sharp_hex_corner_spine_witness'),p);export_stl(w,q,tolerance=.0005,angular_tolerance=.08);result=dict(pair=['harness_fixed_mount_nut_1','harness_four_hole_spine'],native_overlap_mm3=w.volume,native_bounds_mm=n.bounds(w).tolist(),native_solids=len(w.solids()),native_step=str(p.relative_to(ROOT)),mesh=str(q.relative_to(ROOT)),interpretation='Small overlap of a nominal unchamfered AF5 hex-prism nut proxy at the source spine bend. Actual nut corner/chamfer, bend radius, tolerance and seating remain unmeasured; no automatic verdict or inferred physical interference.',sources=[dict(path=str(s.relative_to(ROOT)),sha256=sha(s),bytes=s.stat().st_size)for s in[Path(__file__),np,sp]]);(OUT/'nut-corner-witness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
