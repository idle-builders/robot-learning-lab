"""Restore optional pinned tools after the lean-checkout cleanup.

.venv/bin/python -B infra/storage/restore_native_tool.py kicad
.venv/bin/python -B infra/storage/restore_native_tool.py arm
.venv/bin/python -B infra/storage/restore_native_tool.py bambu
.venv/bin/python -B infra/storage/restore_native_tool.py prusa
"""
import argparse,hashlib,json,shutil,subprocess,tarfile,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RECORDS=ROOT/'data/storage-cleanup-20261029/lean'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('tool',choices=['kicad','arm','bambu','prusa']);a=p.parse_args()
native=json.loads((RECORDS/'native-tool-cleanup.json').read_text())
installers=json.loads((RECORDS/'installer-cleanup.json').read_text())
row=(native[0]if a.tool=='kicad'else native[1])if a.tool in {'kicad','arm'}else next(x for x in installers if a.tool in x['path'])
url=row.get('installer_url',row.get('download_url'));expected=row.get('installer_sha256',row.get('sha256'));target=ROOT/row['path']
if target.exists():raise SystemExit('Tool or installer already exists: '+str(target))
cache=ROOT/'.cache/tool-install';cache.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix=a.tool+'-',dir=cache)as scratch:
 scratch=Path(scratch);archive=scratch/Path(url).name;h=hashlib.sha256()
 with urllib.request.urlopen(url,timeout=120)as source,archive.open('xb')as output:
  while block:=source.read(1<<20):h.update(block);output.write(block)
 if h.hexdigest()!=expected:raise ValueError('Installer SHA-256 differs')
 if a.tool=='arm':
  destination=scratch/'unpacked';destination.mkdir()
  with tarfile.open(archive)as tar:tar.extractall(destination,filter='data')
  source=destination/target.name
  if not source.is_dir():raise ValueError('Unexpected toolchain archive layout')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),target)
 elif a.tool=='kicad':
  mount=scratch/'mounted';mount.mkdir()
  subprocess.run(['hdiutil','attach',str(archive),'-nobrowse','-readonly','-mountpoint',str(mount)],check=True)
  try:
   source=next((x for x in [mount/'KiCad.app',mount/'KiCad/KiCad.app',*mount.glob('*/KiCad.app')]if x.is_dir()),None)
   if source is None:raise ValueError('KiCad app missing from verified image')
   installed=scratch/'installed';installed.mkdir()
   shutil.copytree(source,installed/'KiCad.app',symlinks=True)
  finally:subprocess.run(['hdiutil','detach',str(mount)],check=True)
  target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(installed),target)
 else:
  target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(archive),target)
print('Restored',target)
