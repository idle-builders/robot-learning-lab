"""Actual extracted21routes,CPUlesson fixture,and interrupted-input wrapper."""
from common import project

_source = project('control/skills/forward-stop-learning/compute/prove.py', '1098419a9721d35e2153e4fdf620b8b4aa660a3ef646c6e8ac666cadf94ce6ad', [('control/skills/forward-stop-learning', 'control/skills/forward-heading-correction', 3), ('data/forward-stop-learning-20261030', 'data/forward-heading-correction-20261030', 1)])
_old="    receipt=dict(schema="
_new="    parallel_output=work/'parallel-parent.json'\n    subprocess.run([sys.executable,'-I','-B',str(extracted/'control/skills/forward-heading-correction/compute/prove_parallel.py'),'--out',str(parallel_output)],cwd=extracted,env=env,check=True)\n    paired_fixture=json.loads(parallel_output.read_text())\n    receipt=dict(actual_extracted_paired_parent_fixture=paired_fixture,schema="
if _source.count(_old)!=1:raise ValueError('Exact proof receipt differs')
_source=_source.replace(_old,_new)
exec(compile(_source, __file__+':retained-projection', 'exec'), globals())
