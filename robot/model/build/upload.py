"""Use the existing verified clip uploader for this result only."""
import importlib.util,sqlite3,sys
from common import *

class ScopedConnection(sqlite3.Connection):
    def execute(self,sql,parameters=(),/):
        if sql=="SELECT * FROM clips WHERE verified_at IS NULL ORDER BY result_id,id":
            return super().execute('SELECT * FROM clips WHERE verified_at IS NULL AND result_id=? ORDER BY result_id,id',(RESULT,))
        return super().execute(sql,parameters)

def main():
    sys.path.insert(0,str(ROOT/'infra'/'dashboard'))
    spec=importlib.util.spec_from_file_location('existing_results_hub',ROOT/'infra/dashboard/results_hub.py');hub=importlib.util.module_from_spec(spec);spec.loader.exec_module(hub)
    db=sqlite3.connect(ROOT/'data/results.sqlite',factory=ScopedConnection);db.row_factory=sqlite3.Row
    hub.upload(db)
    rows=[dict(r)for r in db.execute('select id,sha256,bytes,bucket_key,verified_at from clips where result_id=? order by id',(RESULT,))]
    write(DATA/'video-upload-receipt.json',dict(result=RESULT,clips=rows,verification='Existing results_hub.upload full remote readback SHA/bytes; selection restricted to this result.'))

if __name__=='__main__':main()
