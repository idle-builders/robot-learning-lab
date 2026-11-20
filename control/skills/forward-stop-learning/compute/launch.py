"""Explicit one-submit, sparse one-status, or durable restore; never a monitor."""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
import tarfile
import time
import uuid
from pathlib import Path
from common import BASE_LAUNCH, DATA, HERE, ROOT, SIDECAR, activation, load, read, ref, verify, write


def submit(evidence, work, release):
    plan=verify(manifest=False);authority=read(release);activation(plan,authority)
    receipt=evidence/'submission.json'
    if receipt.exists():raise FileExistsError('Original submission exists; never auto-resubmit')
    archive=work/'input.tar.gz';expected=read(DATA/'bundle-receipt.json')['archive']
    if (ref(archive)['sha256'],archive.stat().st_size)!=(expected['sha256'],expected['bytes']):raise ValueError('Frozen capsule differs')
    manifest=read(DATA/'bundle-manifest.json')
    if manifest['files']!=read(DATA/'input-index.json')['files']:raise ValueError('Final manifest/input index differs')
    required=[authority['independent_review'],ref(release)]
    if any(x not in manifest['files']for x in required):raise ValueError('Exact release/review must be packed')
    base=load(BASE_LAUNCH,'forward_original_compute_client');client,bucket,endpoint=base.s3()
    if not endpoint:raise PermissionError('Root must provision the reviewed endpoint before submit')
    prefix='robot-learning-lab/v1/forward-stop-learning/'+plan['job_name']+'/'+uuid.uuid4().hex
    input_key=prefix+'/input.tar.gz';output_key=prefix+'/evidence.tar';nonce=uuid.uuid4().hex
    binding=dict(schema='robot-learning-lab.worker-completion-binding.v1',submission_nonce=nonce,provider_job_id=None,
        endpoint=endpoint,bucket=bucket,output_prefix=prefix,output_key=output_key,completion_key=prefix+'/completion.json',
        input_key=input_key,input_archive=dict(sha256=expected['sha256'],bytes=expected['bytes']),
        input_manifest=ref(DATA/'bundle-manifest.json'),plan=ref(DATA/'worker-plan.json'),worker_source=ref(HERE/'worker.py'),launcher_source=ref(Path(__file__)))
    submitted=time.time();expiry=datetime.datetime.fromtimestamp(submitted+plan['provider_policy']['ttl']/1000,datetime.timezone.utc).isoformat()
    record=dict(endpoint=endpoint,bucket=bucket,input_key=input_key,output_key=output_key,output_prefix=prefix,
        bundle_sha256=expected['sha256'],completion_binding=binding,plan=plan,plan_sha256=ref(DATA/'worker-plan.json')['sha256'],
        plan_relative_path=ref(DATA/'worker-plan.json')['path'],submitted_at_epoch=submitted,
        submitted_at_utc=datetime.datetime.fromtimestamp(submitted,datetime.timezone.utc).isoformat(),absolute_expiry_utc=expiry,
        submission_state='input_upload_pending',release=ref(release))
    write(receipt,record) # Any ambiguous upload/submit outcome remains visible; no retry.
    with archive.open('rb')as stream:client.put_object(Bucket=bucket,Key=input_key,Body=stream)
    response=client.get_object(Bucket=bucket,Key=input_key)
    try:actual=hashlib.file_digest(response['Body'],'sha256').hexdigest()
    finally:response['Body'].close()
    if actual!=expected['sha256']:raise ValueError('Full input transport readback differs')
    sys.path.insert(0,str(ROOT))
    from infra.compute.serverless.s3_artifacts import CONFIG, profile_auth
    if (client.meta.endpoint_url.rstrip('/'),bucket)!=(CONFIG['endpoint'].rstrip('/'),CONFIG['bucket']):
        raise ValueError('Existing transport helper/storage profile differs')
    record.update(submission_state='request_pending',completion_storage_endpoint=CONFIG['endpoint']);write(receipt,record)
    data=dict(storage=dict(CONFIG),storage_auth=profile_auth(),plan=plan,activation=authority,
        helper_sources={k:(ROOT/v['path']).read_text()for k,v in plan['helpers'].items()},
        input_key=input_key,output_key=output_key,output_prefix=prefix,bundle_sha256=expected['sha256'],
        completion_binding=binding,completion_protocol=plan['completion_sidecar'],absolute_expiry_utc=expiry)
    from runpod_flash.runtime.serialization import serialize_kwargs
    worker_code=(HERE/'worker.py').read_text().split("\n\nif __name__ == '__main__':")[0]
    record['dispatched_function_sha256']=hashlib.sha256(worker_code.encode()).hexdigest();write(receipt,record)
    payload=dict(function_name='run_forward_stop',function_code=worker_code,kwargs=serialize_kwargs({'data':data}),
        serialization_format='cloudpickle',dependencies=['boto3==1.43.98'],system_dependencies=[],accelerate_downloads=False)
    accepted=base.call(endpoint,'run',{'input':payload,'policy':plan['provider_policy']})
    record.update(job_id=accepted['id'],submission_state='accepted');write(receipt,record)
    write(evidence/'submission-accepted.json',record)
    return {k:record[k]for k in ('endpoint','job_id','submitted_at_utc','absolute_expiry_utc')}


def status(evidence):
    receipt=evidence/'submission.json';record=read(receipt);path=evidence/'provider-status.json'
    if path.exists()and read(path).get('status')in('COMPLETED','FAILED','CANCELLED','TIMED_OUT'):
        raise PermissionError('Terminal provider status already retained; no further query')
    previous=record.get('last_inspection_at',record['submitted_at_epoch'])
    if time.time()-previous<900:raise PermissionError('First and later inspections at least15minutes apart')
    if record.get('inspection_attempts',0)>=8:raise PermissionError('Finite eight foreground inspection attempts exhausted')
    record.update(last_inspection_at=time.time(),inspection_attempts=record.get('inspection_attempts',0)+1);write(receipt,record)
    base=load(BASE_LAUNCH,'forward_original_status_client')
    result=base.call(record['endpoint'],'status/'+record['job_id']);write(path,result)
    return {k:v for k,v in result.items()if k!='output'}


def restore(evidence, work):
    record=read(evidence/'submission-accepted.json');attempt=evidence/'durable-restore-attempt.json'
    if attempt.exists():raise FileExistsError('Original restore attempt exists; inspect it without automatic retry')
    write(attempt,dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=ref(evidence/'submission-accepted.json')))
    base=load(BASE_LAUNCH,'forward_original_restore_client');client,bucket,_=base.s3()
    if bucket!=record['bucket']:raise ValueError('Original transport bucket changed')
    sidecar=load(SIDECAR,'forward_completion_validator');binding=record['completion_binding']
    response=client.get_object(Bucket=bucket,Key=binding['completion_key'])
    try:raw=response['Body'].read(record['plan']['completion_sidecar']['completion_bytes_max']+1)
    finally:response['Body'].close()
    observed=dict(endpoint=client.meta.endpoint_url,bucket=bucket,key=binding['completion_key'],HTTP_status=response['ResponseMetadata']['HTTPStatusCode'],
        bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),EOF_observed=response.get('ContentLength')==len(raw))
    completion=json.loads(raw);accepted=sidecar.accept_completion_metadata(completion,record,observed)
    (evidence/'completion-original.json').write_bytes(raw);write(evidence/'completion-acceptance.json',accepted)
    scratch=work/'receive-packs';scratch.mkdir(exist_ok=False)
    inventory={};packs=[]
    def unpack(item):
        target=scratch/(item['sha256']+'.tar');response=client.get_object(Bucket=bucket,Key=item['key'])
        try:
            with target.open('wb')as stream:shutil.copyfileobj(response['Body'],stream)
        finally:response['Body'].close()
        with target.open('rb')as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if target.stat().st_size!=item['bytes']or sha!=item['sha256']:raise ValueError('Full evidence pack SHA/bytes differ')
        expected={r['path']:r for r in item['files']}
        with tarfile.open(target)as tf:
            members=tf.getmembers()
            if {m.name for m in members}!=set(expected)or len(members)!=len(expected):raise ValueError('Exact pack member set differs')
            for m in members:
                dest=evidence/m.name
                if not m.isfile()or not dest.resolve().is_relative_to(evidence.resolve()):raise ValueError('Unsafe native member')
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():raise FileExistsError('No overwrite of an original native file: '+m.name)
                with tf.extractfile(m)as source,dest.open('xb')as output:shutil.copyfileobj(source,output)
                if ref(dest)['sha256']!=expected[m.name]['sha256']or dest.stat().st_size!=expected[m.name]['bytes']:raise ValueError('Extracted original SHA/bytes differ')
                inventory[m.name]=expected[m.name]
        packs.append(item)
    unpack(accepted['expected_final_pack'])
    worker=read(evidence/'evidence/worker-report.json')
    for item in worker['packs']:unpack(item)
    index=dict(schema='cove50.forward-stop.closed-native-index.v1',terminal=True,original_inventory_closed=True,
        terminal_semantics='Authenticated worker closure; provider lifecycle is separately observed, never inferred.',
        run_root=evidence.relative_to(ROOT).as_posix(),accepted=ref(evidence/'submission-accepted.json'),
        completion=ref(evidence/'completion-original.json'),worker_report=ref(evidence/'evidence/worker-report.json'),
        worker_status=worker['status'],totals=worker['totals'],files=[inventory[k]for k in sorted(inventory)],packs=packs)
    write(evidence/'closed-native-index.json',index);write(DATA/'execution/native-index-r2.json',index)
    write(evidence/'durable-restore-complete.json',dict(index=ref(evidence/'closed-native-index.json'),verified_originals=len(inventory),verified_pack_count=len(packs),cheap_transport_archives_removed=True))
    shutil.rmtree(scratch)
    return dict(index=ref(evidence/'closed-native-index.json'),worker_status=worker['status'],totals=worker['totals'],files=len(inventory))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('submit','status','restore'))
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--work',type=Path)
    parser.add_argument('--release',type=Path);args=parser.parse_args()
    if args.action=='submit':result=submit(args.evidence.resolve(),args.work.resolve(),args.release.resolve())
    elif args.action=='status':result=status(args.evidence.resolve())
    else:result=restore(args.evidence.resolve(),args.work.resolve())
    print(json.dumps(result))
