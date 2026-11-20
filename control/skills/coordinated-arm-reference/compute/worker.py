def run_coordinated_arm_reference(data):
    """One finite intact whole22 feasibility screen; zero updates and no provider polling."""
    import datetime, hashlib, importlib.util, json, os, shutil, signal, subprocess, sys, tarfile, tempfile, time, traceback
    from pathlib import Path
    def digest(path):
        with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
    def module(path, name):
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
    def helpers(plan, sources):
        result={}
        for label, expected in plan['helpers'].items():
            raw=sources[label].encode()
            if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise ValueError('Reviewed helper differs: '+label)
            scope={};exec(compile(raw,expected['path'],'exec'),scope);result[label]=scope
        return result
    if data.get('source_only_root'):
        root=Path(data['source_only_root']).resolve()
        common=module(root/'control/skills/coordinated-arm-reference/compute/common.py','forward_capsule_common')
        plan=common.verify();loaded=helpers(plan,{k:(root/v['path']).read_text()for k,v in plan['helpers'].items()})
        checkpoint=root/common.read(common.DATA/'config.json')['checkpoint']['path']
        commands=[common.command(c,checkpoint,root/'evidence',root/'work')for c in plan['cases']]
        # Import the actual extracted thin entry with only its extracted directory
        # available for its sibling reference; no task/science imports in this mode.
        sys.path.insert(0,str(root/'control/skills/coordinated-arm-reference'))
        entry=module(root/'control/skills/coordinated-arm-reference/run.py','forward_capsule_native_entry')
        return dict(schema='cove50.coordinated-arm-reference-extracted-startup.v1',root=str(root),common_source=str(Path(common.__file__).resolve()),
            entry_source=str(Path(entry.__file__).resolve()),reference_source=str(Path(sys.modules['reference'].__file__).resolve()),
            helpers={k:list(v.keys())[-3:]for k,v in loaded.items()},commands=commands,stages=len(commands),
            imports_containing_science=[n for n in ('torch','numpy','mujoco','mjlab','warp')if n in sys.modules],
            NN_queries=0,checkpoint_loads=0,models=0,physics=0,optimizer_calls=0,provider_calls=0)
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    from botocore.exceptions import ClientError
    began=time.monotonic();plan=data['plan'];storage=data['storage'];prefix=data['output_prefix']
    loaded=helpers(plan,data['helper_sources']);transport=loaded['transport'];sidecar=loaded['sidecar']
    sidecar['validate_completion_binding'](data,plan['completion_sidecar'])
    client=boto3.client('s3',endpoint_url=storage['endpoint'],region_name=storage['region'],config=Config(
        signature_version='s3v4',s3={'addressing_style':'path'},connect_timeout=10,read_timeout=90,
        retries={'total_max_attempts':1},request_checksum_calculation='when_required',response_checksum_validation='when_required'),**data['storage_auth'])
    progress_key=prefix+'/progress.json'
    # One conditional create suppresses duplicate sequential/concurrent execution
    # before downloading or constructing native objects; no retry on any error.
    try:
        client.put_object(Bucket=storage['bucket'],Key=progress_key,Body=b'{"state":"claimed"}\n',ContentType='application/json',IfNoneMatch='*')
    except ClientError as exc:
        if exc.response.get('ResponseMetadata',{}).get('HTTPStatusCode')==412:
            return dict(status='duplicate_execution_suppressed',native_stage_attempts=0,NN_queries=0,physics=0)
        raise
    root=Path(tempfile.mkdtemp(prefix='cove50-coordinated-arm-reference-')).resolve();evidence=root/'evidence';evidence.mkdir();work=root/'work';work.mkdir()
    report=dict(schema='cove50.coordinated-arm-reference-worker.v1',status='starting',plan=plan,bundle_sha256=data['bundle_sha256'],
        completion_binding=data['completion_binding'],stages=[],packs=[],case_disposition={c['name']:'not_started'for c in plan['cases']})
    budget=plan['hard_deadline_s'];children=plan['children'];closed=set();common=None
    def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def safe_error(exc):
        detail=str(exc)
        for secret in data['storage_auth'].values():
            if secret:detail=detail.replace(secret,'[REDACTED]')
        return dict(error_type=type(exc).__name__,error=detail)
    def progress():
        report['wall_s']=time.monotonic()-began;write(evidence/'progress.json',report)
        client.put_object(Bucket=storage['bucket'],Key=progress_key,Body=(evidence/'progress.json').read_bytes(),ContentType='application/json')
    def publish(name, paths):
        archive=root/(name+'.tar');item=transport['pack'](root,paths,archive);item.update(key=prefix+'/packs/'+name+'.tar',stage=name)
        client.upload_file(str(archive),storage['bucket'],item['key'],Config=TransferConfig(max_concurrency=1,use_threads=False,preferred_transfer_client='classic'))
        response=client.get_object(Bucket=storage['bucket'],Key=item['key'])
        try:stored=hashlib.file_digest(response['Body'],'sha256').hexdigest()
        finally:response['Body'].close()
        if stored!=item['sha256']:raise ValueError('Full stored stage pack SHA differs')
        report['packs'].append(item);progress();archive.unlink()
    def stage(name, argv):
        child=next(c for c in children if c['name']==name)
        required=plan['transport_reserve_s']+sum(c['process_s']+c['join_s']+c['pack_s']for c in children if c['name']not in closed)
        available=budget-(time.monotonic()-began)
        if available<required:raise TimeoutError('Protected remaining child/transport budget unavailable before '+name)
        row=dict(name=name,command=argv,status='starting',remaining_s=available,required_s=required);report['stages'].append(row);progress()
        process=None;timed_out=False;started=time.monotonic()
        try:
            with (evidence/(name+'.log')).open('wb')as log:
                process=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={k:v for k,v in dict(os.environ,WANDB_MODE='offline',WANDB_SILENT='true',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',LAB_EVIDENCE_DIR=str(evidence),LAB_WORK_DIR=str(work)).items() if k not in ('PYTHONPATH','PYTHONHOME')})
                row['pid']=process.pid;row['status']='running'
                try:progress()
                except Exception as exc:row['post_spawn_progress_error']=safe_error(exc)
                try:code=process.wait(timeout=child['process_s'])
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(process.pid,signal.SIGINT)
                    try:code=process.wait(timeout=child['join_s'])
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
            row.update(status='closed',exit_code=code,operator_budget_censored=timed_out)
        finally:
            # Any interruption/setup/metadata exception still joins a spawned child.
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try:process.wait(timeout=child['join_s'])
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            closed.add(name);row['wall_s']=time.monotonic()-started
            original=evidence/name/'report.json'
            if original.exists():
                try:row['native']=json.loads(original.read_text())
                except (ValueError,OSError)as exc:row['report_read_error']=safe_error(exc)
            publish(name,[evidence/name,evidence/(name+'.log')])
        if code or timed_out:raise RuntimeError('Original child incomplete; no retry: '+name)
        return row
    try:
        expiry=datetime.datetime.fromisoformat(data['absolute_expiry_utc'])
        if expiry.tzinfo is None:raise ValueError('Timezone-aware provider expiry required')
        budget=min(budget,expiry.timestamp()-time.time()-plan['absolute_ttl_safety_s'])
        if budget<plan['hard_deadline_s']:raise TimeoutError('Full reviewed entry budget unavailable; no shortened native work')
        response=client.get_object(Bucket=storage['bucket'],Key=data['input_key']);archive=root/'input.tar.gz'
        try:
            with archive.open('wb')as stream:shutil.copyfileobj(response['Body'],stream)
        finally:response['Body'].close()
        if digest(archive)!=data['bundle_sha256'] or archive.stat().st_size!=data['completion_binding']['input_archive']['bytes']:
            raise ValueError('Exact input archive identity differs')
        with tarfile.open(archive)as tf:
            for m in tf:
                if not m.isfile()or not(root/m.name).resolve().is_relative_to(root):raise ValueError('Unsafe capsule member')
            tf.extractall(root,filter='data')
        common=module(root/'control/skills/coordinated-arm-reference/compute/common.py','forward_native_common')
        if common.verify()!=plan:raise ValueError('Dispatch/packed plan differs')
        common.activation(plan,data['activation']);write(evidence/'activation.json',data['activation'])
        stage('hardware',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])
        stage('bootstrap',['bash',str(common.HERE/'bootstrap.sh')])
        for n in ('worker-attestation.json','worker-pip-freeze.txt'):shutil.copy2(root/n,evidence/n)
        checkpoint=root/common.read(common.DATA/'config.json')['checkpoint']['path']
        for case in plan['cases']:
            report['case_disposition'][case['name']]='attempted'
            stage(case['name'],common.command(case,checkpoint,evidence,work))
            common.completed(case,evidence/case['name'])
            report['case_disposition'][case['name']]='completed'
        report['status']='completed'
    except BaseException as exc:
        report.update(status='incomplete',**safe_error(exc))
        detail=traceback.format_exc()[-12000:]
        for secret in data['storage_auth'].values():
            if secret:detail=detail.replace(secret,'[REDACTED]')
        report['traceback']=detail
    finally:
        native_rows=[r for r in report['stages']if r['name']not in ('hardware','bootstrap')]
        report['actual_stage_attempts']=len(native_rows);report['totals']={};report['known_lower_bounds']={};report['unknown_counter_stages']={}
        for key in ('simulator_controls','physics_substeps','optimizer_updates','new_weight_sets'):
            values=[r.get('native',{}).get('counts',{}).get(key)for r in native_rows]
            unknown=[r['name']for r,v in zip(native_rows,values)if not isinstance(v,int)or isinstance(v,bool)]
            known=sum(v for v in values if isinstance(v,int)and not isinstance(v,bool))
            report['totals'][key]=None if unknown else known;report['known_lower_bounds'][key]=known;report['unknown_counter_stages'][key]=unknown
        report['wall_s']=time.monotonic()-began
        report['native_world_slot_instances']=sum(r.get('native',{}).get('actual_world_slots',0)for r in native_rows)if all('actual_world_slots'in r.get('native',{})for r in native_rows)else None
        report['distinct_fixed_policies']=1 if any(r.get('native',{}).get('distinct_fixed_policies')==1 for r in native_rows)else 0 if not native_rows else None
        report['start_families']=sum(r.get('native',{}).get('independent_start_families',0)for r in native_rows)if all('independent_start_families'in r.get('native',{})for r in native_rows)else None
        report['NN_queries']=sum(r.get('native',{}).get('donor_actor_queries',0)for r in native_rows)if all('donor_actor_queries'in r.get('native',{})for r in native_rows)else None
        for name in ('worker-attestation.json','worker-pip-freeze.txt'):
            if (root/name).exists()and not(evidence/name).exists():shutil.copy2(root/name,evidence/name)
        covered={r['path']for pack in report['packs']for r in pack['files']}
        rescue=[p for p in evidence.rglob('*')if p.is_file()and p.relative_to(root).as_posix()not in covered]
        report['final_rescued_original_paths']=[p.relative_to(root).as_posix()for p in rescue]
        write(evidence/'worker-report.json',report)
        try:progress()
        except BaseException as exc:report['final_progress_error']=safe_error(exc);write(evidence/'worker-report.json',report)
        paths=list(dict.fromkeys(rescue+[evidence/'worker-report.json',evidence/'progress.json']))
        paths=[p for p in paths if p.exists()]
        final=root/'evidence.tar';item=transport['pack'](root,paths,final)
        client.upload_file(str(final),storage['bucket'],data['output_key'],Config=TransferConfig(max_concurrency=1,use_threads=False,preferred_transfer_client='classic'))
        completion=sidecar['verify_final_and_publish_completion'](client,data,report,item,evidence,plan['completion_sidecar'],lambda:budget-(time.monotonic()-began))
    return dict(status=report['status'],totals=report['totals'],packs=report['packs'],completion=completion,
        output_sha256=item['sha256'],output_bytes=item['bytes'],output_files=item['files'])


if __name__ == '__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('--source-only-root',required=True)
    args=parser.parse_args();print(json.dumps(run_coordinated_arm_reference(dict(source_only_root=args.source_only_root)),indent=2))
