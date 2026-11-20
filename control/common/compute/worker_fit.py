def run_fit(data):
    """Bounded native PPO continuation with durable whole-episode stage packs.

    Transport credentials remain in this function's memory. This function does
    no provider polling, provisioning, or background scheduling.
    """
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import signal
    import subprocess
    import tarfile
    import tempfile
    import time
    import traceback
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    start = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix='r3-forward-fit-'))
    evidence = root/'evidence'; evidence.mkdir()
    recipe = dict(capacity_worlds=[256,512,1024], capacity_controls=48,
        eval_seeds=[136101,136201], train_seed=136100, eval_velocity_x=.30,
        target_rounds=400, minimum_rounds=200, chunk_rounds=100,
        hard_deadline_s=3420, trace_budget_bytes=8_000_000_000,
        source_checkpoint='artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt')
    recipe.update(data.get('recipe') or {})
    if recipe['hard_deadline_s'] > 3420 or recipe['target_rounds'] > 400 or recipe['minimum_rounds'] < 200:
        raise ValueError('Recipe exceeds the authorized bounded continuation')
    prefix = data['output_prefix'].rstrip('/')
    storage = data['storage']
    client = boto3.client('s3', endpoint_url=storage['endpoint'], region_name=storage['region'],
        config=Config(signature_version='s3v4',s3={'addressing_style':'path'},
            connect_timeout=10,read_timeout=90,retries={'total_max_attempts':1},
            request_checksum_calculation='when_required',response_checksum_validation='when_required'),
        **data['storage_auth'])
    report = dict(schema='cove50.native-forward-fit.v1',status='starting',recipe=recipe,
        bundle_sha256=data['bundle_sha256'],packs=[],stages=[],
        progress_key=prefix+'/progress.json',
        accounting='Every native transition/episode retained. Capacity/replays have zero optimizer steps. Training rounds, optimizer steps, simulator transitions and checkpoints remain separate.')

    def digest(path):
        with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

    def progress():
        report['wall_s'] = time.monotonic()-start
        content = (json.dumps(report,indent=2)+'\n').encode()
        (evidence/'progress.json').write_bytes(content)
        client.put_object(Bucket=storage['bucket'],Key=report['progress_key'],Body=content,ContentType='application/json')

    def publish(name):
        """Upload immutable unique stage pack and hash-verify its bucket copy."""
        archive = root/(name+'.tar.gz')
        with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
            for path in (evidence/name,evidence/(name+'.log')):
                if path.exists(): tf.add(path,arcname='evidence/'+path.name)
        sha = digest(archive); key = prefix+'/packs/'+name+'.tar.gz'
        client.upload_file(str(archive),storage['bucket'],key,
            Config=TransferConfig(max_concurrency=1,use_threads=False))
        response = client.get_object(Bucket=storage['bucket'],Key=key)
        try: actual = hashlib.file_digest(response['Body'],'sha256').hexdigest()
        finally: response['Body'].close()
        if actual != sha: raise RuntimeError('Bucket stage-pack bytes differ: '+name)
        row = dict(key=key,sha256=sha,bytes=archive.stat().st_size,stage=name)
        report['packs'].append(row)
        progress()
        archive.unlink()  # cheap duplicate; original native files stay until worker exit
        return row

    def stage(name,command,timeout,tolerate=False):
        report['active_stage'] = name; progress()
        remaining = min(timeout,recipe['hard_deadline_s']-(time.monotonic()-start)-90)
        if remaining <= 0: raise RuntimeError('Local worker deadline reached before '+name)
        began = time.monotonic(); timed_out = False
        with (evidence/(name+'.log')).open('wb') as log:
            proc = subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True; os.killpg(proc.pid,signal.SIGINT)
                try: code = proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); code = proc.wait()
        row = dict(name=name,exit_code=code,wall_s=time.monotonic()-began,operator_truncated=timed_out)
        receipt = evidence/name/'report.json'
        if receipt.exists(): row['native'] = json.loads(receipt.read_text())
        report['stages'].append(row)
        publish(name)
        if (code or timed_out) and not tolerate: raise RuntimeError('Native stage incomplete: '+name)
        return row

    try:
        progress()
        response = client.get_object(Bucket=storage['bucket'],Key=data['input_key'])
        archive = root/'inputs.tar.gz'
        with archive.open('wb') as f: shutil.copyfileobj(response['Body'],f)
        response['Body'].close()
        if archive.stat().st_size > 200_000_000 or digest(archive) != data['bundle_sha256']:
            raise ValueError('Input bundle identity/size changed')
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not (member.isfile() or member.isdir()) or not (root/member.name).resolve().is_relative_to(root):
                    raise ValueError('Unsafe input archive member')
            tf.extractall(root,filter='data')
        frozen_recipe = json.loads((root/'control/common/compute/recipe.json').read_text())
        if frozen_recipe != recipe: raise ValueError('Dispatch recipe differs from bundled frozen recipe')
        stage('hardware',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],20)
        stage('bootstrap',['bash',data['bootstrap']],700)
        shutil.copy2(root/'worker-attestation.json',evidence/'worker-attestation.json')
        shutil.copy2(root/'worker-pip-freeze.txt',evidence/'worker-pip-freeze.txt')
        python = str(root/'.venv-worker/bin/python')
        task = 'control/skills/walker/task/run.py'
        common = [python,'-u',task,'--device','cuda:0']
        source = ['--source-checkpoint',recipe['source_checkpoint']]
        stage('gpu-smoke',common+['--mode','smoke','--worlds','1','--steps','2',
            '--seed',str(recipe['train_seed']),'--out',str(evidence/'gpu-smoke')]+source,180)
        screens = []
        for worlds in recipe['capacity_worlds']:
            name = 'capacity-'+str(worlds)
            row = stage(name,common+['--mode','capacity','--worlds',str(worlds),
                '--steps',str(recipe['capacity_controls']),'--seed',str(recipe['train_seed']),
                '--out',str(evidence/name)]+source,180,tolerate=True)
            native = row.get('native',{})
            if row['exit_code'] or row['operator_truncated']:
                error = native.get('error','')+'\n'+(evidence/(name+'.log')).read_text(errors='replace')[-12000:]
                resource = row['operator_truncated'] or any(marker in error.lower() for marker in (
                    'out of memory','cuda_error_out_of_memory','contact buffer capacity',
                    'nconmax overflow','njmax overflow'))
                row['failure_classification'] = 'resource_or_capacity_wall_budget' if resource else 'programming_or_unknown_native_error'
                progress()
                if not resource: raise RuntimeError('Capacity stage programming/device error; no repeated larger constructors: '+name)
                report['capacity_increasing_worlds_stopped'] = dict(stage=name,reason=row['failure_classification'])
                break
            if row['exit_code']==0 and native.get('measured_collection_s',0)>0:
                screens.append(dict(worlds=worlds,
                    estimated_round_s=native['measured_collection_s']*24/native['measured_controls_per_world']+.5,
                    bytes_per_transition=native['retained_episode_bytes']/max(1,native['simulator_transitions'])))
        baseline_times = []
        for seed in recipe['eval_seeds']:
            name = 'before-'+str(seed)
            row = stage(name,common+['--mode','replay','--worlds','1','--seed',str(seed),
                '--velocity-x',str(recipe['eval_velocity_x']),'--out',str(evidence/name)]+source,240)
            baseline_times.append(row['wall_s'])
        # Reserve paired complete replays, constructors and durable uploads;
        # measured selection changes compute allocation, never task/rewards.
        reserve = sum(baseline_times)*1.25+150
        available = recipe['hard_deadline_s']-(time.monotonic()-start)-reserve
        allocations = []
        for screen in screens:
            for rounds in range(recipe['minimum_rounds'],recipe['target_rounds']+1,100):
                seconds = rounds*screen['estimated_round_s']*1.2+(rounds//100)*25
                size = rounds*24*screen['worlds']*screen['bytes_per_transition']*1.25
                if seconds <= available and size <= recipe['trace_budget_bytes']:
                    allocations.append(dict(**screen,rounds=rounds,estimated_fit_s=seconds,
                        estimated_trace_bytes=size,planned_transitions=rounds*24*screen['worlds']))
        report['capacity_evidence'] = screens
        report['allocation_candidates'] = allocations
        if not allocations: raise RuntimeError('Measured capacity/time/storage cannot fit200 PPO rounds within this job; retain capacity and paired baseline evidence for revised compute allocation')
        chosen = max(allocations,key=lambda x:(x['planned_transitions'],x['rounds']))
        report['allocation'] = chosen; progress()
        previous = None; completed = 0; last_chunk_wall = None
        for end_round in range(100,chosen['rounds']+1,100):
            remaining = recipe['hard_deadline_s']-(time.monotonic()-start)
            estimate = last_chunk_wall*1.2 if last_chunk_wall else chosen['estimated_round_s']*100*1.2+25
            if remaining < estimate+reserve:
                report['budget_stop'] = dict(completed_rounds=completed,remaining_s=remaining,
                    next_chunk_estimated_s=estimate,final_replay_and_upload_reserve_s=reserve)
                break
            name = 'fit-through-'+str(end_round)
            seed_input = ['--checkpoint',str(previous)] if previous else source
            chunk_start = time.monotonic()
            row = stage(name,common+['--mode','train','--worlds',str(chosen['worlds']),
                '--updates','100','--seed',str(recipe['train_seed']+end_round),
                '--out',str(evidence/name)]+seed_input,
                max(1,recipe['hard_deadline_s']-(time.monotonic()-start)-reserve),tolerate=True)
            last_chunk_wall = time.monotonic()-chunk_start
            candidate = evidence/name/'candidate.pt'
            if not candidate.exists(): candidate = evidence/name/'partial-candidate.pt'
            if not candidate.exists():
                report['training_stop'] = 'No new recoverable checkpoint in '+name
                break
            previous = root/'active-continuation.pt'
            shutil.copy2(candidate,previous)
            completed += row.get('native',{}).get('ppo_rounds',0)
            report['completed_ppo_rounds'] = completed
            report['evaluated_checkpoint_source'] = str(candidate.relative_to(root))
            progress()
            if row['exit_code'] or row['operator_truncated']: break
        if previous is None: raise RuntimeError('No recoverable updated policy/value checkpoint; retain setup evidence')
        final_rows = []
        for seed in recipe['eval_seeds']:
            name = 'after-'+str(seed)
            final_rows.append(stage(name,common+['--mode','replay','--worlds','1','--seed',str(seed),
                '--velocity-x',str(recipe['eval_velocity_x']),'--checkpoint',str(previous),
                '--out',str(evidence/name)],240,tolerate=True))
        report['paired_final_replay_status'] = [dict(seed=seed,status=row.get('native',{}).get('status','unavailable'))
            for seed,row in zip(recipe['eval_seeds'],final_rows)]
        report['status'] = ('completed' if completed == chosen['rounds'] else 'budget_bounded_training_prefix') if all(row['exit_code']==0 for row in final_rows) else 'incomplete_final_replay_evidence'
    except BaseException as exc:
        detail = traceback.format_exc()[-8000:]
        for secret in data['storage_auth'].values(): detail = detail.replace(secret,'[REDACTED]')
        report.update(status='incomplete',error_type=type(exc).__name__,traceback=detail)
    finally:
        report.pop('active_stage',None)
        report['wall_s'] = time.monotonic()-start
        report['totals'] = {}; report['known_lower_bounds'] = {}; report['unknown_counter_stages'] = {}
        for key in ('simulator_transitions','native_physics_steps','optimizer_steps','ppo_rounds',
                    'complete_or_terminated_episodes','censored_episodes'):
            known = 0; unknown = []
            for row in report['stages']:
                native = row.get('native',{})
                if key in native: known += native[key]
                elif row['name'] in ('hardware','bootstrap'): continue  # static construction, zero simulation/learning
                elif key in ('optimizer_steps','ppo_rounds') and not row['name'].startswith('fit-through-'): continue
                else: unknown.append(row['name'])
            report['known_lower_bounds'][key] = known
            report['unknown_counter_stages'][key] = unknown
            report['totals'][key] = None if unknown else known
        # The final compact pack is an index and environment attestation;
        # all native stage trees live in their immutable separately hashed packs.
        (evidence/'worker-report.json').write_text(json.dumps(report,indent=2)+'\n')
        progress()
        final = root/'evidence.tar.gz'
        with tarfile.open(final,'w:gz') as tf:
            for name in ('worker-report.json','progress.json','worker-attestation.json','worker-pip-freeze.txt'):
                path=evidence/name
                if path.exists(): tf.add(path,arcname='evidence/'+name)
        client.upload_file(str(final),storage['bucket'],data['output_key'],
            Config=TransferConfig(max_concurrency=1,use_threads=False))
    return dict(status=report['status'],wall_s=report['wall_s'],progress_key=report['progress_key'],
        output_sha256=digest(final),output_bytes=final.stat().st_size,packs=report['packs'],totals=report['totals'],
        known_lower_bounds=report['known_lower_bounds'],unknown_counter_stages=report['unknown_counter_stages'])
