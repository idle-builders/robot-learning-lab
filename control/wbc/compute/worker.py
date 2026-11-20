def run_wbc_pilot(data):
    """One bounded WBC teacher pilot job; stage packs pushed as they close, the job never retried.

    Only this function's source is dispatched. It claims the job once with a
    conditional progress object, verifies and extracts the frozen capsule,
    runs hardware/bootstrap/benchmark/train/evaluate as child processes under
    one hard deadline, uploads each stage's evidence pack with a full SHA-256
    readback, and finally writes completion.json. It never polls the provider.
    Running stages push a heartbeat (log tail, host/GPU state, newest training
    checkpoint) and are stopped when their log stops growing for stall_s.
    """
    import datetime, hashlib, json, os, shutil, signal, subprocess, tarfile, tempfile, time, traceback
    from pathlib import Path
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    from botocore.exceptions import ClientError

    began = time.monotonic()
    storage, plan, prefix = data['storage'], data['plan'], data['output_prefix']
    connect = lambda attempts: boto3.client('s3', endpoint_url=storage['endpoint'], region_name=storage['region'], config=Config(
        signature_version='s3v4', s3={'addressing_style': 'path'}, connect_timeout=10, read_timeout=90,
        retries={'total_max_attempts': attempts, 'mode': 'standard'}, request_checksum_calculation='when_required',
        response_checksum_validation='when_required'), **data['storage_auth'])
    client = connect(1)
    bucket = storage['bucket']
    try:
        client.put_object(Bucket=bucket, Key=prefix + '/progress.json', Body=b'{"state":"claimed"}\n', IfNoneMatch='*')
    except ClientError as exc:
        if exc.response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 412:
            return dict(status='duplicate_execution_suppressed')
        raise
    # After the conditional claim every write is a plain put of a hash-checked
    # object, so storage timeouts are retried (one cost teacher-widths-r3).
    client = connect(5)

    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def write(path, value):
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    def redact(text):
        for secret in data['storage_auth'].values():
            if secret:
                text = text.replace(secret, '[REDACTED]')
        return text

    root = Path(tempfile.mkdtemp(prefix='cove50-wbc-')).resolve()
    evidence = root / 'evidence'
    (evidence / 'bootstrap').mkdir(parents=True)
    expiry = datetime.datetime.fromisoformat(data['absolute_expiry_utc']).timestamp()
    budget = min(plan['hard_deadline_s'], expiry - time.time() - plan['ttl_safety_s'])
    report = dict(schema='cove50.wbc.worker.v1', status='starting', plan=plan, bundle_sha256=data['bundle_sha256'],
        stages=[], packs=[], budget_s=budget)

    def remaining():
        return budget - (time.monotonic() - began)

    def progress():
        report['wall_s'] = time.monotonic() - began
        write(evidence / 'progress.json', report)
        client.put_object(Bucket=bucket, Key=prefix + '/progress.json', Body=(evidence / 'progress.json').read_bytes())

    def publish(name, paths):
        archive, rows = root / (name + '.tar'), []
        with tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as tf:
            for path in paths:
                for entry in sorted(path.rglob('*')) if path.is_dir() else [path]:
                    if entry.is_file() and not entry.is_symlink():
                        member = str(entry.relative_to(root))
                        info = tf.gettarinfo(str(entry), member)
                        info.mode, info.uid, info.gid, info.uname, info.gname = 0o644, 0, 0, '', ''
                        with entry.open('rb') as stream:
                            tf.addfile(info, stream)
                        rows.append(dict(path=member, bytes=entry.stat().st_size, sha256=digest(entry)))
        item = dict(stage=name, key=prefix + '/packs/' + name + '.tar', sha256=digest(archive),
            bytes=archive.stat().st_size, files=rows)
        client.upload_file(str(archive), bucket, item['key'], Config=TransferConfig(max_concurrency=1, use_threads=False))
        response = client.get_object(Bucket=bucket, Key=item['key'])
        try:
            stored = hashlib.file_digest(response['Body'], 'sha256').hexdigest()
        finally:
            response['Body'].close()
        if stored != item['sha256']:
            raise ValueError('Stored pack differs: ' + name)
        archive.unlink()
        report['packs'].append(item)
        progress()

    def stop(process):
        """SIGINT, then SIGKILL; a child stuck in the GPU driver may never exit, so every wait is bounded."""
        for sig in (signal.SIGINT, signal.SIGKILL):
            os.killpg(process.pid, sig)
            try:
                return process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        return None

    def host(row):
        row['disk_free_gb'] = shutil.disk_usage(root).free / 2**30
        try:
            row['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip()
        except subprocess.TimeoutExpired:
            row['gpu'] = 'nvidia-smi timed out'
        meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        row['ram_available_gb'] = int(meminfo['MemAvailable'].split()[0]) / 2**20

    def heartbeat(row, log_path, partial):
        """Mid-stage evidence: log tail, newest training checkpoint, host/GPU state."""
        with log_path.open('rb') as stream:
            stream.seek(max(0, log_path.stat().st_size - 3000))
            row['log_tail'] = redact(stream.read().decode(errors='replace'))
        try:
            host(row)
        except Exception as exc:
            row['host_error'] = f'{type(exc).__name__}: {exc}'
        if partial and partial.exists() and partial.stat().st_mtime != row.get('partial', {}).get('mtime'):
            snapshot = root / 'partial-snapshot.pt'  # a hard link keeps this inode while the trainer replaces latest.pt
            snapshot.unlink(missing_ok=True)
            os.link(partial, snapshot)
            item = dict(key=f'{prefix}/partial/{row["name"]}.pt', sha256=digest(snapshot), bytes=snapshot.stat().st_size,
                mtime=partial.stat().st_mtime, elapsed_s=row['elapsed_s'])
            client.upload_file(str(snapshot), bucket, item['key'], Config=TransferConfig(max_concurrency=1, use_threads=False))
            snapshot.unlink()
            row['partial'] = item
        progress()

    def stage(name, argv, timeout_s, partial=None):
        if remaining() < timeout_s + plan['reserve_s']:
            raise TimeoutError(f'Not enough budget left for {name}')
        row = dict(name=name, command=argv, timeout_s=timeout_s, status='running')
        report['stages'].append(row)
        progress()
        started, timed_out, stalled, code = time.monotonic(), False, False, None
        log_path = evidence / (name + '.log')
        with log_path.open('wb') as log:
            process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MUJOCO_GL='disable'))
            size, grew = 0, started
            while True:
                try:
                    code = process.wait(timeout=max(1, min(plan['heartbeat_s'], started + timeout_s - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if log_path.stat().st_size != size:
                    size, grew = log_path.stat().st_size, now
                timed_out, stalled = now - started >= timeout_s, now - grew >= plan['stall_s']
                if timed_out or stalled:
                    code = stop(process)
                    row['unkillable'] = code is None
                    break
                row['elapsed_s'] = now - started
                try:
                    heartbeat(row, log_path, partial and root / partial)
                except Exception as exc:
                    row['heartbeat_error'] = redact(f'{type(exc).__name__}: {exc}')
        row.update(status='closed', exit_code=code, timed_out=timed_out, stalled=stalled, wall_s=time.monotonic() - started)
        native = evidence / name / 'report.json'
        if native.exists():
            row['native'] = json.loads(native.read_text()).get('native')
        publish(name, [p for p in (evidence / name, evidence / (name + '.log')) if p.exists()])
        if code or code is None or timed_out or stalled:
            raise RuntimeError(f'{name} incomplete (exit {code}, timed out {timed_out}, stalled {stalled}); no retry')

    try:
        archive = root / 'input.tar.gz'
        client.download_file(bucket, data['input_key'], str(archive))
        if digest(archive) != data['bundle_sha256'] or archive.stat().st_size != data['bundle_bytes']:
            raise ValueError('Input capsule identity differs')
        with tarfile.open(archive) as tf:
            tf.extractall(root, filter='data')
        archive.unlink()
        manifest = json.loads((root / 'capsule-manifest.json').read_text())
        for row in manifest['files']:
            if digest(root / row['path']) != row['sha256']:
                raise ValueError('Capsule member differs: ' + row['path'])
        py, st = str(root / '.venv-worker/bin/python'), plan['stage_script']
        stage('hardware', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 60)
        stage('bootstrap', ['bash', plan['bootstrap']], plan['timeouts_s']['bootstrap'])
        arms, evaluate_s = plan['arms'], plan['timeouts_s']['evaluate']
        common = (['--motion', plan['motion'], '--seed', str(plan['seed'])] + (['--sonic', plan['sonic']] if plan.get('sonic') else [])
            + (['--recipe', plan['recipe']] if plan.get('recipe') else [])
            + (['--reward-widths', plan['reward_widths']] if plan.get('reward_widths') else [])
            + (['--peak-torque'] if plan.get('peak_torque') else [])
            + (['--actuator', plan['actuator']] if plan.get('actuator') else [])
            + (['--dynamics', plan['dynamics']] if plan.get('dynamics') else [])
            + (['--stop-weight', str(plan['stop_weight'])] if plan.get('stop_weight') else [])
            + (['--joint-weight', str(plan['joint_weight'])] if plan.get('joint_weight') else [])
            + (['--knee-weight', str(plan['knee_weight'])] if plan.get('knee_weight') else [])
            + (['--reward-set', plan['reward_set']] if plan.get('reward_set') else [])
            + (['--command-delay', str(plan['command_delay'])] if plan.get('command_delay') is not None else [])
            + (['--feedback-delay', str(plan['feedback_delay'])] if plan.get('feedback_delay') is not None else [])
            + (['--feet-scale', *map(str, plan['feet_scale'])] if plan.get('feet_scale') else [])
            + (['--preview-steps', *map(str, plan['preview_steps'])] if plan.get('preview_steps') else [])
            + (['--arm-scale', *map(str, plan['arm_scale'] if isinstance(plan['arm_scale'], list) else [plan['arm_scale']])] if plan.get('arm_scale') else [])
            + (['--reference-lead', str(plan['reference_lead'])] if plan.get('reference_lead') else [])
            + (['--fist', *map(str, plan['fist'])] if plan.get('fist') else [])
            + (['--residual-action'] if plan.get('residual_action') else [])
            + (['--anchor-tolerance', str(plan['anchor_tolerance'])] if plan.get('anchor_tolerance') else [])
            + (['--error-sampling', str(plan['error_sampling'])] if plan.get('error_sampling') else [])
            + (['--physics-dt', str(plan['physics_dt'])] if plan.get('physics_dt') else [])
            + (['--terrain', plan['terrain']] if plan.get('terrain') else []))
        # Evaluation may use its own tape (eval_motion: held-out motions the training tape never holds) and, after
        # training on the obstacle world, also runs on the plane; eval_terrain adds a run on the obstacle world.
        eval_motion = plan.get('eval_motion', plan['motion'])
        swap = lambda args: [eval_motion if x == plan['motion'] else x for x in args]
        flat = common[:-2] if plan.get('terrain') else common
        evaluation = ['--eval-worlds', str(plan['eval_worlds']), '--push-step', str(plan['push_step'])]
        evaluation += ['--conditions', *plan['eval_conditions']] if plan.get('eval_conditions') else []
        terrain_evaluation = list(evaluation)
        if plan.get('eval_segments'):  # multi-motion tape: evaluate these segments from their first frames
            modes = ['--modes', *plan.get('eval_modes', ['teacher', 'reference'])]
            listing = ['--segments', str(Path(eval_motion).parent / 'motions.json')]
            evaluation += [*listing, '--segment-names', *plan['eval_segments'], *modes]
            terrain_evaluation += [*listing, '--segment-names', *plan.get('eval_terrain_segments', plan['eval_segments']), *modes]
        if not plan.get('evaluate_only'):
            stage('benchmark', [py, st, 'benchmark', *common, '--variant', arms[0], '--out', 'evidence/benchmark',
                '--worlds', *map(str, plan['benchmark_worlds']), '--steps', str(plan['benchmark_steps'])], plan['timeouts_s']['benchmark'])
        start = ['--checkpoint', plan['init_from']] if plan.get('init_from') else []
        if plan.get('zero_shot') or plan.get('evaluate_only'):  # the starting weights before any training
            name = 'evaluate-start' if start else 'evaluate-zero-shot'
            stage(name, [py, st, 'evaluate', *swap(common), *(terrain_evaluation if plan.get('terrain') else evaluation),
                '--variant', arms[0], '--out', f'evidence/{name}', *start], evaluate_s)
            if plan.get('eval_flat_references') and plan.get('terrain'):  # the same evaluation on the flat tape (A/B)
                stage(f'{name}-flat', [py, st, 'evaluate', *swap(common), *terrain_evaluation, '--flat-references',
                    '--variant', arms[0], '--out', f'evidence/{name}-flat', *start], evaluate_s)
        if plan.get('evaluate_only'):  # scoring a retained teacher on new motions: no training stage
            arms = []
        iterations, report['train_budget_s'] = None, {}
        for i, arm in enumerate(arms):
            # Later arms get the first arm's PPO iteration count within an equal time share.
            left = len(arms) - i
            evaluations = 0 if plan.get('train_only') else 2 if plan.get('eval_terrain') else 1
            train_s = (remaining() - left * (evaluations * evaluate_s + plan['train_join_s']) - 2 * plan['reserve_s']) / left
            train_s = min(train_s, plan.get('train_s', train_s))  # a fixed training wall for matched comparisons
            if train_s < plan['minimum_train_s']:
                raise TimeoutError(f'Only {train_s:.0f} s left for training {arm}')
            report['train_budget_s'][arm] = train_s
            out = f'evidence/train-{arm}'
            stage(f'train-{arm}', [py, st, 'train', *common, '--variant', arm, '--out', out, '--benchmark', 'evidence/benchmark',
                '--seconds', str(int(train_s))] + (['--iterations', str(iterations or plan.get('iterations'))] if iterations or plan.get('iterations') else [])
                + (['--minibatch-samples', str(plan['minibatch_samples'])] if plan.get('minibatch_samples') else [])
                + (['--init-checkpoint', plan['init_from']] if plan.get('init_from') else [])
                + (['--reset-actor-output'] if plan.get('reset_actor_output') and plan.get('init_from') else [])
                + (['--init-widths'] if plan.get('init_widths') and plan.get('init_from') else [])
                + (['--plateau-window', str(plan['plateau']['window']), '--plateau-gain', str(plan['plateau']['gain'])] if plan.get('plateau') else []),
                int(train_s) + plan['train_join_s'], partial=f'{out}/latest.pt')
            iterations = iterations or json.loads((root / out / 'report.json').read_text())['native']['ppo_rounds']
            if plan.get('train_only'):  # a training half: its teacher is evaluated by a later job
                continue
            stage(f'evaluate-{arm}', [py, st, 'evaluate', *swap(flat), *evaluation, '--variant', arm,
                '--out', f'evidence/evaluate-{arm}', '--checkpoint', f'{out}/teacher.pt'], evaluate_s)
            if plan.get('eval_terrain'):
                stage(f'evaluate-{arm}-terrain', [py, st, 'evaluate', *swap(flat), '--terrain', plan['eval_terrain'], *terrain_evaluation,
                    '--variant', arm, '--out', f'evidence/evaluate-{arm}-terrain', '--checkpoint', f'{out}/teacher.pt'], evaluate_s)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=redact(str(exc)),
            traceback=redact(traceback.format_exc()[-8000:]))
    finally:
        report['wall_s'] = time.monotonic() - began
        write(evidence / 'worker-report.json', report)
        try:
            publish('final', [evidence / 'worker-report.json'])
        except BaseException as exc:
            report['final_pack_error'] = redact(str(exc))
        completion = dict(schema='cove50.wbc.completion.v1', status=report['status'], packs=report['packs'],
            report=report, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        client.put_object(Bucket=bucket, Key=prefix + '/completion.json', Body=json.dumps(completion, allow_nan=False).encode())
    return dict(status=report['status'], packs=len(report['packs']), wall_s=report['wall_s'], error=report.get('error'))


if __name__ == '__main__':
    raise SystemExit('Dispatched by launch.py; not run locally.')
