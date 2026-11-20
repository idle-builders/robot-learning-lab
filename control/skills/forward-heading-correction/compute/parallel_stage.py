"""Literal bounded evaluation-only parent supervision inserted in remote worker."""
PAIR_SOURCE = r'''
    def pair_stages(cases, checkpoint):
        if len(cases) != 2 or any(c['mode'] != 'replay' for c in cases):
            raise ValueError('Only two unchanged evaluation children may overlap')
        required = plan['transport_reserve_s'] + sum(c['process_s']+c['join_s']+c['pack_s'] for c in children if c['name'] not in closed)
        if budget-(time.monotonic()-began) < required:
            raise TimeoutError('Protected full pair/remaining transport budget unavailable')
        jobs=[];errors=[]
        try:
            for case in cases:
                name=case['name'];child=next(c for c in children if c['name']==name)
                argv=common.command(case,checkpoint,evidence,work)
                row=dict(name=name,command=argv,status='starting',evaluation_group=[c['name']for c in cases])
                report['stages'].append(row);report['case_disposition'][name]='attempted'
                log=(evidence/(name+'.log')).open('wb');job=dict(case=case,child=child,row=row,log=log,process=None,started=time.monotonic(),timed_out=False)
                jobs.append(job)
                process=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,WANDB_MODE='offline',WANDB_SILENT='true',PYTHONDONTWRITEBYTECODE='1'))
                job['process']=process;row.update(pid=process.pid,status='running',spawned_worker_s=time.monotonic()-began)
                report['evaluation_peak_children']=max(report.get('evaluation_peak_children',0),len(jobs))
                try:progress()
                except Exception as exc:row['post_spawn_progress_error']=safe_error(exc)
            active=list(jobs)
            while active:
                for job in active[:]:
                    process=job['process'];child=job['child'];code=process.poll()
                    if code is None and time.monotonic()-job['started']>=child['process_s']:
                        job['timed_out']=True
                        try:os.killpg(process.pid,signal.SIGINT)
                        except ProcessLookupError:pass # Natural exit raced the signal; still join.
                        try:code=process.wait(timeout=child['join_s'])
                        except subprocess.TimeoutExpired:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            code=process.wait()
                    if code is not None:
                        job['row'].update(status='closed',exit_code=code,operator_budget_censored=job['timed_out'],
                            exit_detected_worker_s=time.monotonic()-began,wall_s=time.monotonic()-job['started'])
                        active.remove(job)
                if active:time.sleep(.05)
        finally:
            # Always visit every child before any log close or pack operation.
            # Signal races and one finalizer error cannot strand its sibling.
            for job in jobs:
                process=job['process'];row=job['row'];child=job['child']
                if process is None:continue
                try:
                    if process.poll() is None:
                        try:os.killpg(process.pid,signal.SIGINT)
                        except ProcessLookupError:pass
                        except BaseException as exc:row['signal_error']=safe_error(exc);errors.append(exc)
                    try:process.wait(timeout=child['join_s'])
                    except subprocess.TimeoutExpired:
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        process.wait()
                except BaseException as exc:
                    row['join_error']=safe_error(exc);errors.append(exc)
                row.setdefault('exit_code',process.returncode);row.setdefault('operator_budget_censored',True)
                row.update(status='closed' if process.returncode is not None else 'join_incomplete')
                row.setdefault('wall_s',time.monotonic()-job['started'])
            for job in jobs:
                try:job['log'].close()
                except BaseException as exc:job['row']['log_close_error']=safe_error(exc);errors.append(exc)
            for job in jobs:
                row=job['row'];name=job['case']['name'];closed.add(name)
                original=evidence/name/'report.json'
                if original.exists():
                    try:row['native']=json.loads(original.read_text())
                    except (ValueError,OSError)as exc:row['report_read_error']=safe_error(exc)
                try:
                    publish(name,[evidence/name,evidence/(name+'.log')])
                    if row.get('status')!='closed' or row.get('exit_code') or row.get('operator_budget_censored') or 'log_close_error'in row or 'join_error'in row:
                        raise RuntimeError('Original paired child incomplete; no retry: '+name)
                    common.completed(job['case'],evidence/name)
                    report['case_disposition'][name]='completed'
                except BaseException as exc:
                    row['closure_error']=safe_error(exc);errors.append(exc)
        if errors:raise errors[0]
'''
