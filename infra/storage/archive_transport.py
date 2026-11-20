"""Bounded byte uploads for the private S3 provider, without Expect:100 stalls.

Transport only. The caller still verifies full remote hashes and recovery before
releasing source files. Interrupted multipart uploads are aborted on failure.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib,json,time

def compatible(s3):
    if getattr(s3,'_robotlab_byte_transport',False):return s3
    # Reuse the already-authorized service identity; never serialize credentials.
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError,ClientError
    credentials=s3._request_signer._credentials.get_frozen_credentials()
    config=s3.meta.config.merge(Config(connect_timeout=5,read_timeout=30,max_pool_connections=24,retries={'max_attempts':2,'mode':'standard'}))
    replacement=boto3.client('s3',endpoint_url=s3.meta.endpoint_url,region_name=s3.meta.region_name,
        aws_access_key_id=credentials.access_key,aws_secret_access_key=credentials.secret_key,aws_session_token=credentials.token,config=config)
    s3.close();s3=replacement
    def remove_expect(request,**kwargs):
        if 'Expect'in request.headers:del request.headers['Expect']
    s3.meta.events.register('before-sign.s3',remove_expect)
    def upload_file(Filename,Bucket,Key,ExtraArgs=None,Config=None,**kwargs):
        path=Path(Filename);extra=ExtraArgs or {};chunk=16<<20;start=time.monotonic()
        def journal(phase,**values):
            row=dict(phase=phase,key=Key,**values)
            folder=Path(__file__).resolve().parents[2]/'.cache/storage';folder.mkdir(parents=True,exist_ok=True)
            with(folder/'transfer-attempts.jsonl').open('a')as log:log.write(json.dumps(row)+'\n')
        # A prior attempt may have completed its immutable payload before a
        # later verification request failed. store_pack still reads/hash-checks
        # the entire remote object, so this avoids only a redundant upload.
        try:
            head=s3.head_object(Bucket=Bucket,Key=Key)
            if extra.get('Metadata',{}).get('sha256') and head['ContentLength']==path.stat().st_size and head.get('Metadata',{}).get('sha256')==extra['Metadata']['sha256']:
                # A timed-out multipart attempt can leave an object whose HEAD
                # matches but whose body is unreadable (BrokenMPU). Reuse only a
                # readable payload; otherwise replace it with the same bytes.
                try:
                    s3.get_object(Bucket=Bucket,Key=Key,Range='bytes=0-0')['Body'].read()
                    journal('existing-payload-will-be-fully-read-back');return
                except ClientError as error:
                    journal('existing-payload-unreadable-reuploading',error=error.response['Error']['Code'])
        except ClientError as error:
            if error.response['Error']['Code']not in {'404','NoSuchKey','NotFound'}:raise
        # Stale incomplete uploads of this exact key would otherwise linger.
        for stale in s3.list_multipart_uploads(Bucket=Bucket,Prefix=Key).get('Uploads',[]):
            if stale['Key']==Key:
                s3.abort_multipart_upload(Bucket=Bucket,Key=Key,UploadId=stale['UploadId']);journal('stale-multipart-aborted',upload_id=stale['UploadId'])
        # These compressed packs fit a bounded ordinary PUT. This avoids the
        # provider's intermittently stalled multipart-part endpoint entirely.
        if path.stat().st_size<=192<<20:
            try:
                journal('ordinary-put-start',bytes=path.stat().st_size)
                s3.put_object(Bucket=Bucket,Key=Key,Body=path.read_bytes(),**extra);journal('ordinary-put-complete');return
            except BotoCoreError as error:
                journal('ordinary-put-fallback',error=type(error).__name__)
                print(json.dumps(dict(transport='multipart-fallback',key=Key,error=type(error).__name__)),flush=True)
        created=s3.create_multipart_upload(Bucket=Bucket,Key=Key,**extra);uid=created['UploadId'];completed=False
        journal('multipart-created',upload_id=uid)
        try:
            pending=set();parts=[];sent=0
            def send(number,body):
                reply=s3.upload_part(Bucket=Bucket,Key=Key,UploadId=uid,PartNumber=number,Body=body)
                return dict(PartNumber=number,ETag=reply['ETag'])
            with path.open('rb')as stream,ThreadPoolExecutor(max_workers=3)as pool:
                number=0
                while body:=stream.read(chunk):
                    number+=1;pending.add(pool.submit(send,number,body));sent+=len(body)
                    if len(pending)>=3:
                        done,pending=wait(pending,return_when=FIRST_COMPLETED)
                        parts.extend(job.result()for job in done)
                    if number%20==0:print(json.dumps(dict(transport='bounded-byte-parts',key_suffix=Key[-16:],queued_MB=round(sent/1e6,1),completed_parts=len(parts),elapsed_s=round(time.monotonic()-start,1))),flush=True)
                parts.extend(job.result()for job in pending)
            s3.complete_multipart_upload(Bucket=Bucket,Key=Key,UploadId=uid,MultipartUpload=dict(Parts=sorted(parts,key=lambda p:p['PartNumber'])))
            completed=True;journal('multipart-complete',upload_id=uid)
        finally:
            if not completed:
                try:
                    s3.abort_multipart_upload(Bucket=Bucket,Key=Key,UploadId=uid);journal('multipart-aborted',upload_id=uid)
                except ClientError as error:
                    if error.response['Error']['Code'] not in {'NoSuchUpload', 'NoSuchKey'}:raise
    s3.upload_file=upload_file;s3._robotlab_byte_transport=True
    return s3
