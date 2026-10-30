"""CPU-only contracts shared by the RoboTwin evaluator, continuation, and queue."""
import json
import hashlib
import math
import time
from pathlib import Path

WINDOWS=6075103
UPDATES=47462
PARENT_UPDATES=5*UPDATES
IMMUTABLE=('model','partition','video_loops','action_loops','memory_tokens','history_frame',
           'init_sha256','vae_sha256','normalization_sha256','data_manifest_sha256','seed',
           'global_batch','world','microbatch','shuffle','windows_per_epoch','updates_per_epoch',
           'last_batch_windows','betas','eps','weight_decay','clip','precision','cosine_floor')

def continuation_contract(parent,sha):
    if parent['epochs']!=5 or parent['planned_updates']!=PARENT_UPDATES or parent['planned_windows']!=5*WINDOWS:
        raise ValueError('Continuation requires the complete five-epoch contract')
    return dict(parent,epochs=10,planned_updates=10*UPDATES,planned_windows=10*WINDOWS,
                lr_backbone=3e-5,lr_robot_io=3e-5,warmup_fraction=0.,save_epochs='6,7,8,9,10',
                phase_start_update=PARENT_UPDATES,phase_start_epoch=5,parent_checkpoint_sha256=sha)

def continuation_factor(update):
    step=update-PARENT_UPDATES
    if not 0<=step<PARENT_UPDATES:raise ValueError('Continuation LR step outside epochs6-10')
    return .01+.99*.5*(1+math.cos(math.pi*step/(PARENT_UPDATES-1)))

def validate_cursor(update,windows,epoch,batch):
    if not 0<=epoch<=10 or not 0<=batch<UPDATES or (epoch==10 and batch):
        raise ValueError('Invalid epoch/batch cursor')
    if update!=epoch*UPDATES+batch or windows!=epoch*WINDOWS+128*batch:
        raise ValueError('Invalid update/window cursor')

def validate_parent_state(payload,contract):
    state=payload.get('training_state',{});old=state.get('contract',{})
    if any(old.get(k)!=contract.get(k) for k in IMMUTABLE):raise ValueError('Parent training contract mismatch')
    if old.get('epochs')!=5 or old.get('planned_updates')!=PARENT_UPDATES or old.get('planned_windows')!=5*WINDOWS:
        raise ValueError('Parent is not a five-epoch run')
    if (payload.get('step'),state.get('update'),state.get('windows_seen'),state.get('next_epoch'),state.get('next_batch'))!=(PARENT_UPDATES,PARENT_UPDATES,5*WINDOWS,5,0):
        raise ValueError('Parent must be exactly the completed epoch5 boundary')
    if not payload.get('optimizer',{}).get('state'):raise ValueError('Parent optimizer state missing')
    validate_cursor(state['update'],state['windows_seen'],5,0)

def reconcile_metrics(path,saved_update,first_update=1):
    """Preserve rolled-back records separately; never silently duplicate updates on resume."""
    path=Path(path)
    if not path.exists():
        if saved_update>=first_update:raise ValueError('Missing pre-checkpoint metrics')
        return
    lines=path.read_text().splitlines();keep=[];extra=[]
    for line in lines:
        try:row=json.loads(line)
        except ValueError:extra.append(line);continue
        if row['update']<=saved_update:keep.append(row)
        else:extra.append(line)
    if [r['update'] for r in keep]!=list(range(first_update,saved_update+1)):
        raise ValueError('Metrics are missing or duplicated before resume checkpoint')
    if extra:
        backup=path.with_name(f'metrics.superseded.{time.time_ns()}.jsonl')
        backup.write_text('\n'.join(extra)+'\n')
        tmp=path.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(x)+'\n' for x in keep));tmp.replace(path)

def validate_eval_summary(summary,sha,tasks,episodes=100):
    expected={f'{t}/{c}' for t in tasks for c in ('demo_clean','demo_randomized')}
    if summary.get('checkpoint_sha256')!=sha or set(summary.get('finished',{}))!=expected:
        raise ValueError('Evaluation checkpoint or task coverage mismatch')
    for key,cell in summary['finished'].items():
        records=cell.get('episodes',[])
        if not cell.get('complete') or cell.get('exit_code')!=0 or len(records)!=episodes:
            raise ValueError(f'Incomplete evaluation: {key}')
        if len({r['seed'] for r in records})!=episodes:
            raise ValueError(f'Duplicate evaluation seeds: {key}')
        if any(type(r['success']) is not bool or r.get('video_verified') is not True for r in records):
            raise ValueError(f'Invalid outcome/video: {key}')

def training_recovery_action(saved_update,target,has_records):
    if saved_update is None:return 'quarantine' if has_records else 'fresh'
    if not PARENT_UPDATES<=saved_update<=10*UPDATES:raise ValueError('Invalid continuation checkpoint update')
    return 'done' if saved_update>=target else 'resume'

def verify_file_hash(path,expected):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    if h.hexdigest()!=expected:raise ValueError(f'Immutable artifact changed:{path}')

def fingerprint_files(paths):
    return {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(map(str,paths))}
