import hashlib,json,shutil,subprocess,time
from pathlib import Path
from run_full_mix_job import source_hashes,write
r=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/mix_full_20261010')
assert json.loads((r/'status.json').read_text())['stage']=='prepared_all_trials'
assert not (r/'failed.json').exists()
prepared=json.loads((r/'prepared.json').read_text())
trials=json.loads((r/'all_benchmarks.json').read_text())
best=min(trials,key=lambda t:t['steady_seconds'])
assert {t['backend'] for t in trials}=={'ddp','zero1','zero2'}
assert all(prepared['config'].get(k,False)==best[k] for k in ['backend','microbatch','checkpoint_blocks'])
smoke=r/('native_smoke_checkpoint' if best['checkpoint_blocks'] else 'native_smoke')
assert json.loads((smoke/'fairness.json').read_text())['passed']
assert '9 passed' in (r/'checkpoint_tests.log').read_text()
base=r.parent/'source_kv_3a89bfa'
changed=[str(p.relative_to(base)) for root in ['src','scripts'] for p in (base/root).rglob('*.py') if p.read_bytes()!=(Path.cwd()/p.relative_to(base)).read_bytes()]
assert changed==['scripts/train_loopwam.py'],changed
checkpoint_bytes=(smoke/'latest.pt').stat().st_size
if best['backend']!='ddp':checkpoint_bytes+=sum(p.stat().st_size for p in (smoke/'deepspeed').rglob('*') if p.is_file())
minimum=checkpoint_bytes*9
free=shutil.disk_usage(r).free
assert free>=minimum,(free,minimum)
release=dict(created_unix=time.time(),base_revision='3a89bfac5cfccad66bfebbe00014f7f739a78adc',local_implementation_revision='713c32f',snapshot_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes=source_hashes(),prepared_path=str(r/'prepared.json'),prepared_sha256=hashlib.sha256((r/'prepared.json').read_bytes()).hexdigest(),minimum_free_bytes=minimum,free_bytes_at_release=free,checkpoint_bytes=checkpoint_bytes,selected=prepared['config'],native_training_hours=prepared['training_hours'],gates=dict(checkpoint_tests_passed=9,native_updates=10,fairness_passed=True,all_backends_measured=True),benchmarks=[{k:t[k] for k in ['backend','microbatch','checkpoint_blocks','steady_seconds']} for t in trials])
assert not (r/'release.json').exists()
write(r/'release.json',release)
print(json.dumps({k:v for k,v in release.items() if k!='source_hashes'},indent=2))
