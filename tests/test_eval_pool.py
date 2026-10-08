import importlib.util
from pathlib import Path
import pytest

def module():
 p=Path(__file__).resolve().parents[1]/'scripts/evaluate_loopwam_pool.py'
 s=importlib.util.spec_from_file_location('eval_pool',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_three_rounds_cover_same_seed_grid_without_duplicates():
 m=module();jobs=m.make_jobs(m.SUITES,[42,42,42],range(10),10)
 assert len(jobs)==1200 and len({j['id'] for j in jobs})==1200
 for round_id in range(3):
  js=[j for j in jobs if j['round']==round_id]
  assert len(js)==400
  assert {(j['suite'],j['task'],j['episode']) for j in js}=={(s,t,e) for s in m.SUITES for t in range(10) for e in range(10)}
  assert all(j['seed']==42+j['task']*100000+j['episode']*1000 for j in js)

@pytest.mark.parametrize('seeds,tasks,eps',[([],range(10),10),([42],[0,0],10),([42],[-1],10),([42],[10],10),([42],[0],0)])
def test_invalid_grids_rejected(seeds,tasks,eps):
 with pytest.raises(ValueError):module().make_jobs(module().SUITES,seeds,tasks,eps)

def test_round_seed_selection_preserves_formula():
 m=module();jobs=m.make_jobs(['libero_10'],[42,43,44],[2],2)
 assert [j['seed'] for j in sorted(jobs,key=lambda j:(j['round'],j['episode']))]==[200042,201042,200043,201043,200044,201044]

def test_summary_rejects_missing_duplicate_wrong_seed_and_hash(tmp_path):
 m=module();jobs=m.make_jobs(['libero_10'],[42],[0],2)
 rows=[]
 for j in jobs:
  p=tmp_path/f"{j['id']}.mp4";p.write_bytes(b'video')
  rows.append(dict(**j,success=True,video=str(p),checkpoint_sha256='hash',mode='final_rollout'))
 assert m.verify_results(jobs,rows,'hash','final_rollout')['successes']==2
 for bad in (rows[:1],[rows[0],rows[0]],[dict(rows[0],seed=999),rows[1]],[dict(rows[0],checkpoint_sha256='wrong'),rows[1]]):
  with pytest.raises(ValueError):m.verify_results(jobs,bad,'hash','final_rollout')
