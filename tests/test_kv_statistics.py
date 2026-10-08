import pytest
from report_kv_campaign import wilson,paired,decision,two_proportion


def test_preregistered_thresholds():
 assert decision(267,300)=='access hypothesis supported'
 assert decision(252,300)=='action-depth hypothesis supported'
 assert decision(253,300)==decision(266,300)=='inconclusive'
 with pytest.raises(ValueError):decision(89,100)
 assert wilson(0,100)[0]==0
 assert 0.82<wilson(90,100)[0]<.83
 assert two_proportion(90,100,90,100)['p_two_sided']==1


def test_exact_paired_mcnemar_and_identity_guard():
 a=[dict(base_seed=42,suite='libero_10',task=0,episode=i,success=True) for i in range(5)]
 b=[dict(r,success=False) for r in a]
 assert paired(a,b)['p_exact']==.0625
 assert paired(a,a)['p_exact']==1
 with pytest.raises(ValueError):paired(a,b[:-1])


def test_expanded_report_preserves_original_decision(tmp_path):
 from report_kv_followup import expanded_report
 import json
 for label in ('concat','aligned'):
  path=tmp_path/f'expanded_{label}_seed42';path.mkdir()
  rows=[dict(base_seed=42,suite='libero_10',task=t,episode=e,success=(e<45)) for t in range(10) for e in range(50)]
  (path/'summary.json').write_text(json.dumps(dict(total_episodes=500,mode='final_rollout',episodes=rows)))
 result=expanded_report(tmp_path)
 assert result['original_decision']=='inconclusive' and result['new_decision'] is None
 assert result['models']['concat']['successes']==450 and result['paired']['n']==500
 assert (tmp_path/'expanded_analysis/report.md').exists()


def test_full_report_requires_all_seed_suite_episode_identities():
 from report_kv_followup import summarize_full_rows
 suites=['libero_spatial','libero_object','libero_goal','libero_10']
 rows=[dict(base_seed=s,suite=suite,task=t,episode=e,success=True) for s in (42,43,44) for suite in suites for t in range(10) for e in range(10)]
 assert summarize_full_rows(rows)['successes']==1200
 with pytest.raises(ValueError):summarize_full_rows(rows[:-1]+[rows[0]])


def test_long_report_end_to_end_and_atomic_completion(tmp_path,monkeypatch):
 import json,sys
 from report_kv_campaign import main
 cr=tmp_path/'job0/campaign';mr=tmp_path/'job1/campaign'
 def save(path,obj):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
 def rows(seed,wins):return [dict(base_seed=seed,suite='libero_10',task=t,episode=e,success=t*10+e<wins) for t in range(10) for e in range(10)]
 save(mr.parent/'control_aligned_v4a1/summary.json',dict(episodes=sum([rows(s,k) for s,k in zip((42,43,44),(81,82,83))],[])))
 save(cr.parent/'control_repeat_v4a4/summary.json',dict(episodes=[dict(base_seed=seed,task=t,episode=e,suite='libero_10',success=(t*10+e<count)) for seed,count in [(43,87),(44,91)] for t in range(10) for e in range(10)]))
 for mode,root in [('concat',cr),('mix',mr)]:
  run=root/f'{mode}_long';train=run/'train'
  for seed in (42,43,44):save(run/f'eval_seed{seed}/summary.json',dict(mode='final_rollout',total_episodes=100,episodes=rows(seed,87)))
  save(train/'timing.json',dict(elapsed_training_seconds=100,status='complete'))
  records=[dict(update=i,epoch=(i-1)//725+1,loss_video=.1,loss_action=.2,grad_norm=.3) for i in range(1,7251)]
  for epoch in range(1,11):records.append(dict(event='kv_diagnostics',epoch=epoch,attention_mass={str(j):[.2]*5 for j in range(6)},weights=[[[.25]*4]*2]*6))
  (train/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 out=cr/'long_analysis';monkeypatch.setattr(sys,'argv',['report','--concat-root',str(cr),'--mix-root',str(mr),'--output',str(out)])
 main();result=json.loads((out/'evidence.json').read_text())
 assert result['decision']=='inconclusive' and (out/'concat_weights.png').exists()
 assert not out.with_name(out.name+'.partial').exists()
 assert '91.67%' in (out/'report.md').read_text()
