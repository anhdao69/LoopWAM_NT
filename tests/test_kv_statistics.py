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
