import pytest
from report_kv_campaign import wilson,paired,decision,two_proportion


def test_preregistered_thresholds():
 assert decision(267,300)=='access hypothesis supported'
 assert decision(252,300)=='action-depth hypothesis supported'
 assert decision(253,300)==decision(266,300)=='inconclusive'
 with pytest.raises(ValueError):decision(89,100)
 assert wilson(0,100)[0]==0
 assert 0.83<wilson(90,100)[0]<.84
 assert two_proportion(90,100,90,100)['p_two_sided']==1


def test_exact_paired_mcnemar_and_identity_guard():
 a=[dict(base_seed=42,suite='libero_10',task=0,episode=i,success=True) for i in range(5)]
 b=[dict(r,success=False) for r in a]
 assert paired(a,b)['p_exact']==.0625
 assert paired(a,a)['p_exact']==1
 with pytest.raises(ValueError):paired(a,b[:-1])
