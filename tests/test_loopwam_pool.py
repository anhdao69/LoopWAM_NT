from evaluate_loopwam_pool import make_jobs


def test_episode_groups_preserve_original_environment_lifecycle():
 from evaluate_loopwam_pool import episode_groups,environment_key
 jobs=make_jobs(['libero_10'],[42,42,43],range(10),10)
 groups=episode_groups(jobs)
 assert len(groups)==30 and sum(len(g) for g in groups)==300
 assert len({environment_key(g[0]) for g in groups})==30
 for group in groups:
  assert [j['episode'] for j in group]==list(range(10))
  assert len({environment_key(j) for j in group})==1
 assert sorted(j['id'] for g in groups for j in g)==sorted(j['id'] for j in jobs)
