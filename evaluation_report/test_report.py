import copy,itertools,unittest
from build_report import validate_grid,build
class ReportTests(unittest.TestCase):
 def setUp(self):
  self.rows=[dict(suite='libero_10',seed=s,task=t,episode=e,initial_state_index=e,success=True,checkpoint_sha256='test') for s,t,e in itertools.product([42,43,44],range(10),range(50))]
 def test_exact_grid(self):validate_grid(self.rows,['libero_10'],'test')
 def test_missing_and_duplicate_rejected(self):
  for rows in [self.rows[:-1],self.rows+[self.rows[0]]]:
   with self.assertRaises(ValueError):validate_grid(rows,['libero_10'],'test')
 def test_identity_state_and_success_rejected(self):
  for field,value in [('checkpoint_sha256','wrong'),('initial_state_index',99),('success',1),('seed',45)]:
   rows=copy.deepcopy(self.rows);rows[0][field]=value
   with self.assertRaises(ValueError):validate_grid(rows,['libero_10'],'test')
 def test_final_refuses_pending_queues(self):
  with self.assertRaisesRegex(ValueError,'queues'):build(True)
if __name__=='__main__':unittest.main()
