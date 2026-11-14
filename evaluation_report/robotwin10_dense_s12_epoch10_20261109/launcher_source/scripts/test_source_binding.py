import hashlib,tempfile,unittest
from pathlib import Path
import verify_runtime as runtime
class SourceBinding(unittest.TestCase):
 def function(self):
  f=getattr(runtime,'pin_baseline_sources',None);self.assertIsNotNone(f,'Baseline source pinning is missing');return f
 def test_baseline_evaluator_and_graph_are_pinned(self):
  fn=self.function()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=root/'evaluate.py';g=root/'cuda_graph.py';a.write_text('original evaluator');g.write_text('original graph')
   hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a,g]};hashes[str(root/'runtime_manifest.json')]='obsolete deployment manifest';bound={}
   self.assertEqual(fn({'runtime_hashes':hashes},bound),2)
   self.assertEqual(set(bound),{str(a),str(g)})
   for p in [a,g]:self.assertEqual(bound[str(p)]['sha256'],hashes[str(p)])
 def test_graph_drift_is_rejected_before_new_hashes_are_accepted(self):
  fn=self.function()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'cuda_graph.py';p.write_text('baseline');expected=hashlib.sha256(p.read_bytes()).hexdigest();p.write_text('changed before smoke')
   with self.assertRaises((AssertionError,ValueError)):fn({'runtime_hashes':{str(p):expected}}, {})
if __name__=='__main__':unittest.main()
