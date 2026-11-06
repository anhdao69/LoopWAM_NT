import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parent
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class RestartTests(unittest.TestCase):
 def test_preserve_bound_manifest_bytes_on_restart(self):
  m=module('safety','queue_safety_template.py')
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'runtime.json';first=dict(created_unix=10,files={'asset':{'sha256':'abc'}},variant='v0')
   m.publish_runtime(p,first);before=p.read_bytes()
   m.publish_runtime(p,{**first,'created_unix':20})
   self.assertEqual(before,p.read_bytes())
   with self.assertRaises(ValueError):m.publish_runtime(p,{**first,'variant':'wrong'})
 def test_completed_restart_finalizes_without_rerunning_evaluator(self):
  m=module('safety2','queue_safety_template.py')
  with tempfile.TemporaryDirectory() as temp:
   b=Path(temp);(b/'results').mkdir();self.assertFalse(m.reuse_completed(b,lambda *_:None))
   complete=dict(checkpoint_sha256='abc',finished={'x':{}},running={},failed=[],success_rates={'demo_clean':1,'demo_randomized':1})
   (b/'results/COMPLETE.json').write_text(json.dumps(complete));before=(b/'results/COMPLETE.json').read_bytes()
   (b/'results/manifest.json').write_text(json.dumps(dict(checkpoint_sha256='abc',tasks=['x'],episodes=100)))
   (b/'launch.json').write_text(json.dumps(dict(status='running',started_unix=1)))
   checks=[];self.assertTrue(m.reuse_completed(b,lambda *args:checks.append(args)))
   self.assertEqual(len(checks),1);self.assertEqual(before,(b/'results/COMPLETE.json').read_bytes())
   self.assertEqual(json.loads((b/'launch.json').read_text())['status'],'complete')
   del complete['success_rates'];(b/'results/COMPLETE.json').write_text(json.dumps(complete))
   with self.assertRaises(ValueError):m.reuse_completed(b,lambda *_:None)
if __name__=='__main__':unittest.main()
