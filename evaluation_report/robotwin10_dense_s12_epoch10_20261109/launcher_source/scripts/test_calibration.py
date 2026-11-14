import json,tempfile,unittest,shutil
from pathlib import Path
from calibrate import calibrate,TASKS
SOURCE=Path(__file__).resolve().parent.parent/'finalize_results.py'
class CalibrationTests(unittest.TestCase):
 def exercise(self,b,fail_render=False):
  shutil.copy2(SOURCE,b/'finalize_results.py');calls=[]
  def write(p,d):p.write_text(json.dumps(d))
  def run(name,args,reserve_mib):
   out=Path(args[args.index('--output-dir')+1]);self.assertFalse(out.exists(),'Cached results must not be timed as fresh evaluation');out.mkdir();calls.append(int(args[args.index('--per-gpu')+1]));finished={}
   for task in TASKS:
    for setting in ['demo_clean','demo_randomized']:
     d=out/f'{task}__{setting}'/'attempt_01';d.mkdir(parents=True);(d/'eval.log').write_text('OIDN Error: out of memory' if fail_render else 'Success rate: 2/2')
     finished[f'{task}/{setting}']=dict(complete=True,exit_code=0,directory=str(d),successes=2,episodes=[dict(seed=42+i,success=True,video_verified=True) for i in range(2)])
   write(out/'COMPLETE.json',dict(checkpoint_sha256='testhash',finished=finished))
   return dict(seconds=100 if name.endswith('w4') else 70,peak_gpu_mib=60000,peak_added_mib=59950,memory_margin_mib=37887)
  c=calibrate(b,Path('/unused'),'python',run,lambda *a,**kw:None,lambda:dict(free_mib=97800),write,'testhash')
  return c,calls,run,write
 def test_only_four_workers_even_on_idle_gpu(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,calls,_,_=self.exercise(Path(tmp));self.assertEqual(calls,[4]);self.assertEqual(c['selected']['workers'],4);self.assertEqual(len(c['profiles']),1)
 def test_unmeasured_trial_archived_before_new_timing(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);old=b/'smoke_graph_w4';old.mkdir();(old/'sentinel').write_text('old incomplete trial')
   c,calls,_,_=self.exercise(b);self.assertEqual(calls,[4]);self.assertEqual(len(list((b/'calibration_abandoned').glob('*/smoke_graph_w4/sentinel'))),1)
 def test_renderer_failure_rejects_smoke(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaisesRegex(ValueError,'renderer/resource'):self.exercise(Path(tmp),fail_render=True)
 def test_cached_eight_worker_selection_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);c,_,run,write=self.exercise(b)
   c['selected']['workers']=8;write(b/'calibration.json',c)
   with self.assertRaises((AssertionError,ValueError)):
    calibrate(b,Path('/unused'),'python',run,lambda *a,**kw:None,lambda:dict(free_mib=97800),write,'testhash')
if __name__=='__main__':unittest.main()
