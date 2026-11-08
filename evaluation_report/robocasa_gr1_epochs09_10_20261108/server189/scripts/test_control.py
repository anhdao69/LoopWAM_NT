import unittest,tempfile,pathlib,json,sys,subprocess,os,time,signal
from control_helpers import verified_inputs,stop_requested_dir,cleanup_group
class Controls(unittest.TestCase):
 def test_fresh_inputs_after_verification(self):
  with tempfile.TemporaryDirectory() as t:
   r=pathlib.Path(t);(r/'scripts').mkdir();(r/'evidence').mkdir();p=r/'evidence/inputs_epoch09.json';p.write_text(json.dumps({'verified':True,'sha':'stale'}));(r/'scripts/verify_inputs.py').write_text("import pathlib,json,sys\np=pathlib.Path(sys.argv[1])/'evidence/inputs_epoch09.json';p.write_text(json.dumps({'verified':True,'sha':'fresh'}))\n")
   self.assertEqual(verified_inputs(r,9,sys.executable)['sha'],'fresh')
 def test_stop_in_every_nonterminal_phase(self):
  with tempfile.TemporaryDirectory() as t:
   r=pathlib.Path(t)
   for phase in ('verifying','retry_wait','running'):
    (r/'STATUS.json').write_text(json.dumps({'phase':phase,'supervisor_pid':999999999}));self.assertTrue(stop_requested_dir(r,r));self.assertTrue((r/'STOP').exists());(r/'STOP').unlink()
   (r/'STATUS.json').write_text(json.dumps({'phase':'complete','supervisor_pid':999999999}));self.assertFalse(stop_requested_dir(r,r));self.assertFalse((r/'STOP').exists())
 def test_cleanup_kills_term_ignoring_process(self):
  p=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'],start_new_session=True,stdout=subprocess.PIPE,text=True)
  self.assertEqual(p.stdout.readline().strip(),'ready');cleanup_group(p,timeout=.1);self.assertIsNotNone(p.poll());p.stdout.close()
 def test_staged_source_change_detected(self):
  with tempfile.TemporaryDirectory() as t:
   r=pathlib.Path(t);subprocess.run(['git','init','-q',t],check=True);(r/'x').write_text('old');subprocess.run(['git','-C',t,'add','x'],check=True);subprocess.run(['git','-C',t,'-c','user.name=test','-c','user.email=t@example.org','commit','-qm','base'],check=True);(r/'x').write_text('changed');subprocess.run(['git','-C',t,'add','x'],check=True);self.assertEqual(subprocess.check_output(['git','-C',t,'diff','HEAD','--name-only'],text=True).strip(),'x')
if __name__=='__main__':unittest.main()
