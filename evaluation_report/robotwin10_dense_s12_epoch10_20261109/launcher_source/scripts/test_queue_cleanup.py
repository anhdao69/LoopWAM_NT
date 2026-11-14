"""Fault-injection regressions for safe calibration and subprocess ownership."""
import importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
p=Path(__file__).parent/'queue_pipeline.py';spec=importlib.util.spec_from_file_location('q',p);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
class Cleanup(unittest.TestCase):
 def test_initial_receipt_failure_stops_owned_child(self):
  with tempfile.TemporaryDirectory() as d:
   child=Mock(pid=12345,returncode=0);child.poll.return_value=None
   with patch.object(q,'B',Path(d)),patch.object(q,'status'),patch.object(q,'process',return_value={'start_ticks':'123'}),patch.object(q.subprocess,'Popen',return_value=child),patch.object(q,'write',side_effect=[OSError('injected receipt failure'),None]),patch.object(q.os,'killpg') as kill:
    with self.assertRaisesRegex(OSError,'injected receipt failure'):q.run('fake',['python','fake.py'])
    kill.assert_called_once_with(child.pid,q.signal.SIGTERM);child.wait.assert_called_once_with(240)
 def test_low_gpu_margin_stops_calibration_before_scoring(self):
  with tempfile.TemporaryDirectory() as d:
   child=Mock(pid=12345,returncode=0);child.poll.return_value=None
   with patch.object(q,'B',Path(d)),patch.object(q,'status'),patch.object(q,'process',return_value={'start_ticks':'123'}),patch.object(q.subprocess,'Popen',return_value=child),patch.object(q,'write'),patch.object(q.os,'killpg') as kill,patch.object(q,'gpu',side_effect=[dict(total_mib=97887,used_mib=52,free_mib=97835),dict(total_mib=97887,used_mib=91000,free_mib=6887)]):
    with self.assertRaisesRegex(RuntimeError,'GPU memory guard'):q.run('calibration',['python','fake.py'],reserve_mib=8000)
    kill.assert_called_once_with(child.pid,q.signal.SIGTERM);child.wait.assert_called_once_with(240)
if __name__=='__main__':unittest.main()
