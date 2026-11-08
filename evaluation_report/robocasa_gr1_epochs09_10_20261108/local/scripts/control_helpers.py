"""Small CPU-only lifecycle helpers, shared by launcher and stop command."""
import json,os,pathlib,signal,subprocess,time
TERMINAL={'complete','failed','stopped'}
def verified_inputs(root,epoch,python):
 subprocess.run([python,str(root/'scripts/verify_inputs.py'),str(root),str(epoch)],check=True)
 result=json.loads((root/'evidence'/f'inputs_epoch{epoch:02d}.json').read_text())
 assert result['verified'];return result

def stop_requested_dir(d,root):
 s=json.loads((d/'STATUS.json').read_text())
 if s['phase'] in TERMINAL:return False
 (d/'STOP').touch();pid=s['supervisor_pid'];cmd=pathlib.Path(f'/proc/{pid}/cmdline')
 if cmd.exists() and str(root/'scripts/run.py').encode() in cmd.read_bytes():
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 return True

def cleanup_group(child,timeout=10):
 if child is None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:child.wait(timeout=timeout)
 except subprocess.TimeoutExpired:pass
 # A driver may exit before its descendant simulations; KILL the remaining owned group.
 try:os.killpg(child.pid,signal.SIGKILL)
 except ProcessLookupError:pass
 try:child.wait(timeout=timeout)
 except subprocess.TimeoutExpired:pass
