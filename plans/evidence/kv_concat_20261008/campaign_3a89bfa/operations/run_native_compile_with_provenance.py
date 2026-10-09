import hashlib,json,subprocess,sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()
checkpoint,output=map(lambda s:Path(s).resolve(),sys.argv[1:])
sidecar=output.with_suffix('.provenance.json')
assert not output.exists() and not sidecar.exists()
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert revision=='3a89bfac5cfccad66bfebbe00014f7f739a78adc'
script=Path('scripts/verify_kv_native_inference.py');script_hash=sha(script);checkpoint_hash=sha(checkpoint)
command=[sys.executable,str(script),'--checkpoint',str(checkpoint),'--output',str(output)]
subprocess.run(command,check=True)
assert sha(checkpoint)==checkpoint_hash and sha(script)==script_hash
result=json.loads(output.read_text());assert result['passed'] and result['repeat_bit_identical']
import torch
payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
provenance=dict(source_revision=revision,test_script_sha256=script_hash,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,version=payload['version'],video_loops=payload['inference_loops'],action_loops=payload.get('action_loops',payload['inference_loops']),action_kv_mode=payload.get('action_kv_mode','aligned'),step=payload['step'],command=command,returncode=0,result_sha256=sha(output),torch_version=str(torch.__version__))
with sidecar.open('x') as f:json.dump(provenance,f,indent=2)
print(json.dumps(provenance),flush=True)
