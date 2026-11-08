import pathlib,hashlib,json,sys,subprocess,torch
root=pathlib.Path(sys.argv[1]); epoch=int(sys.argv[2]); out=root/'evidence'/f'inputs_epoch{epoch:02d}.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
code=root/'code';sys.path.insert(0,str(code/'src'))
from fastwam.datasets.robocasa_gr1 import stats_sha256, GR1_ENV_NAMES
rev=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(); assert rev=='d937167ba61a0df4a9818d1917de6c7155400dff'
assert not subprocess.check_output(['git','-C',str(code),'diff','HEAD','--name-only'],text=True).strip()
ck=root/'models/Robocasa-GR1'/f'epoch_{epoch:02d}.pt'; receipt=json.loads(ck.with_suffix('.pt.uploaded.json').read_text()); csha=sha(ck);assert csha==receipt['sha256'];assert ck.stat().st_size==receipt['size']
pinned=next(x for x in json.loads((root/'evidence/hf_tree.json').read_text()) if x['path']=='Robocasa-GR1/'+ck.name)
assert csha==pinned['lfs']['oid'];assert ck.stat().st_size==pinned['size']
head=torch.load(ck,map_location='cpu',mmap=True,weights_only=False);contract=head['training_state']['contract'];assert head['step']==epoch*23516==receipt['step']
vae=sha(root/'models/vae/Wan2.1_VAE.pth');assert vae==contract['vae_sha256']=='38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'
norm=stats_sha256(json.loads((root/'models/Robocasa-GR1/dataset_stats.json').read_text()));assert norm==contract['normalization_sha256']=='bad54856a2835c9a65c5b6288cd6c8e55660df2d3990777009f22842197be9eb'
cache=root/'models/text_cache'; files=[]
for line in (root/'scripts/text_cache.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);name=name.lstrip('*');p=cache/name;assert sha(p)==digest,name;files.append(name)
assert len(files)==187;assert len(GR1_ENV_NAMES)==24
result=dict(verified=True,epoch=epoch,step=head['step'],checkpoint=str(ck),checkpoint_sha256=csha,checkpoint_size=ck.stat().st_size,code_revision=rev,vae_sha256=vae,normalization_sha256=norm,text_cache_files=len(files),tasks=list(GR1_ENV_NAMES),contract=contract,torch=torch.__version__)
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('contract','tasks')}))
