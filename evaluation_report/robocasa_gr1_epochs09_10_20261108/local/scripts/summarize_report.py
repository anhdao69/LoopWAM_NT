import pathlib,json,datetime,time
report=pathlib.Path('/storage/anhdh35/LoopWAM_NT/evaluation_report/robocasa_gr1_epochs09_10_20261108')
rows=[];details=[]
for epoch,host,folder,live in [(9,'Local RTX PRO 6000','local','local_live_status.json'),(10,'Server189 RTX PRO 5000','server189','server189_live_status.json')]:
 try:states=json.loads((report/live).read_text())
 except (FileNotFoundError,json.JSONDecodeError):states=[]
 state=next((s for s in states if f'/official_/epoch_{epoch:02d}' in s.get('out','')),{});d=report/folder/'results/official_'/f'epoch_{epoch:02d}';cells=[]
 for p in sorted((d/'n12').glob('*.json')):
  x=json.loads(p.read_text());cells.append(x);details.append(f"| {epoch} | {x['env_name'].split('/')[-1]} | {x['successes_first_n']}/{x['n_episodes']} | {100*x['success_rate']:.2f}% |")
 episodes=sum(x['n_episodes'] for x in cells);successes=sum(x['successes_first_n'] for x in cells);valid=None
 if (d/'validation.json').exists():valid=json.loads((d/'validation.json').read_text()).get('valid')
 metric=f'{successes}/{episodes} across completed task cells' if cells else 'Awaiting completed task cells'
 if valid and len(cells)==24:metric=f'{successes}/1200 = {100*successes/1200:.2f}% (validated)'
 rows.append(f"| {epoch} | {host} | {state.get('phase','pending')} | {state.get('finished_videos',0)}/1200 | {len(cells)}/24 | {metric} |")
text='# Live reevaluation summary\n\nUpdated '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' (local UTC). Source report sync runs every60seconds.\n\n| Epoch | Host | Phase | Finished episode videos | Complete task cells | Any-step success |\n|---|---|---|---:|---:|---|\n'+'\n'.join(rows)+'\n\nSuccess comes only from native task JSON, never from video filenames. Each task writes its JSON after all50episodes finish. Partial completed-task rates are not the full24-task result. Smoke episodes are excluded. Environment seed0, policy noise seed42, 24tasks×50episodes, 720steps, action prefix12/32, NFE10, CFG1, all videos.\n\n'
if details:text+='| Epoch | Completed task | Successes | Rate |\n|---|---|---:|---:|\n'+'\n'.join(details)+'\n'
for old,new in [('every60seconds','every 60 seconds'),('all50episodes','all 50 episodes'),('full24-task','full 24-task'),('seed0','seed 0'),('seed42','seed 42'),('24tasks×50episodes','24 tasks × 50 episodes'),('720steps','720 steps'),('prefix12/32','prefix 12/32'),('NFE10','NFE 10'),('CFG1','CFG 1')]:text=text.replace(old,new)
p=report/'SUMMARY.md';temp=p.with_suffix('.tmp');temp.write_text(text);temp.replace(p)
