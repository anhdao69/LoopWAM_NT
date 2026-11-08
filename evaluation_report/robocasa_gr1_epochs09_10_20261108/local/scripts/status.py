import pathlib,json,time,sys,os
root=pathlib.Path(__file__).resolve().parents[1];rows=[]
for d in sorted((root/'results').glob('*/epoch_*')):
 if not (d/'STATUS.json').exists():continue
 s=json.loads((d/'STATUS.json').read_text());s['out']=str(d);s['supervisor_alive']=pathlib.Path(f"/proc/{s['supervisor_pid']}").exists();s['finished_tasks']=len(list((d/'n12').glob('*.json')));vids=list((d/'videos/n12').glob('*/*_success[01].mp4'));s['finished_videos']=len(vids)
 if vids:s['latest_video_utc_seconds']=max(v.stat().st_mtime for v in vids)
 rows.append(s)
print(json.dumps(rows,indent=2))
