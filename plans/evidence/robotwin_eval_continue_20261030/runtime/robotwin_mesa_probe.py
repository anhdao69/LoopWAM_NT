from pathlib import Path
import subprocess,json
r=Path('/mnt/data/vmo-ai-task/anhdh35/robotwin_eval_runtime_20261030');m=r/'mesa_layer';layers=m/'layers';layers.mkdir(exist_ok=True)
p=m/'root/usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json';v=json.loads(p.read_text());v['layer']['library_path']=str(m/'root/usr/lib/x86_64-linux-gnu/libVkLayer_MESA_device_select.so');(layers/p.name).write_text(json.dumps(v,indent=2))
(r/'mesa_inner.py').write_text('''import os,sapien
s=sapien.Engine();s.set_renderer(sapien.SapienRenderer());scene=s.create_scene();d=scene.render_system.device
print('RENDER_DEVICE',os.environ['CUDA_VISIBLE_DEVICES'],d.cuda_id,d.pci_string,flush=True)
c=scene.add_camera('test',64,64,1.,.1,10.);scene.step();scene.update_render();c.take_picture();print('RENDER_OK',c.get_picture('Color').shape,flush=True)
''')
cmd=f'set -e; source {r}/eval_env.sh; export VK_LAYER_PATH={layers}; export VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select; export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1; CUDA_VISIBLE_DEVICES=0 DRI_PRIME=pci-0000_1b_00_0 python -u {r}/mesa_inner.py; CUDA_VISIBLE_DEVICES=1 DRI_PRIME=pci-0000_1c_00_0 python -u {r}/mesa_inner.py'
with (r/'mesa_probe.log').open('w') as f:p=subprocess.Popen(['srun','--jobid=5489','--overlap','--ntasks=1','--cpus-per-task=2','--gres=gpu:8','bash','-c',cmd],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
