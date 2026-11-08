import pathlib,sys
from control_helpers import stop_requested_dir
root=pathlib.Path(__file__).resolve().parents[1];epoch=int(sys.argv[1]);found=False
for phase in ('official_','smoke_'):
 d=root/'results'/phase/f'epoch_{epoch:02d}'
 if (d/'STATUS.json').exists() and stop_requested_dir(d,root):found=True;print('Stop requested:',d)
if not found:print('No active matching supervisor')
