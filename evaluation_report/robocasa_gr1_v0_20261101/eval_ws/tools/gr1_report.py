#!/usr/bin/env python3
"""Per-task table for RoboCasa-GR1 official evals (validation JSONs from gr1_validate.py) + StarVLA README reference.
Usage: gr1_report.py OUT.md label=validation.json [label=validation.json ...]"""
import json, re, sys
REF = {'OFT': {}, 'GR00T-N1.6': {}}
ref_rows = """BottleToCabinetClose 51.5 30.0|CanToDrawerClose 13.0 76.0|CupToDrawerClose 8.5 44.0|MilkToMicrowaveClose 14.0 44.0|PotatoToMicrowaveClose 41.5 32.0|WineToCabinetClose 16.5 36.0|FromCuttingboardToBasket 58.0 50.0|FromCuttingboardToCardboardbox 46.5 40.0|FromCuttingboardToPan 68.5 70.0|FromCuttingboardToPot 65.0 54.0|FromCuttingboardToTieredbasket 46.5 38.0|FromPlacematToBasket 58.5 32.0|FromPlacematToBowl 57.5 58.0|FromPlacematToPlate 63.0 52.0|FromPlacematToTieredshelf 28.5 24.0|FromPlateToBowl 57.0 60.0|FromPlateToCardboardbox 43.5 50.0|FromPlateToPan 51.0 66.0|FromPlateToPlate 78.7 68.0|FromTrayToCardboardbox 51.5 44.0|FromTrayToPlate 71.0 56.0|FromTrayToPot 64.5 62.0|FromTrayToTieredbasket 57.0 54.0|FromTrayToTieredshelf 31.5 30.0"""
for r in ref_rows.split('|'):
    k, g, o = r.split(); REF['GR00T-N1.6'][k] = float(g); REF['OFT'][k] = float(o)
def key(task):
    return re.sub(r'^(PnP|PosttrainPnPNovel)', '', task).replace('SplitA', '').replace('Novel', '')
runs = [(a.split('=', 1)[0], json.load(open(a.split('=', 1)[1]))) for a in sys.argv[2:]]
tasks = sorted(runs[0][1]['per_task'])
L = ['| Task | ' + ' | '.join(l for l, _ in runs) + ' | StarVLA-OFT-Qwen3 | GR00T-N1.6 |',
     '|---|' + '---|' * (len(runs) + 2)]
for t in tasks:
    k = key(t)
    L.append(f'| {t} | ' + ' | '.join(f"{100 * r['per_task'][t]['success_rate']:.0f}" for _, r in runs)
             + f" | {REF['OFT'].get(k, float('nan')):.0f} | {REF['GR00T-N1.6'].get(k, float('nan')):.0f} |")
L.append('| **Average** | ' + ' | '.join(f"**{100 * r['mean_success']:.2f}**" for _, r in runs) + ' | 48.8 | 47.6 |')
L.append('')
L.append('Success rate (%) over 50 rollouts per task (any-step success, 720 steps, n_action_steps 12, n_envs 1, seed 0, '
         'EGL); reference columns from starVLA examples/simBenchmarks/Robocasa_tabletop/README.md @13d8ad4.')
for l, r in runs:
    L.append(f"- {l}: step {r['step']}, validation {'VALID' if r['valid'] else 'PROBLEMS ' + str(r['problems'])}, dir `{r['out']}`")
open(sys.argv[1], 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
