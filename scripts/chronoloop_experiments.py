"""ChronoLoop experiment registry (plans/ChronoLoop.md section 4/5.3). One entry per run.

Each run differs from its comparison partner only in the listed flags. Shared settings
(init, data, schedule, budget, optimizer) live in train_chronoloop.py defaults.
"""
import argparse
import json
import shlex

EXPERIMENTS = {
    'CL-0':       dict(flags=dict(memory_tokens=0, mem_source='none', mem_write='none', action_loops=4, history_frame=0),
                       hf='CL-0_no-memory_a4', job='chrono-CL-0',
                       role='Control: no memory, same traversal file/budget, K_a = 4'),
    'CL-A':       dict(flags=dict(memory_tokens=16, mem_source='learned', mem_write='loop', action_loops=4, history_frame=0),
                       hf='CL-A_mem16-learned-loopwrite_a4', job='chrono-CL-A',
                       role='Default ChronoLoop (Q1 vs CL-0)'),
    'CL-REG':     dict(flags=dict(memory_tokens=16, mem_source='reset', mem_write='loop', action_loops=4, history_frame=0),
                       hf='CL-REG_mem16-reset-registers_a4', job='chrono-CL-REG',
                       role='16 registers reset every query (Q1b vs CL-A)'),
    'CL-W2':      dict(flags=dict(memory_tokens=16, mem_source='learned', mem_write='external', action_loops=4, history_frame=0),
                       hf='CL-W2_mem16-learned-external-updater_a4', job='chrono-CL-W2',
                       role='External 1-block updater, read-only memory in backbone (Q2 vs CL-A)'),
    'CL-0@1':     dict(flags=dict(memory_tokens=0, mem_source='none', mem_write='none', action_loops=1, history_frame=0),
                       hf='CL-0-at1_no-memory_a1', job='chrono-CL-0-at1',
                       role='CL-0 with one action core loop (Q3)'),
    'CL-A@1':     dict(flags=dict(memory_tokens=16, mem_source='learned', mem_write='loop', action_loops=1, history_frame=0),
                       hf='CL-A-at1_mem16-learned-loopwrite_a1', job='chrono-CL-A-at1',
                       role='CL-A with one action core loop (Q3)'),
    'CL-FRAME':   dict(flags=dict(memory_tokens=0, mem_source='none', mem_write='none', action_loops=4, history_frame=3),
                       hf='CL-FRAME_history-frame-k3_a4', job='chrono-CL-FRAME',
                       role='Optional frame-stacking baseline (vs CL-A)'),
}


def train_args(name):
    flags = EXPERIMENTS[name]['flags']
    out = ['--run-name', name]
    for key, value in flags.items():
        out += [f'--{key.replace("_", "-")}', str(value)]
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name', nargs='?')
    p.add_argument('--field', choices=['args', 'hf', 'job', 'dir', 'json'], default='args')
    a = p.parse_args()
    if a.name is None:
        print(json.dumps(EXPERIMENTS, indent=2)); raise SystemExit
    e = EXPERIMENTS[a.name]
    print({'args': lambda: shlex.join(train_args(a.name)), 'hf': lambda: e['hf'], 'job': lambda: e['job'],
           'dir': lambda: e['hf'], 'json': lambda: json.dumps(e)}[a.field]())
