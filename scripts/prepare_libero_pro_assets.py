#!/usr/bin/env python3
"""Fetch official PRO states/BDDL and generate its environment dimension."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import random
import shutil
import time
import urllib.request


def fetch(item):
    remote, local = item
    if local.exists():
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve('https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/resolve/main/'+remote,
                                       local.with_suffix(local.suffix+'.tmp'))
            local.with_suffix(local.suffix+'.tmp').replace(local)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2*(attempt+1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--generate-env', action='store_true')
    a=p.parse_args()
    root=Path(a.repo).resolve(); benchmark=root/'libero/libero'
    suites=['libero_spatial','libero_object','libero_goal','libero_10']
    if not a.generate_env:
        files=[]
        for kind in ['bddl_files','init_files']:
            for base in suites:
                for dimension in ['object','swap','lan','task']:
                    folder=f'{kind}/{base}_{dimension}'
                    url='https://huggingface.co/api/datasets/zhouxueyang/LIBERO-Pro/tree/main/'+folder+'?recursive=false&limit=1000'
                    for item in json.load(urllib.request.urlopen(url)):
                        if item['type']=='file':
                            files.append((item['path'],benchmark/item['path']))
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(fetch,files))
        print(json.dumps(dict(downloaded_files=len(files))),flush=True)
        return
    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    spec=importlib.util.spec_from_file_location('official_pro_perturbation',root/'perturbation.py')
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    for base in suites:
        generated=Path(mod.process_bddl_file_mixed(str(benchmark/'bddl_files'/base),base,
            mod.PerturbFlags(use_environment=True),
            {'environment':str(root/'libero_ood/ood_environment.yaml')},seed=42))
        target=benchmark/'bddl_files'/(base+'_env'); target.mkdir(exist_ok=True)
        states=benchmark/'init_files'/(base+'_env'); states.mkdir(exist_ok=True)
        for path in sorted(generated.glob('*.bddl')):
            shutil.copy2(path,target/path.name)
            output=states/(path.stem+'.pruned_init')
            if output.exists(): continue
            env=OffScreenRenderEnv(bddl_file_name=str(target/path.name),camera_heights=128,camera_widths=128)
            try:
                initial=[]
                for episode in range(10):
                    seed=42+episode
                    random.seed(seed); np.random.seed(seed); env.seed(seed)
                    env.reset()
                    initial.append(env.get_sim_state().copy())
                torch.save(np.stack(initial),output)
            finally:
                env.close()
            print(json.dumps(dict(event='environment_states_generated',suite=base,task=path.stem,episodes=10)),flush=True)
    (root/'environment_generation.json').write_text(json.dumps(dict(
        source='official process_bddl_file_mixed, environment-only, seed42',
        initial_states='Ten reset simulator states using seeds42..51 per task',
        reason='Official Hugging Face snapshot supplies the other four dimensions but omits environment files'),indent=2))


if __name__=='__main__':
    main()
