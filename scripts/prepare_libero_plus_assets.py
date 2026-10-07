#!/usr/bin/env python3
"""Extract assets needed by all four Plus suites from the official archive."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import zipfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--archive',required=True)
    a=p.parse_args()
    root=Path(a.repo)/'libero/libero'
    used=set()
    for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
        for path in (root/'bddl_files'/suite).glob('*.bddl'):
            used.update(re.findall(r'-\s*([\w]+)',path.read_text()))
    tree=ast.parse((root/'envs/objects/custom_objects.py').read_text())
    needed=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            key='_'.join(re.sub(r'([A-Z0-9])',r' \1',node.name).split()).lower()
            if key in used:
                needed.extend(n.value for n in ast.walk(node) if isinstance(n,ast.Constant)
                    and isinstance(n.value,str) and n.value.startswith('assets/new_objects/')
                    and n.value.endswith('.xml'))
    model_prefixes={str(Path(path).relative_to('assets').parents[2])+'/' for path in needed}
    archive=zipfile.ZipFile(a.archive)
    prefix=next(x.filename for x in archive.infolist() if x.filename.endswith('/assets/'))
    entries=[]
    for info in archive.infolist():
        if not info.filename.startswith(prefix) or info.is_dir():continue
        relative=info.filename[len(prefix):]
        if relative.startswith('new_objects/') and not any(relative.startswith(s) for s in model_prefixes):
            continue
        if '..' in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError('Invalid archive path')
        entries.append((info,root/'assets'/relative))
    def extract(item):
        info,path=item
        if path.exists() and path.stat().st_size==info.file_size:return
        path.parent.mkdir(parents=True,exist_ok=True)
        with archive.open(info) as source,path.open('wb') as dest:
            shutil.copyfileobj(source,dest)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(extract,entries)):
            if i%1000==0:print(json.dumps(dict(extracted=i+1,total=len(entries))),flush=True)
    for path in needed:
        if not (root/path).is_file():raise ValueError('Required custom model missing: '+path)
    (root/'assets_manifest.json').write_text(json.dumps(dict(archive=str(a.archive),
        extracted_files=len(entries),required_custom_models=needed,
        scope='All four official Plus suite BDDL object types; complete base/scene/texture assets; archive-only unused custom models omitted'),indent=2))


if __name__=='__main__':main()
