"""Snapshot every branch's report documents into evaluation_report/branch_reports (run after `git fetch origin`).

For each remote branch, every Markdown file under plans/ whose content differs from origin/LoopWAM_NT (or that
LoopWAM_NT does not have) is copied to branch_reports/<branch>/<path>; documents identical to LoopWAM_NT's are
already on every branch. index.json records the source branch, commit, path and git blob id of every copy.
Relative links inside a copy point into its source branch.
"""
import json, pathlib, shutil, subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / 'evaluation_report' / 'branch_reports'
BASE = 'origin/LoopWAM_NT'


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True, capture_output=True).stdout


def md_blobs(ref):
    rows = git('ls-tree', '-r', ref, '--', 'plans').decode().splitlines()
    return {line.split('\t', 1)[1]: line.split()[2] for line in rows if line.endswith('.md')}


base = md_blobs(BASE)
shutil.rmtree(OUT, ignore_errors=True)  # documents deleted on a branch must not survive
branches = sorted(b.strip()[len('origin/'):] for b in git('branch', '-r').decode().splitlines()
                  if '->' not in b and b.strip() != BASE)
index = dict(base=dict(branch='LoopWAM_NT', commit=git('rev-parse', BASE).decode().strip()), branches={})
for b in branches:
    ref = f'origin/{b}'
    commit = git('rev-parse', ref).decode().strip()
    docs = {p: blob for p, blob in md_blobs(ref).items() if base.get(p) != blob}
    for path, blob in sorted(docs.items()):
        target = OUT / b / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git('cat-file', 'blob', blob))
    index['branches'][b] = dict(
        commit=commit, date=git('log', '-1', '--format=%cI', ref).decode().strip(),
        subject=git('log', '-1', '--format=%s', ref).decode().strip(),
        documents=[dict(path=p, blob=blob, new_on_branch=p not in base) for p, blob in sorted(docs.items())])
(OUT / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
print(json.dumps({b: len(v['documents']) for b, v in index['branches'].items()}))
