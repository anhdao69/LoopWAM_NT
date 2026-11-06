"""Snapshot every branch's report documents into evaluation_report/branch_reports without deleting anything.

For each branch, every Markdown file under plans/ whose content differs from LoopWAM_NT's (or that LoopWAM_NT does not
have) is copied to branch_reports/<branch>/<path>; documents identical to LoopWAM_NT's are already on every branch.
index.json records the source branch, commit, path and git blob id of every copy. Relative links inside a copy point
into its source branch.

Nothing is deleted: when a stored copy changes, or a document is no longer on its branch (or now equals LoopWAM_NT's),
the previous copy is moved to preserved_versions/<date>/<sha256[:16]>/branch_reports/<branch>/<path>, as are previous
versions of index.json. Usage (after `git fetch origin`):

    python3 sync_branch_reports.py DATE [BRANCH=REF ...]

BRANCH=REF overrides the ref of a branch (default origin/BRANCH) or adds a branch that is not on origin yet.
Prints a JSON record of every change for the sync manifest.
"""
import hashlib, json, pathlib, shutil, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
ER = REPO / 'evaluation_report'
OUT = ER / 'branch_reports'
BASE = 'origin/LoopWAM_NT'
DATE = sys.argv[1]
OVERRIDE = dict(a.split('=', 1) for a in sys.argv[2:])


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True, capture_output=True).stdout


def md_blobs(ref):
    rows = git('ls-tree', '-r', ref, '--', 'plans').decode().splitlines()
    return {line.split('\t', 1)[1]: line.split()[2] for line in rows if line.endswith('.md')}


def preserve(path, reason, log):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    rel = path.relative_to(ER)
    target = ER / 'preserved_versions' / DATE / digest[:16] / rel
    if target.exists():
        assert target.read_bytes() == data, target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    path.unlink()
    log.append(dict(path=f'evaluation_report/{rel}', preserved=f'evaluation_report/{target.relative_to(ER)}', sha256=digest, reason=reason))


base = md_blobs(BASE)
branches = {b.strip()[len('origin/'):]: b.strip() for b in git('branch', '-r').decode().splitlines()
            if b.strip().startswith('origin/') and '->' not in b and b.strip() != BASE}
branches.update(OVERRIDE)
index = dict(base=dict(branch='LoopWAM_NT', commit=git('rev-parse', BASE).decode().strip()), branches={})
log, added, updated = [], [], []
want = {}
for b in sorted(branches):
    ref = branches[b]
    docs = {p: blob for p, blob in md_blobs(ref).items() if base.get(p) != blob}
    for path, blob in docs.items():
        want[f'{b}/{path}'] = blob
    index['branches'][b] = dict(commit=git('rev-parse', ref).decode().strip(),
                                documents=[dict(path=p, blob=blob, new_on_branch=p not in base) for p, blob in sorted(docs.items())])
# Copies that are no longer current: keep them, as preserved versions.
for f in sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'index.json'):
    rel = str(f.relative_to(OUT))
    if rel not in want:
        preserve(f, 'no longer a current branch document (removed on its branch or now identical to LoopWAM_NT)', log)
for rel, blob in sorted(want.items()):
    target = OUT / rel
    data = git('cat-file', 'blob', blob)
    if target.exists():
        if target.read_bytes() == data:
            continue
        preserve(target, 'superseded by a newer version on its branch', log)
        updated.append(rel)
    else:
        added.append(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
new_index = (json.dumps(index, indent=2) + '\n').encode()
if (OUT / 'index.json').read_bytes() != new_index:
    preserve(OUT / 'index.json', 'previous branch index', log)
    (OUT / 'index.json').write_bytes(new_index)
print(json.dumps(dict(documents={b: len(v['documents']) for b, v in index['branches'].items()}, added=added, updated=updated,
                      preserved=log), indent=1))
