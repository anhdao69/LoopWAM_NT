#!/usr/bin/env python3
"""Run the branch's scripts/v0_init_variant_verify.py (smoke | run) on the H100 cluster and separate the two known
cluster differences from real problems.

The branch verifier is used unchanged except for the canonical donor artifact hash it pins (see h100_build_variant.py).
Known differences of this cluster (identical for the v0 4/2 run here), accepted only in exactly this form:
  * data manifest `text_cache_files`: the 40 T5 text-embedding files were regenerated here, so their sha256 differ;
    the file names (prompt hashes) and valid-token counts must be identical to the v0 release.
Everything else must pass. For `run`, the branch's RUN/fairness.json is kept and RUN/fairness_h100.json is written.
    python h100_verify_variant.py smoke|run --train DIR --reference REF --init VARIANT   (cwd: the worktree)
"""
import json
import sys
from pathlib import Path

import v0_init_variant_verify as verifier
from dense_libero_hf_export import read, strip_paths

H100_CANONICAL_SHA256 = 'a990a5831c63e88ae5143689adad2f2aeb02cb1cd151d61ae0dd992990ac3296'
TEXT_CACHE_PROBLEM = "data manifest differs from the v0 release: ['text_cache_files']"


def text_cache_names_and_tokens_equal(train, reference):
    ref = strip_paths(read(reference / 'data_manifest.json'))['text_cache_files']
    here = strip_paths(read(train / 'data/data_manifest.json'))['text_cache_files']
    drop = lambda files: {name: {k: v for k, v in entry.items() if k != 'sha256'} for name, entry in files.items()}
    return len(ref) == 40 and drop(ref) == drop(here)


def main():
    verifier.CANONICAL_SHA256 = H100_CANONICAL_SHA256
    args = sys.argv[1:]
    mode, train, reference = args[0], Path(args[args.index('--train') + 1]), Path(args[args.index('--reference') + 1])
    problems = []
    real_need = verifier.contract

    def contract(train_, reference_, init, need):
        def record(ok, what):
            if not ok:
                problems.append(what)
            need(ok, what)
        return real_need(train_, reference_, init, record)

    verifier.contract = contract
    sys.argv = ['v0_init_variant_verify.py', *args]
    failure = None
    try:
        verifier.main()
    except SystemExit as exc:
        failure = str(exc)
    if failure is None:
        print(json.dumps(dict(h100_passed=True, known_cluster_exceptions=[])), flush=True)
        return
    exceptions, residual = [], []
    for problem in problems:
        if problem == TEXT_CACHE_PROBLEM and text_cache_names_and_tokens_equal(train, reference):
            exceptions.append(dict(problem=problem, accepted_because='text-embedding files regenerated on this cluster; '
                                   'same 40 prompt-hash file names and valid-token counts as the v0 release'))
        else:
            residual.append(problem)
    # Problems found outside contract() (completion, metrics, finiteness) are always real.
    branch_failure = failure.split(': ', 1)[-1].split('; ')
    residual += [p for p in branch_failure if p not in problems]
    out = dict(h100_passed=not residual, residual_problems=residual, known_cluster_exceptions=exceptions,
               canonical_artifact_sha256=H100_CANONICAL_SHA256, branch_verifier_message=failure)
    if mode == 'run':
        report = read(train.parent / 'fairness.json')
        report.update(h100=out)
        (train.parent / 'fairness_h100.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(out), flush=True)
    if residual:
        raise SystemExit('H100 verification failed: ' + '; '.join(residual))


if __name__ == '__main__':
    main()
