#!/usr/bin/env bash
# Push branch elastic of LoopWAM_elastic using a PAT from the GH_TOKEN environment variable (never written to disk;
# a transient credential helper reads it). Retries transient GitHub errors. Output is redacted.
[ -n "$GH_TOKEN" ] || { echo "no GH_TOKEN in environment; push skipped"; exit 2; }
cd /lustre/fs1/home/an221229/code/LoopWAM_elastic || exit 1
for i in 1 2 3 4 5; do
  git -c credential.helper= -c 'credential.helper=!f() { test "$1" = get && echo username=x-access-token && echo "password=$GH_TOKEN"; }; f' \
    push origin elastic 2>&1 | sed -E 's/(ghp_|github_pat_)[A-Za-z0-9_]+/<redacted>/g'
  [ "${PIPESTATUS[0]}" = 0 ] && { echo "push ok: $(git rev-parse --short HEAD)"; exit 0; }
  echo "push attempt $i failed"; sleep 30
done
exit 1
