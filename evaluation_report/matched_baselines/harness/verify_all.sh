#!/usr/bin/env bash
# Re-verify every local matched run with verify_run.py (2026-10-30, for the report); results in reports/verification.
# Checkpoint digests: the four partition exports downloaded 2026-10-27/28 against their HF LFS sha256 (download logs);
# every other run against the digests in its own manifest (computed from the files verified at download).
ROOT=/data/anh/code/wam_baselines; OUT=$ROOT/reports/verification; mkdir -p "$OUT"; cd "$ROOT"
PY=$ROOT/.venvs/lightwam-libero/bin/python
manifest_sha() { python3 -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1]))["protocol"]["checkpoint_sha256"]))' "runs/$1_matched/manifest.json"; }
check() {  # check <run> <sha or json> <loops json>
  python3 -I matched_eval/verify_run.py "runs/$1_matched" "$2" "$3" --json="$OUT/$1.json" > "$OUT/$1.txt" 2>&1
  echo "$1 exit=$?" >> "$OUT/status.txt"
}
: > "$OUT/status.txt"
{
check fasterwam "$(manifest_sha fasterwam)" null &
check lightwam "$(manifest_sha lightwam)" null &
check loopwam_dense_s12 "$(manifest_sha loopwam_dense_s12)" null &
check loopwam_dense_s30 "$(manifest_sha loopwam_dense_s30)" null &
wait
check loopwam_v0_ka3 "$(manifest_sha loopwam_v0_ka3)" '[2,3,4]' &
check loopwam_v0_ka5 "$(manifest_sha loopwam_v0_ka5)" '[1,1,2,3,4]' &
check loopwam_v0_ka6 "$(manifest_sha loopwam_v0_ka6)" '[1,1,1,2,3,4]' &
check loopwam_partition_p0c6s6 "$(manifest_sha loopwam_partition_p0c6s6)" '[1,2,3,4]' &
wait
check loopwam_partition_p1c6s5 "$(manifest_sha loopwam_partition_p1c6s5)" '[1,2,3,4]' &
check loopwam_init_core16_21 "$(manifest_sha loopwam_init_core16_21)" '[1,2,3,4]' &
check loopwam_partition_p5c6s1 2fa44095e0ef4ead6e8ed35e4d831b8e3fe5c0e3fba65db33615181cf77accd3 '[1,2,3,4]' &
check loopwam_partition_p6c6s0 2dabf8e01924836ed5cb05deb92b734cef9601f3583cadc22b6e306266101eda '[1,2,3,4]' &
wait
check loopwam_partition_p1c9s2_k3 4f0dc3ac79201763a5116b985d7e9bf55346dbbbb3557553f35e7c2d50c1cc9c '[1,2,3]' &
check loopwam_partition_p4c3s5_k7 39f574b8c35b0b52ff0d6a449ce4cc0a320a376680b01edfef153c282b139e7f '[1,2,3,4,5,6,7]' &
wait
}
echo done >> "$OUT/status.txt"
