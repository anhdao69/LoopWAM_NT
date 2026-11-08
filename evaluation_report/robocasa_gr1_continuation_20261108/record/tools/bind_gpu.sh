#!/bin/bash
# torchrun --no-python entry point for the 4-GPU RoboTwin-10 runs on worker-2: pin this rank to the 15 CPUs that the
# 8-GPU layout (resume_v3_20261104/bind_rank.sh, same algorithm) assigns to its PHYSICAL GPU
# CUDA_VISIBLE_DEVICES[LOCAL_RANK], so a run on GPUs 0-3 and one on GPUs 4-7 get disjoint NUMA-local CPUs.
# Affinity only; then runs the trainer exactly as torchrun would (python -u <script> <args>).
set -eu
CG=$(awk -F: '$1==0{print $3}' /proc/self/cgroup)
EFF=$(cat "/sys/fs/cgroup$CG/../../cpuset.cpus.effective" 2>/dev/null || cat "/sys/fs/cgroup$CG/cpuset.cpus.effective")
GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((LOCAL_RANK + 1)))
CPUS=$(python3 -I - "$GPU" 8 "$EFF" /sys/devices/system/node/node0/cpulist /sys/devices/system/node/node1/cpulist <<'PY'
import sys
def parse(s):
    out = []
    for part in s.strip().split(','):
        a, _, b = part.partition('-'); out += range(int(a), int(b or a) + 1)
    return out
rank, world, eff = int(sys.argv[1]), int(sys.argv[2]), parse(sys.argv[3])
nodes = [set(parse(open(f).read())) for f in sys.argv[4:6]]
per = min(15, len(eff) // world)
order = lambda cs: sorted(cs, key=lambda c: (c % 64, c))          # core c next to its hyperthread sibling c+64
pools = [order([c for c in eff if c in n]) for n in nodes]
home = lambda r: 0 if r < world // 2 else 1                          # GPUs 0-3 on NUMA 0, 4-7 on NUMA 1
take = {}
for r in range(world):
    n = home(r); take[r] = pools[n][:per]; pools[n] = pools[n][per:]
for r in range(world):
    while len(take[r]) < per:
        n = 1 - home(r); take[r].append(pools[n].pop(0))
assert len({c for v in take.values() for c in v}) == per * world
print(','.join(map(str, sorted(take[rank]))))
PY
)
echo "[$(date -u +%FT%TZ)] $BIND_TAG rank $RANK local $LOCAL_RANK gpu $GPU pid $$ step-cpuset $EFF -> cpus $CPUS" >> "$BIND_LOG"
exec taskset -c "$CPUS" /mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin/python -u "$@"
