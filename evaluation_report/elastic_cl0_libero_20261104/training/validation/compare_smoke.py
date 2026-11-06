import json, sys, torch
from pathlib import Path
S = Path(sys.argv[1])
def rows(p):
    out = {}
    for line in open(p):
        r = json.loads(line); out[r["update"]] = r
    return out
a, b = rows(S/"A/metrics.jsonl"), rows(S/"B/metrics.jsonl")
skip = ("seconds", "peak_allocated_gb", "measured_action_blocks")
mism = [(u, k) for u in range(1, 31) for k in set(a[u]) | set(b[u]) if k not in skip and a[u].get(k) != b[u].get(k)]
wa = torch.load(S/"A/latest.pt", map_location="cpu", weights_only=False); wb = torch.load(S/"B/latest.pt", map_location="cpu", weights_only=False)
wsame = all(torch.equal(wa["weights"][k], wb["weights"][k]) for k in wa["weights"]) and all(torch.equal(wa["ema"][k].cpu(), wb["ema"][k].cpu()) for k in wa["ema"])
osame = all(torch.equal(x, y) for sa, sb in zip(wa["optimizer"]["state"].values(), wb["optimizer"]["state"].values()) for x, y in zip(sa.values(), sb.values()) if torch.is_tensor(x))
c = rows(S/"C/metrics.jsonl")
sec = lambda r: sorted(r[u]["seconds"] for u in r if u > 3)
med = lambda s: s[len(s)//2]
print(json.dumps(dict(updates=(len(a), len(b), len(c)), mismatches=mism[:10], metrics_bit_identical=not mism,
    weights_ema_bit_identical=wsame, optimizer_state_bit_identical=osame, status=(wa["status"], wa["update"], wb["update"]),
    median_seconds=dict(E2=med(sec(a)), E0=med(sec(c))), mean_seconds=dict(E2=sum(sec(a))/len(sec(a)), E0=sum(sec(c))/len(sec(c))),
    peak_gb=dict(E2=max(r["peak_allocated_gb"] for r in a.values()), E0=max(r["peak_allocated_gb"] for r in c.values())),
    finite=all(all(v == v for v in (r["local"], r["end"], r["grip"], r["anchor"], r["grad_norm"])) for r in list(a.values()) + list(c.values())),
    schedules_A=[a[u]["schedule"].replace("flat_", "") for u in range(1, 31)]), indent=1))
