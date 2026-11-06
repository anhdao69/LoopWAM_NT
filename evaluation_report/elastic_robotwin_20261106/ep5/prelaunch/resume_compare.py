import json, sys, torch
from pathlib import Path
R = Path(sys.argv[1])
def rows(d): return [json.loads(l) for l in (R / d / "metrics.jsonl").read_text().splitlines()]
A, B = rows("A"), rows("B")
skip = {"seconds", "time", "data_wait_seconds", "peak_allocated_gb"}   # wall clock and allocator high-water marks
metrics_equal = len(A) == len(B) == 30 and all({k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}
                                               for a, b in zip(A, B))
import numpy as np
def eq(x, y):
    if torch.is_tensor(x): return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray): return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)   # numpy RNG state
    if isinstance(x, dict): return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)): return type(x) == type(y) and len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
    return x == y
a, b = (torch.load(R / d / "latest.pt", map_location="cpu", weights_only=False) for d in ("A", "B"))
parts = {k: eq(a[k], b[k]) for k in ("weights", "optimizer", "ema", "rng", "contract", "update", "epoch", "next_micro", "windows_seen", "status", "scheduler")}
b20 = torch.load(R / "B_update20_latest.pt", map_location="cpu", weights_only=False)
moved = sum(not torch.equal(b20["weights"][k], a["weights"][k]) for k in a["weights"])
result = dict(metrics_equal=metrics_equal, updates=(len(A), len(B)), parts=parts, tensors_changed_20_to_30=f"{moved}/{len(a['weights'])}",
              resumed_launch=[json.loads(l)["event"] for l in (R / "B.log").read_text().splitlines() if l.startswith('{"event": "stopped"')][:2],
              passed=metrics_equal and all(parts.values()) and moved > 0 and a["update"] == 30)
print(json.dumps(result, indent=1)); sys.exit(0 if result["passed"] else 1)
