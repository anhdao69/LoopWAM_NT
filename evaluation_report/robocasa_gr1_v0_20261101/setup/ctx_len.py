import glob, json, sys, torch
files = sorted(glob.glob(sys.argv[1] + "/*.pt")); lens = []
x = torch.load(files[0], map_location="cpu", weights_only=False)
print(type(x), {k: (tuple(v.shape), v.dtype) for k, v in x.items()} if isinstance(x, dict) else None)
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    m = d["mask"] if "mask" in d else d["context_mask"]
    e = d["context"] if "context" in d else d["embedding"]
    L = int(m.sum()); lens.append(L)
    tail_zero = bool((e.reshape(-1, e.shape[-1])[L:] == 0).all()); assert bool(m.reshape(-1)[:L].all()) and not bool(m.reshape(-1)[L:].any())
print(json.dumps(dict(files=len(files), max_len=max(lens), min_len=min(lens), tail_zero_last=tail_zero)))
