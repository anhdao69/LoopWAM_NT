import sys, torch, json
for path in sys.argv[1:]:
    p = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    print(path)
    for k, v in p.items():
        if k in ("mot", "proprio_encoder", "optimizer"):
            print(" ", k, f"<{len(v)} tensors>"); continue
        s = json.dumps(v, default=str)
        print(" ", k, s[:300])
