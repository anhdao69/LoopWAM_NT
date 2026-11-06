import json, hashlib, os, sys
run, cache = sys.argv[1], sys.argv[2]
P = json.load(open(f"{run}/shared/data/prompts.json"))
miss = [p["prompt"] for p in P if not os.path.exists(os.path.join(cache, hashlib.sha256(p["prompt"].encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"))]
print("training prompts", len(P), "missing cache", len(miss))
print("windows per prompt min/max", min(p["windows"] for p in P), max(p["windows"] for p in P))
for p in P[:5] + P[-3:]: print(p["prompt_id"], p["windows"], repr(p["text"]))
