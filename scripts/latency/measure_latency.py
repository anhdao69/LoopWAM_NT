#!/usr/bin/env python3
"""All-model batch-one action-query latency on one idle H100 (LIBERO, two 224x224 cameras, 32-action chunk).

Every query replays one raw observation from record_observations.py and is timed from the raw simulator observation
to the denormalized host action chunk (preprocessing, host-to-device copies, online VAE encode, visual prefill, action
denoising, device-to-host copy and denormalization), with CUDA synchronized before the clock starts. Text embeddings
come from the T5 cache (as in every evaluation); simulator stepping, text encoding and model loading are excluded.
Every query uses seed 42, as in the matched evaluation protocol.

Backends per model and inference configuration:
  eager  the model's native eager inference (the reference for exactness).
  graph  the CUDA-graph backend used by the matched evaluations (prefill + denoising captured; VAE eager).
  fast   graph + BF16-stored Linear weights + one CUDA graph for the whole query including the VAE encode.
Each non-eager backend must reproduce the eager raw action chunk bit for bit on every compared observation; the
result file records the counts. Run one process at a time on an otherwise idle GPU.
"""
import argparse, hashlib, json, pickle, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stats(ms):
    import numpy as np
    a = np.asarray(ms)
    return dict(n=int(a.size), mean_ms=float(a.mean()), median_ms=float(np.median(a)), p5_ms=float(np.percentile(a, 5)),
                p95_ms=float(np.percentile(a, 95)), std_ms=float(a.std(ddof=1)), min_ms=float(a.min()))


def time_backend(query, records, warmup, reset=None):
    """query(record) -> (raw action tensor, host action, preprocessing ms). Returns timings and raw actions.
    reset() (stateful models) restarts the episode memory before the warmup and before the timed pass, so every
    backend sees the same query sequence."""
    import torch
    reset and reset()
    for r in records[:warmup]:
        query(r)
    reset and reset()
    total, pre, raw = [], [], []
    for r in records:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action, host, pre_ms = query(r)
        total.append((time.perf_counter() - t0) * 1e3)
        pre.append(pre_ms)
        raw.append(action)
    return total, pre, raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-root", required=True, help="worktree whose model code builds this family")
    p.add_argument("--family", required=True)
    p.add_argument("--model", required=True, help="display name")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--configs", required=True, help="comma-separated inference configurations (schedules / loops)")
    p.add_argument("--stats", help="dataset_stats.json (default: next to the checkpoint)")
    p.add_argument("--vae-path", required=True)
    p.add_argument("--text-cache-dir", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--observations-sha256", required=True)
    p.add_argument("--eager-stride", type=int, default=5, help="eager is timed on every k-th observation")
    p.add_argument("--graph-stride", type=int, default=1,
                   help="graph/compile are timed on every k-th observation (fast always uses all of them)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eager-warmup", type=int, default=None, help="warmup queries for eager (default: --warmup)")
    p.add_argument("--backends", default="eager,graph,fast")
    p.add_argument("--extra", default="{}", help="JSON of family-specific loader arguments")
    p.add_argument("--limit", type=int, default=0, help="smoke test on this many observations (0 = all)")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.code_root).resolve()
    sys.path[:0] = [str(HERE), str(root), str(root / "src")]
    import numpy as np, torch
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    blob = Path(a.observations).read_bytes()
    if hashlib.sha256(blob).hexdigest() != a.observations_sha256:
        raise SystemExit("observation file does not match the expected sha256")
    records = pickle.loads(blob)
    if a.limit:
        records = records[::len(records) // a.limit][:a.limit]     # smoke tests only
    import families
    model = families.build(a.family, a, json.loads(a.extra))
    backends = a.backends.split(",")
    result = dict(model=a.model, family=a.family, checkpoint=a.checkpoint, code_root=str(root),
                  code_commit=families.git_head(root), gpu=torch.cuda.get_device_name(0), host=platform.node(),
                  torch=torch.__version__, cuda=torch.version.cuda, observations=len(records),
                  observations_sha256=a.observations_sha256, limit=a.limit, warmup=a.warmup, info=model.info(),
                  configs={})
    stateful = getattr(model, "stateful", False)
    if stateful:
        a.eager_stride = a.graph_stride = 1     # identical query sequences are required to compare stateful backends
    if a.eager_stride % a.graph_stride:
        raise SystemExit("--eager-stride must be a multiple of --graph-stride")
    stride = dict(eager=a.eager_stride, graph=a.graph_stride, compile=a.graph_stride, fast=1)
    configs = a.configs.split(",")
    result["configs"] = {c: {} for c in configs}
    result.update(eager_stride=a.eager_stride, graph_stride=a.graph_stride,
                  eager_warmup=a.warmup if a.eager_warmup is None else a.eager_warmup)
    refs = {}
    # Backend-major: the in-place BF16 weight conversion of `fast` happens once, after all FP32 measurements.
    for backend in backends:
        if backend == "fast":
            result["fast_converted_linear_modules"] = model.enable_fast()
        for config in configs:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            recs = records[::stride[backend]]
            warmup = result["eager_warmup"] if backend == "eager" else a.warmup
            try:
                total, pre, raw = time_backend(lambda r: model.query(r, config, backend), recs, warmup,
                                               getattr(model, "reset", None))
            except Exception as error:      # an optional backend (e.g. Inductor compile) may not support a model
                if backend == "eager":
                    raise
                import traceback
                traceback.print_exc()
                result["configs"][config][backend] = dict(error=f"{type(error).__name__}: {error}"[:2000])
                print(json.dumps({"model": a.model, "config": config, "backend": backend, "error": str(error)[:300]}),
                      flush=True)
                continue
            e = dict(stats(total), preprocess=stats(pre), peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
            if backend == "eager":
                refs[config] = raw
            elif config in refs:
                cmp = raw[::a.eager_stride // stride[backend]]     # the observations eager was timed on
                e["exact_vs_eager"] = sum(bool(torch.equal(x, y)) for x, y in zip(refs[config], cmp))
                e["compared_vs_eager"] = len(refs[config])
                e["max_abs_diff_vs_eager"] = max(float((x - y).abs().max()) for x, y in zip(refs[config], cmp))
            result["configs"][config][backend] = e
            result["configs"][config].setdefault("config", model.config_info(config))
            # One operating point at a time: drop this configuration's captured graphs before the next one.
            model.release_graphs()
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            print(json.dumps({"model": a.model, "config": config, "backend": backend,
                              **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()
                                 if not isinstance(v, dict)}}), flush=True)
    if "fast" in backends and len(configs) > 1:
        # Memory with every configuration's fast graph resident at once (one deployed elastic checkpoint).
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        keep = []
        for config in configs:
            model.query(records[0], config, "fast")
            keep.append(model.graphs[config] if hasattr(model, "graphs") else None)
        torch.cuda.synchronize()
        result["all_fast_graphs_resident"] = dict(graphs=len(configs),
            allocated_gb=torch.cuda.memory_allocated() / 1e9, peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            reserved_gb=torch.cuda.memory_reserved() / 1e9)
        print(json.dumps({"model": a.model, "all_fast_graphs_resident": result["all_fast_graphs_resident"]}), flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
