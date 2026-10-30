"""Markdown tables for the public-WAM latency report from results/<model>_<node>.json.
Usage: build_tables.py <results dir> <out.md>"""
import json, sys
from pathlib import Path

ORDER = ["fastwam", "fastwam_nocompile", "fasterwam", "lightwam", "lilawam", "lilawam_libero"]
res = {}
for p in sorted(Path(sys.argv[1]).glob("*.json")):
    if p.name.endswith(".config.yaml") or "smoke" in p.name:
        continue
    key, node = p.stem.rsplit("_", 1)
    res[(key, node)] = json.loads(p.read_text())
nodes = sorted({n for _, n in res})
md = []

md.append("### Per-query latency (batch 1, H100 80GB HBM3), ms\n")
md.append("Each cell: mean (p50 / p95) over the timed queries, after 20 warmup queries. Hz = 1000 / mean. "
          "Two jobs on two nodes measured every model independently.\n")
md.append("| Model | Prompt mode | " + " | ".join(f"{n} mean (p50 / p95)" for n in nodes) +
          " | Hz (" + nodes[0] + "; the other node where it is -) | Peak GPU GB | Identical to native |")
md.append("|---|---|" + "---|" * len(nodes) + "---|---|---|")
for key in ORDER:
    rows = {n: res.get((key, n)) for n in nodes}
    any_r = next((r for r in rows.values() if r), None)
    if not any_r:
        continue
    for mode in any_r["modes"]:
        cells = []
        for n in nodes:
            m = rows[n]["modes"].get(mode) if rows[n] else None
            cells.append(f"{m['mean_ms']:.1f} ({m['median_ms']:.1f} / {m['p95_ms']:.1f})" if m else "-")
        first = rows[nodes[0]] or any_r
        m0 = first["modes"].get(mode) or any_r["modes"][mode]
        peak = m0.get("peak_allocated_gb")
        ident = first.get("prompt_cached_identical_to_native", "") if mode == "prompt_cached" else ""
        md.append(f"| {any_r['model']} | {mode} | " + " | ".join(cells) +
                  f" | {1000 / m0['mean_ms']:.1f} | {'' if peak is None else f'{peak:.1f}'} | {ident} |")
md.append("")

md.append("### Configuration actually executed (read back from each run)\n")
md.append("| Model | Repo @ commit | Checkpoint | Weights dtype | Denoising steps | Inference path | Prompt handling (native) | Action chunk | Parameters (incl. text encoder, VAE) | Load s |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for key in ORDER:
    r = next((res[(key, n)] for n in nodes if (key, n) in res), None)
    if not r:
        continue
    repo = Path(r["repo"]).name
    if key == "lilawam":
        path, prompt = "RobotWinInference._predict_chunk (DINOv3 features + flow steps + B-spline smoothing)", \
            "no language input; precomputed task-condition vector"
        steps, dtype = r["num_inference_steps"], r["dtype"]
    elif key == "lilawam_libero":
        path, prompt = ("public _predict_chunk procedure on LIBERO inputs ([agentview | wrist] 512x256, 8-D state; "
                        "DINOv3 features + flow steps + B-spline smoothing)"), "no language input; task-condition vector"
        steps, dtype = r["num_inference_steps"], r["dtype"]
    else:
        path = r.get("action_infer_mode") or "infer_action"
        if key == "fastwam":
            path += " + torch.compile (EVALUATION.compile_action_infer=true, the repo default)"
        if key == "fastwam_nocompile":
            path += " (EVALUATION.compile_action_infer=false)"
        prompt = ("pre-encoded once per task (EVALUATION.preencode_prompt_context=true)" if r.get("preencode_prompt_context")
                  else "T5 prompt encoding inside every query")
        steps, dtype = r.get("num_inference_steps"), r.get("model_dtype")
    md.append(f"| {r['model']} | {repo} @ `{r['repo_commit'][:7]}` | `{Path(r['checkpoint']).name}` | {dtype} | {steps} | "
              f"{path} | {prompt} | {r.get('action_chunk_shape')} | {r['parameters'] / 1e9:.2f} B | {r['load_seconds']:.0f} |")
md.append("")
Path(sys.argv[2]).write_text("\n".join(md) + "\n")
print("\n".join(md))
