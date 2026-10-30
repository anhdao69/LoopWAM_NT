"""Exact inference speedups shared by every model family of the latency benchmark.

* `bf16_linear(module)`: stores every FP32 nn.Linear weight/bias in BF16. Under BF16 autocast, F.linear casts FP32
  weights to BF16 on every call (the autocast cache only covers leaf tensors that require grad, so frozen inference
  weights are recast per call, i.e. once per loop iteration and denoising step). Pre-casting performs the same
  round-to-nearest-even conversion once, so the matmul inputs are identical. Norms, modulation tables and anything
  else that runs in FP32 are left untouched. Exactness is still verified per model by the benchmark.
* `WholeQueryGraph`: one CUDA graph for VAE encode + prefill + denoising of a batch-one query, fed through static
  input buffers. The eager per-op launch cost of the VAE is removed as well as that of the policy.
"""
import torch


def bf16_linear(*modules):
    converted, seen = 0, set()
    for module in modules:
        for m in module.modules():
            if id(m) in seen:
                continue
            seen.add(id(m))
            if isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.float32:
                m.to(torch.bfloat16)
                converted += 1
    return converted


class WholeQueryGraph:
    """`fn(**static_inputs) -> tensor` captured once per input signature and replayed with copied-in inputs."""

    def __init__(self, fn, warmup=3):
        self.fn, self.warmup, self.graphs = fn, warmup, {}

    @torch.no_grad()
    def __call__(self, **inputs):
        key = tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(inputs.items()))
        if key not in self.graphs:
            static = {k: v.to("cuda", copy=True) for k, v in inputs.items()}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(self.warmup):
                    self.fn(**static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.fn(**static)
            self.graphs[key] = (graph, static, out)
        graph, static, out = self.graphs[key]
        for k, v in inputs.items():
            static[k].copy_(v)
        graph.replay()
        return out
