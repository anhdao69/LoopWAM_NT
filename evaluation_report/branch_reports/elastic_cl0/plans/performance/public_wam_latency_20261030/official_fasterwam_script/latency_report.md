# Inference Latency

- GPU: NVIDIA H100 80GB HBM3
- Setting: BF16, batch 1, 224 × 448 input, action horizon 32, 10 denoising steps
- Protocol: 5 warm-up iterations followed by 10 GPU-synchronized measurements

Values are mean ± population standard deviation in milliseconds.

| Model | VAE Encode | Visual Branch | Action Branch | Model Infer |
|---|---:|---:|---:|---:|
| Joint-WAM | 4.838 ± 0.068 | N/A | N/A | 427.924 ± 1.439 |
| Fast-WAM | 4.697 ± 0.043 | 20.846 ± 0.405 | 214.925 ± 0.418 | 244.709 ± 0.786 |
| Faster-WAM | 4.795 ± 0.049 | 22.124 ± 0.123 | 147.169 ± 0.759 | 178.542 ± 0.774 |

- **VAE Encode:** encode the current observation into the first-frame latent.
- **Visual Branch:** video pre-DiT, attention masks, visual forward, and reusable KV-cache construction.
- **Action Branch:** action-latent initialization, all denoising steps, and final action transfer to CPU.
- **Model Infer:** GPU-synchronized timing of the native `infer_action*` entry point.
- Joint-WAM denoises video and action jointly, so its Visual and Action branches are reported as N/A.
