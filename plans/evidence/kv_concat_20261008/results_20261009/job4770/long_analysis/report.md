# LoopWAM 4/1 all-loop KV measured results

Decision: **inconclusive**.

Preregistered: ≥89% supports access; ≤84% supports action depth; otherwise inconclusive and expand to 500 initial-state rollouts. One training seed42.

| Model | KV | Video/action depth | Block calls/chunk | Parameters | Seed42/43/44 | Pooled SR (Wilson95) |
|---|---|---|---|---|---|---|
| 4/1 | aligned | 30/12 | 150 | 584,536,135 | 81/88/84 | 84.33% (79.79%–88.01%) |
| 4/1 | concat | 30/12 | 150 | 584,536,135 | 83/86/86 | 85.00% (80.52%–88.60%) |
| 4/1 | mix | 30/12 | 150 | 584,536,423 | 90/89/87 | 88.67% (84.58%–91.78%) |
| 4/4 original | aligned | 30/30 | 330 | 584,536,135 | 96/88/91 | 91.67% (87.99%–94.29%) |
| 4/4 repeat | aligned | 30/30 | 330 | 584,536,135 | 91/85/92 | 89.33% (85.33%–92.34%) |

## Paired exact McNemar

| Candidate | Seed/grid | Candidate-only wins | Baseline-only wins | Exact p |
|---|---|---|---|---|
| concat | 42 | 16 | 14 | 0.855536 |
| concat | 43 | 9 | 11 | 0.823803 |
| concat | 44 | 10 | 8 | 0.814529 |
| concat | pooled | 35 | 33 | 0.903597 |
| mix | 42 | 16 | 7 | 0.0931396 |
| mix | 43 | 10 | 9 | 1 |
| mix | 44 | 12 | 9 | 0.663624 |
| mix | pooled | 38 | 25 | 0.129918 |

## Training and diagnostics

### concat

Training: 7.57 h; 15.14 GPU-hours.

First/last100 losses and per-epoch losses: `{"first100": {"loss_video": 0.27746415346860887, "loss_action": 0.5437650872766971, "grad_norm": 2.6350901877880095}, "last100": {"loss_video": 0.06370190220574538, "loss_action": 0.019360594234118858, "grad_norm": 0.14218908362090588}, "epochs": {"1": {"loss_video": 0.15312682214139522, "loss_action": 0.18839967914353845, "grad_norm": 1.1701237791160057}, "2": {"loss_video": 0.09606587268155196, "loss_action": 0.09091401886323403, "grad_norm": 0.3695050420637788}, "3": {"loss_video": 0.08466248674296785, "loss_action": 0.0786178058163188, "grad_norm": 0.2857117445510009}, "4": {"loss_video": 0.07857455870885958, "loss_action": 0.06692518916452068, "grad_norm": 0.2449311815048086}, "5": {"loss_video": 0.07439101116410617, "loss_action": 0.055322014905255415, "grad_norm": 0.21605577725788644}, "6": {"loss_video": 0.07078944038385632, "loss_action": 0.0443220147815929, "grad_norm": 0.19622275350422694}, "7": {"loss_video": 0.06835032130452409, "loss_action": 0.035008224850070886, "grad_norm": 0.17162628860309206}, "8": {"loss_video": 0.06605058101744488, "loss_action": 0.027849833661283574, "grad_norm": 0.15474179089069368}, "9": {"loss_video": 0.06454218661990659, "loss_action": 0.022766889560667945, "grad_norm": 0.14419966004018125}, "10": {"loss_video": 0.06370419969161352, "loss_action": 0.020078269265934656, "grad_norm": 0.13867863416671752}}}`

![concat diagnostics](concat_weights.png)

### mix

Training: 7.31 h; 14.61 GPU-hours.

First/last100 losses and per-epoch losses: `{"first100": {"loss_video": 0.27482486933469774, "loss_action": 0.5423175691068173, "grad_norm": 2.389037797451019}, "last100": {"loss_video": 0.0639253247777621, "loss_action": 0.018683917845288913, "grad_norm": 0.13658317819237709}, "epochs": {"1": {"loss_video": 0.15283727770906758, "loss_action": 0.18763059048131964, "grad_norm": 1.1286746241717502}, "2": {"loss_video": 0.09672559297975454, "loss_action": 0.0905967398861359, "grad_norm": 0.3738519770318064}, "3": {"loss_video": 0.08515354107508714, "loss_action": 0.07760325845973245, "grad_norm": 0.2891715291450764}, "4": {"loss_video": 0.07907022820635773, "loss_action": 0.06516086775681068, "grad_norm": 0.25255398964059766}, "5": {"loss_video": 0.07486401541822259, "loss_action": 0.05289540165628509, "grad_norm": 0.22572526047969688}, "6": {"loss_video": 0.07122444091000776, "loss_action": 0.042126181476760186, "grad_norm": 0.19830354429524522}, "7": {"loss_video": 0.06874795130100743, "loss_action": 0.03333904569347699, "grad_norm": 0.1721750241106954}, "8": {"loss_video": 0.06634143638884883, "loss_action": 0.02657601108838772, "grad_norm": 0.15149744540452958}, "9": {"loss_video": 0.06480396487246984, "loss_action": 0.021930537725659623, "grad_norm": 0.14024008527911944}, "10": {"loss_video": 0.06394185659015317, "loss_action": 0.019360386969029218, "grad_norm": 0.13521851808860383}}}`

![mix diagnostics](mix_weights.png)

## Per-task success out of30

| Task | Aligned | Concat | Mix |
|---|---|---|---|
| 0 | 20 | 26 | 23 |
| 1 | 23 | 29 | 30 |
| 2 | 25 | 25 | 25 |
| 3 | 28 | 27 | 30 |
| 4 | 22 | 24 | 26 |
| 5 | 30 | 30 | 30 |
| 6 | 24 | 20 | 26 |
| 7 | 26 | 26 | 25 |
| 8 | 29 | 26 | 25 |
| 9 | 26 | 22 | 26 |

## Latency

| Model | Mode | Total ms | Video prefill ms | Action denoising ms |
|---|---|---|---|---|
| concat | eager | 139.843 | 21.938 | 102.105 |
| mix | eager | 129.967 | 19.128 | 95.894 |
| aligned41 | eager | 118.428 | 18.569 | 85.613 |
| original44 | eager | 237.776 | 18.370 | 205.187 |

## Unpaired comparison against original4/4

- concat: {"z": -2.5434234203957833, "p_two_sided": 0.010977211637449113, "assumption": "Unpaired episode-level approximation; ignores task/state clustering"}
- mix: {"z": -1.2339384472520785, "p_two_sided": 0.21722584151358715, "assumption": "Unpaired episode-level approximation; ignores task/state clustering"}

## Statistical limitations

- One training seed42; training-seed variance not measured.
- Wilson and two-proportion calculations treat episodes as independent; shared tasks/states create clustering.
- Threshold rule is preregistered heuristic, not proof of mechanism.

## Supplemental evaluation

[Expanded 500-episode comparison](../expanded_analysis/report.md). The original decision is retained.
