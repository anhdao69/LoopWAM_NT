#!/usr/bin/env bash
# Node-local, verified copy of the frozen-VAE latent cache (NFS random memmap reads are ~30-40 ms/row).
set -euo pipefail
source "$(dirname "$0")/env.sh"
[ -f "$ELASTIC_LATENTS/STAGED.json" ] || python "$ELASTIC_ROOT/plans/elastic_evidence/campaign/stage_latents.py" \
    "$ELASTIC_NFS_LATENTS" "$ELASTIC_LATENTS"
cat "$ELASTIC_LATENTS/STAGED.json"
