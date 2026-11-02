#!/usr/bin/env bash
# CPU-only asset preparation of the CL-0 campaign (any node): CL-0 re-wrap as the loopwam-s-v1 teacher, then
# the hashed common_v1 initialization built from it. Inputs were fetched with pinned size + LFS sha256.
set -euo pipefail
source "$(dirname "$0")/env.sh"
A=$CL0_WORK/assets
[ -f "$ELASTIC_TEACHER" ] || python "$ELASTIC_ROOT/scripts/convert_chronoloop_parent.py" \
    --chronoloop "$A/CL-0_epoch_10/policy.pt" --template "$A/BASE10_v0-video4-action4/policy.pt" --out "$ELASTIC_TEACHER"
[ -f "$ELASTIC_INIT" ] || python "$ELASTIC_ROOT/scripts/elastic_init.py" common_v1 \
    --teacher "$ELASTIC_TEACHER" --vae-path "$ELASTIC_VAE" --out "$ELASTIC_INIT"
echo SETUP_ASSETS_DONE
