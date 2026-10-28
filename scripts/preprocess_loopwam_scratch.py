"""Create LoopWAM-S donors without video pretraining from the canonical donor artifact.

Wan-derived tensors get Wan2.1's from-scratch initialization; the new action input/output and proprio
tensors are copied unchanged, so this differs from the canonical artifact only in pretraining.
"""
import argparse
import json
from pathlib import Path

import torch

from fastwam.models.wan22.loopwam_init import (
    build_scratch_donors, load_init_artifact, sha256_file, validate_artifact,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, required=True, help='Canonical Wan-initialized donor artifact')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    canonical_sha = sha256_file(args.canonical)
    artifact = build_scratch_donors(load_init_artifact(args.canonical), canonical_sha256=canonical_sha, seed=args.seed)
    validate_artifact(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save(artifact, temporary)
    temporary.replace(args.output)
    manifest = {**artifact['metadata'], 'artifact_sha256': sha256_file(args.output)}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved scratch 30-layer donor pair: {args.output}', flush=True)


if __name__ == '__main__':
    main()
