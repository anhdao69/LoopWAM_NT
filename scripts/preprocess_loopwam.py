"""Create canonical donors from a local, pinned native Wan2.1 checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from fastwam.models.wan22.loopwam_init import (
    NativeSafetensors, build_wan_initialized_donors, sha256_file, validate_artifact,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--source-revision', required=True, help='Resolved Hugging Face source commit hash')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if len(args.source_revision) != 40 or any(c not in '0123456789abcdef' for c in args.source_revision.lower()):
        parser.error('--source-revision must be a full 40-character commit hash.')
    torch.set_num_threads(args.threads)
    config_path = args.source_dir / 'config.json'
    paths = sorted(args.source_dir.glob('diffusion_pytorch_model*.safetensors'))
    if not paths:
        parser.error('No native diffusion_pytorch_model*.safetensors found in source directory.')
    config = json.loads(config_path.read_text())
    print(f'Hashing {len(paths)} native source files...', flush=True)
    hashes = {path.name: sha256_file(path) for path in [config_path, *paths]}
    with NativeSafetensors(paths) as source:
        print(f'Converting {len(source)} tensors on CPU in FP32...', flush=True)
        artifact = build_wan_initialized_donors(source, config, source_revision=args.source_revision,
                                               source_hashes=hashes, seed=args.seed)
    validate_artifact(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save(artifact, temporary)
    temporary.replace(args.output)
    manifest = {**artifact['metadata'], 'artifact_sha256': sha256_file(args.output)}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved canonical 30-layer donor pair: {args.output}', flush=True)


if __name__ == '__main__':
    main()
