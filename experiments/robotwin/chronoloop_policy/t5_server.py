"""One shared T5 encoder for all RoboTwin evaluation processes on a node.

python experiments/robotwin/chronoloop_policy/t5_server.py --address /path/t5.sock

Every policy process would otherwise hold its own 11 GB umt5-xxl. Clients (deploy_policy.TextEncoder with
`t5_server`) send a prompt and receive (context bf16 [128, 4096], mask bool [128]); the encoding is the policy's own
`TextEncoder` code, so it matches the training text cache bit for bit. One request is encoded at a time.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from multiprocessing.connection import Listener
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_policy import AUTHKEY, TextEncoder  # noqa: E402


def serve(conn, encoder, lock):
    with conn:
        while True:
            try:
                prompt = conn.recv()
            except EOFError:
                return
            with lock:
                context, mask = encoder(prompt)
            # numpy, not tensors: torch's pickler shares tensor storage through a per-process-tree resource sharer
            conn.send((context.view(torch.int16).numpy(), mask.numpy()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--address', required=True)
    a = p.parse_args()
    encoder = TextEncoder('/nonexistent', 'none', 'cuda')
    encoder._load_encoder()
    lock = threading.Lock()
    if os.path.exists(a.address):
        os.unlink(a.address)
    with Listener(a.address, family='AF_UNIX', authkey=AUTHKEY) as listener:
        print(f'T5 server ready at {a.address}', flush=True)
        while True:
            threading.Thread(target=serve, args=(listener.accept(), encoder, lock), daemon=True).start()


if __name__ == '__main__':
    main()
