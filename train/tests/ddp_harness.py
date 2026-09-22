"""Two-rank CPU (gloo) runs of the real Trainer, for the review's DDP findings (#2, #3, #5)."""
import json
import os
import socket
import time
from pathlib import Path

import torch.multiprocessing as mp


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CountingStore:
    """LocalStore that records which rank downloaded."""

    def __init__(self, root, rank, log):
        from train.store import LocalStore
        self._inner, self.rank, self.log = LocalStore(root), rank, Path(log)

    def download_dir(self, prefix, local_dir):
        with open(self.log, "a") as f:
            f.write(f"{self.rank}\n")
        return self._inner.download_dir(prefix, local_dir)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _rank(rank, port, scenario, root, cfg_over):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))
    import torch
    import torch.distributed as dist

    # macOS only: torch 2.11's barrier() picks the MPS device, which gloo cannot serve. A CPU all_reduce
    # is an equivalent barrier. Kaggle (CUDA) never takes this path.
    dist.barrier = lambda *a, **k: dist.all_reduce(torch.zeros(1))

    from train import lora
    from train.config import load_config
    from train.data import SyntheticDataset
    from train.model import build_cfm, load_vocab
    from train.tests.conftest import TINY_PATH
    from train.trainer import Trainer

    root = Path(root)
    cfg = load_config(TINY_PATH)
    for k, v in cfg_over.items():
        cfg[k].update(v)
    vocab = load_vocab()
    torch.manual_seed(0)
    model = lora.attach(build_cfm(cfg, vocab), cfg)
    ds = SyntheticDataset(96, 3, vocab, min_frames=100, max_frames=100)
    if scenario == "guard":
        start = time.monotonic()
        # rank 0's clock runs 10,000x fast: only rank 0 would see the guard on its own
        clock = (lambda: (time.monotonic() - start) * 10_000) if rank == 0 else time.monotonic
    else:
        clock = time.monotonic
    on_micro = None
    if scenario == "poison":
        fired = []

        def on_micro(update, micro):               # one NaN batch on rank 1 only, once
            ds.poison = rank == 1 and update == 3 and micro == 0 and not fired
            if ds.poison:
                fired.append(True)
    store = CountingStore(root / "hub", rank, root / "downloads.txt")
    t = Trainer(cfg, model=model, train_set=ds, val_items=SyntheticDataset(4, 9, vocab).val_items(), store=store,
                work_dir=root / "work", base_sha="b" * 64, manifest_sha=ds.manifest_sha256(), clock=clock,
                on_micro=on_micro)
    reason = t.run()
    (root / f"rank{rank}.json").write_text(json.dumps({"reason": reason, "state": t.state}))


def run_two_ranks(scenario, root, cfg_over=None, timeout=90):
    ctx = mp.start_processes(_rank, args=(free_port(), scenario, str(root), cfg_over or {}), nprocs=2,
                             join=False, start_method="spawn")
    deadline = time.monotonic() + timeout
    while not ctx.join(timeout=1):
        if time.monotonic() > deadline:
            for p in ctx.processes:
                p.kill()
            raise TimeoutError(f"{scenario}: ranks hung (mismatched collectives)")
    return [json.loads((Path(root) / f"rank{r}.json").read_text()) for r in (0, 1)]
