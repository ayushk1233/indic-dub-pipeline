import json

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from train.sampler import OversizedItemError, ResumableBatchSampler

LENS = [40 + (i * 37) % 120 for i in range(50)]


def test_packing_respects_frames_and_max_samples():
    s = ResumableBatchSampler(LENS, 400, 4, seed=1)
    assert sorted(i for b in s.batches for i in b) == list(range(50))
    for b in s.batches:
        assert len(b) <= 4 and sum(LENS[i] for i in b) <= 400


def test_oversized_item_raises():
    with pytest.raises(OversizedItemError, match="index 2"):
        ResumableBatchSampler([10, 20, 500], 400, 4, seed=1)


def test_epochs_are_deterministic_and_differ():
    a, b = ResumableBatchSampler(LENS, 400, 4, seed=1), ResumableBatchSampler(LENS, 400, 4, seed=1)
    assert a.epoch_batches(0) == b.epoch_batches(0) != a.epoch_batches(1)


def _rank_main(rank, world, init, out):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    s = ResumableBatchSampler(LENS, 400, 4, seed=7, world_size=dist.get_world_size(), rank=dist.get_rank())
    mine = {e: s.epoch_batches(e) for e in (0, 1)}
    gathered = [None] * world
    dist.all_gather_object(gathered, mine)
    fresh = ResumableBatchSampler(LENS, 400, 4, seed=7, world_size=world, rank=rank)   # "resumed" process
    resumed_next = fresh.epoch_batches(1)[3]
    if rank == 0:
        with open(out, "w") as f:
            json.dump({"gathered": gathered, "resumed_next": [resumed_next, mine[1][3]]}, f)
    dist.destroy_process_group()


@pytest.mark.c9
@pytest.mark.slow
def test_two_gloo_ranks_see_disjoint_equal_batches_and_resume_reproduces(tmp_path):
    out = tmp_path / "out.json"
    mp.spawn(_rank_main, args=(2, f"file://{tmp_path}/pg", str(out)), nprocs=2, join=True)
    got = json.loads(out.read_text())
    for e in ("0", "1"):
        r0, r1 = (set(map(tuple, got["gathered"][r][e])) for r in (0, 1))
        assert r0.isdisjoint(r1)
        assert len(got["gathered"][0][e]) == len(got["gathered"][1][e])
    assert got["resumed_next"][0] == got["resumed_next"][1]
