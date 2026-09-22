"""
Length-sorted dynamic batches (as the vendored DynamicBatchSampler) with three changes (§7b): an
item longer than the frame budget raises instead of vanishing, the per-epoch order is a pure
function of (seed, epoch) so resume can rebuild it, and batches are sharded per DDP rank with equal
counts so no rank waits on a collective the others never reach.
"""
from __future__ import annotations

import random


class OversizedItemError(ValueError):
    pass


class ResumableBatchSampler:
    def __init__(self, frame_lens, frames_threshold, max_samples, seed, world_size=1, rank=0):
        too_long = [i for i, n in enumerate(frame_lens) if n > frames_threshold]
        if too_long:
            raise OversizedItemError(f"index {too_long[0]} ({frame_lens[too_long[0]]:.0f} frames) exceeds "
                                     f"{frames_threshold} frames per GPU; {len(too_long)} such items")
        self.seed, self.world_size, self.rank = seed, world_size, rank
        self.batches, batch, frames = [], [], 0
        for i in sorted(range(len(frame_lens)), key=lambda i: (frame_lens[i], i)):
            if batch and (frames + frame_lens[i] > frames_threshold or len(batch) >= max_samples):
                self.batches.append(batch)
                batch, frames = [], 0
            batch.append(i)
            frames += frame_lens[i]
        if batch:
            self.batches.append(batch)

    def epoch_batches(self, epoch):
        order = list(range(len(self.batches)))
        random.Random(self.seed * 100_003 + epoch).shuffle(order)
        usable = len(order) - len(order) % self.world_size
        return [self.batches[i] for i in order[self.rank:usable:self.world_size]]
