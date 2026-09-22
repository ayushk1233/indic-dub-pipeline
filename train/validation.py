"""
Fixed-noise validation loss (FINETUNE_PLAN §10a): for each clip the noise, time and span are drawn
once from a fixed seed and reused at every evaluation, so checkpoints compare on the same draws.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from train.patches import import_vendor, strict_list_str_to_idx


def flow_loss(cfm, mel, text_ids, lens, x0, time, frac, start_rand):
    u = import_vendor().utils
    mask = u.lens_to_mask(lens, length=mel.shape[1])
    lengths = (frac * lens).long()
    start = ((lens - lengths) * start_rand).long().clamp(min=0)
    span = u.mask_from_start_end_indices(lens, start, start + lengths) & mask
    t = time[:, None, None]
    phi = (1 - t) * x0 + t * mel
    cond = torch.where(span[..., None], torch.zeros_like(mel), mel)
    pred = cfm.transformer(x=phi, cond=cond, text=text_ids, time=time, drop_audio_cond=False, drop_text=False)
    return F.mse_loss(pred, mel - x0, reduction="none")[span].mean()


class FixedNoiseValidator:
    def __init__(self, items, vocab, seed):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for it in items:
            mel = it["mel"].T.unsqueeze(0).float()               # [1, n, d]
            frac = torch.empty(1).uniform_(0.7, 1.0, generator=g)
            start_rand = torch.rand(1, generator=g)
            x0 = torch.randn(mel.shape, generator=g)
            time = torch.rand(1, generator=g)
            self.items.append(dict(group=it["group"], mel=mel, lens=torch.tensor([mel.shape[1]]),
                                   text=strict_list_str_to_idx([it["text"]], vocab),
                                   x0=x0, time=time, frac=frac, start_rand=start_rand))

    @torch.no_grad()
    def evaluate(self, cfm) -> dict[str, float]:
        was_training = cfm.training
        cfm.eval()
        device = next(cfm.parameters()).device
        per_group = defaultdict(list)
        try:
            for it in self.items:
                loss = flow_loss(cfm, *(it[k].to(device) for k in ("mel", "text", "lens", "x0", "time", "frac", "start_rand")))
                per_group[it["group"]].append(float(loss))
        finally:
            cfm.train(was_training)
        out = {g: sum(v) / len(v) for g, v in per_group.items()}
        out["all"] = sum(sum(v) for v in per_group.values()) / sum(len(v) for v in per_group.values())
        return out
