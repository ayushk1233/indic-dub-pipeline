"""EMA of the trainable parameters only (FINETUNE_PLAN §5, §7b): the frozen base never has a shadow."""
from __future__ import annotations

from contextlib import contextmanager

import torch


class TrainableEMA:
    def __init__(self, model, decay: float, start_after: int):
        self.decay, self.start_after = decay, start_after
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model, step: int) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                if step < self.start_after:
                    self.shadow[n].copy_(p.detach().float())
                else:
                    self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    def shadow_state(self) -> dict[str, torch.Tensor]:
        return {n: t.clone() for n, t in self.shadow.items()}

    def load_shadow(self, sd) -> None:
        if set(sd) != set(self.shadow):
            raise KeyError("EMA names differ from the model's trainable parameters")
        for n in self.shadow:
            self.shadow[n].copy_(sd[n])

    @contextmanager
    def swapped(self, model):
        params = {n: p for n, p in model.named_parameters() if n in self.shadow}
        saved = {n: p.detach().clone() for n, p in params.items()}
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(self.shadow[n])
        try:
            yield model
        finally:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(saved[n])
