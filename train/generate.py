"""
One generation, exactly as the vendored infer_batch_process does it for a single text batch
(FINETUNE_PLAN §0: fix_duration sets TOTAL frames, reference included). Used by G3, G6 and G13.
"""
from __future__ import annotations

import torch

from train.patches import import_vendor

SR, HOP, TARGET_RMS = 24000, 256, 0.1


def generate(cfm, ref_audio, ref_text, gen_text, *, seed, fix_duration=None, vocoder=None,
             nfe=32, cfg_strength=2.0, sway=-1.0):
    audio = ref_audio.mean(dim=0, keepdim=True) if ref_audio.shape[0] > 1 else ref_audio
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    text = import_vendor().utils.convert_char_to_pinyin([ref_text + gen_text])
    ref_len = audio.shape[-1] // HOP
    if fix_duration is not None:
        duration = int(fix_duration * SR / HOP)
    else:
        duration = ref_len + int(ref_len / len(ref_text.encode("utf-8")) * len(gen_text.encode("utf-8")))
    device = next(cfm.parameters()).device
    torch.manual_seed(seed)
    with torch.inference_mode():
        out, _ = cfm.sample(cond=audio.to(device), text=text, duration=duration, steps=nfe,
                            cfg_strength=cfg_strength, sway_sampling_coef=sway)
        gen = out.to(torch.float32)[:, ref_len:, :]
        wave = None
        if vocoder is not None:
            w = vocoder.decode(gen.permute(0, 2, 1))
            if rms < TARGET_RMS:
                w = w * rms / TARGET_RMS
            wave = w.squeeze().cpu().numpy()
    return {"mel": gen[0].permute(1, 0).cpu(), "wave": wave}
