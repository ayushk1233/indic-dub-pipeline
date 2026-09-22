"""
G3 / G13 reference output from the SHIPPING stack (FINETUNE_PLAN §8b, step-4 plan Decision 3): the loader the
dubbing notebook uses (colab.runtime.load_indicf5) and the installed f5_tts infer_batch_process, fp32 on CUDA.
Runs in Kaggle's own Python after `pip install git+https://github.com/ai4bharat/IndicF5.git` and
`-r requirements-gpu.txt`, as a subprocess so no kernel restart is needed. Writes mel [d, n] and the inputs,
so the train venv (train.preflight_runners g3/g13) regenerates from exactly the same request.

  python train/kaggle/official_mel.py --ref clip.flac --ref-text "..." --gen-text "..." --seed 1234 \
      --fix-duration 9.5 --out official_g3.npz [--weights merged_release.safetensors]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # repo root, for colab.runtime


def main(argv=None):
    p = argparse.ArgumentParser()
    for flag in ("--ref", "--ref-text", "--gen-text", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fix-duration", type=float, required=True)
    p.add_argument("--weights")
    a = p.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file

    from colab.runtime import load_indicf5

    model = load_indicf5().to("cuda").float()
    ema = getattr(model.ema_model, "_orig_mod", model.ema_model)
    if a.weights:
        prefix = "ema_model._orig_mod."
        sd = {k[len(prefix):]: t for k, t in load_file(a.weights).items() if k.startswith(prefix)}
        ema.load_state_dict(sd, strict=True)
    from f5_tts.infer.utils_infer import infer_batch_process

    audio, sr = sf.read(a.ref, dtype="float32", always_2d=True)
    torch.manual_seed(a.seed)
    wave, _, spec = infer_batch_process((torch.from_numpy(audio.T), sr), a.ref_text, [a.gen_text], ema,
                                        model.vocoder, mel_spec_type="vocos", fix_duration=a.fix_duration,
                                        device="cuda")
    np.savez(a.out, mel=spec, wave=wave, ref_path=str(Path(a.ref).resolve()), ref_text=a.ref_text,
             gen_text=a.gen_text, seed=a.seed, fix_duration=a.fix_duration)
    print(f"wrote {a.out}: mel {spec.shape}, {len(wave) / 24000:.2f} s")


if __name__ == "__main__":
    main()
