# GPU pre-flight — the CPU-capable part, run locally (FINETUNE_PLAN §8b)

commit `023012e`, 2026-09-22T09:06:56Z

| check | status | detail |
|---|---|---|
| G2 | pass | release file `ai4bharat/IndicF5@ba85abed` (sha256 `ba7f3671…865c`) loads strictly after stripping `ema_model._orig_mod.`: 337,096,804 parameters, 0 layout problems against §0, nothing on meta. The same file's 83 Vocos tensors load strictly into `charactr/vocos-mel-24khz`'s architecture (13,531,650 parameters). |
| G4 | pass | the training mel (vendored `CustomDataset`) equals the inference mel (`CFM.mel_spec`) for the same file: max abs diff 0.0. The clip's RMS is 0.0645, so `infer_batch_process` would scale a reference like it by 1.55× while training sees it unscaled; §2c's −20 LUFS normalisation in data prep (step 2) closes that gap. |
| G3 | not run here | needs the shipping stack on CUDA (Kaggle notebook). Locally, `train.generate` is proven equal to the vendored `infer_batch_process` (`train/tests/test_generate.py`). |

## Full-size generation on CPU (not a GPU check)

The real model plus release vocoder, fp32, 32 NFE steps: 3.0 s of audio in 36 s. Output RMS 0.078, peak 0.54.
It uses the G3 request: reference = the first Route A `val_en` clip, text = "the weather is lovely today so let
us go for a walk" in Devanagari, fix_duration = reference + 3 s.

## Pre-flight dataset (GLOBE, "India and South Asia" accent label, CC0)

| split | clips | speakers | hours |
|---|---|---|---|
| train | 2,486 | 191 | 3.0 |
| val_en | 150 | 8 (held out) | 0.18 |

Gates (both transcript forms): 2,630 kept, 6 rejected on characters per second, 0 on vocabulary, 0 on script.
It is written twice over identical clips: `route_a/` (Devanagari English) and `route_b/` (Latin English), with
`overfit16/` in each. 274 MB of FLAC.
