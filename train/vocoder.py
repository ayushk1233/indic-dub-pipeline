"""Vocos from the IndicF5 release file: charactr/vocos-mel-24khz's published hyperparameters, weights from the release."""
from __future__ import annotations

from safetensors.torch import load_file

from train.model import BaseWeightMismatch, split_release


def build_vocos():
    from vocos import Vocos
    from vocos.feature_extractors import MelSpectrogramFeatures
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone

    return Vocos(feature_extractor=MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256,
                                                          n_mels=100, padding="center"),
                 backbone=VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8),
                 head=ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="center"))


def load_vocoder(path):
    _, sd = split_release(load_file(str(path)))
    if not sd:
        raise BaseWeightMismatch(f"{path}: no vocoder tensors in this file")
    v = build_vocos()
    v.load_state_dict(sd, strict=True)
    return v.eval()
