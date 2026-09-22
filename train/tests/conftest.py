import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: builds the full-size model or spawns processes")
    for n in range(1, 13):
        config.addinivalue_line("markers", f"c{n}: FINETUNE_PLAN §8a check C{n}")


@pytest.fixture(scope="session")
def vendor():
    from train.patches import import_vendor

    return import_vendor()


from pathlib import Path

TINY_PATH = Path(__file__).parent / "configs" / "tiny.yaml"


@pytest.fixture(scope="session")
def vocab(vendor):
    from train.model import load_vocab

    return load_vocab()


@pytest.fixture
def tiny_cfg():
    from train.config import load_config

    return load_config(TINY_PATH)


@pytest.fixture
def tiny_model(tiny_cfg, vocab):
    import torch

    from train.model import build_cfm

    torch.manual_seed(0)
    return build_cfm(tiny_cfg, vocab)
