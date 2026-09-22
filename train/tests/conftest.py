import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: builds the full-size model or spawns processes")
    for n in range(1, 13):
        config.addinivalue_line("markers", f"c{n}: FINETUNE_PLAN §8a check C{n}")


@pytest.fixture(scope="session")
def vendor():
    from train.patches import import_vendor

    return import_vendor()
