"""
Build the fixture media once per session, before any test needs it.

See tests/fixture_media.py for what it is and why each property of it
matters. It lives in the system temp directory, not the repo: the only
recordings that belong to this project are english.mov and hindi.mov.
"""

import sys
from pathlib import Path

import pytest

# conftest is imported before pytest puts the test directory on sys.path, so
# the sibling module is not importable yet without this.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_media import build  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def fixture_media():
    try:
        return build()
    except RuntimeError as exc:
        pytest.skip(str(exc), allow_module_level=True)
