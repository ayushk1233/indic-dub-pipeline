"""
Where to write, on whichever GPU host this is running on.

Every module here hardcoded /content, which exists on Colab and nowhere else.
Kaggle gives 30 GPU hours a week against Colab's unpredictable allowance, so
the host is not a constant and the paths cannot be either.

Resolution order, most explicit first:

  INDIC_DUB_WORKSPACE   set it to put outputs anywhere
  /kaggle/working       Kaggle's persisted directory, the only one whose
                        contents survive the session and can be downloaded
  /content              Colab
  the current directory otherwise, which covers a plain GPU box

Nothing here creates a directory. Callers do that for the subdirectory they
actually want, so an import never has a side effect on the filesystem.
"""

import os
from pathlib import Path

ENV_VAR = "INDIC_DUB_WORKSPACE"
CANDIDATES = ("/kaggle/working", "/content")


def root():
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)

    for candidate in CANDIDATES:
        if Path(candidate).is_dir():
            return Path(candidate)

    return Path.cwd()


def out(name):
    """A directory for generated audio, under the workspace root."""
    return root() / name


def report(name):
    """A report file, under the workspace root."""
    return root() / name
