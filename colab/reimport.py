"""
Pick up a `git pull` in a notebook kernel that has already imported this repo.

`importlib.reload(module)` reloads exactly one module. It does not touch that
module's imports, and this repo's modules are layered — `indicf5_xlit_probe`
imports `transcribe_outputs` and `ASR_DECODE` from `indicf5_check`, which
imports from `indicf5_english`. Reloading the top of that stack rebinds nothing
underneath it.

The failure is quiet and expensive. Measured 2026-09-18: a decode parameter was
fixed, committed, pulled, and `importlib.reload` run on the probe. The probe was
new; `indicf5_check` was four commits old, still held the wrong parameter name,
and still put the exception where the transcript belongs. Three separate runs
reported no content, and the diagnostic that finally caught it printed the name
of a parameter that had not been in the working tree for an hour.

    from colab.reimport import fresh
    probe = fresh("colab.indicf5_xlit_probe")

Restarting the runtime does the same thing and costs the loaded models.
"""

import importlib
import sys

PACKAGES = ("colab", "src")


def purge():
    """Drop every module of this repo from the import cache. Returns the names."""
    dropped = [name for name in list(sys.modules)
               if name in PACKAGES or name.startswith(tuple(p + "." for p in PACKAGES))]
    for name in dropped:
        del sys.modules[name]
    return sorted(dropped)


def fresh(module_name, verbose=True):
    """Re-import `module_name` and everything of ours beneath it, from disk."""
    dropped = purge()
    module = importlib.import_module(module_name)
    if verbose:
        print(f"re-imported {len(dropped)} module(s) from disk")
    return module
