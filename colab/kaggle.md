# Running this on Kaggle

Kaggle gives 30 GPU hours a week on a fixed schedule, against Colab's
unpredictable allowance. Two things differ and both bite immediately.

## The Hugging Face token

IndicF5 is a gated repository. Colab's secrets panel is wired into
`huggingface_hub` — a secret named `HF_TOKEN` with notebook access enabled is
picked up with no code. **Kaggle's is not.** Adding `HF_TOKEN` under Add-ons →
Secrets stores it and nothing reads it, so every download returns

    401 Client Error ... Access to model ai4bharat/IndicF5 is restricted.

which reads like a permissions problem and is not one. Read it out explicitly:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

The token must also belong to an account that has accepted the terms at
https://huggingface.co/ai4bharat/IndicF5 . A valid token on an account without
access returns the same 401, so if the login succeeds and the download still
fails, check the model page rather than the token.

Internet access is off by default in Kaggle notebooks. Turn it on under
Settings → Internet, or pip and every download fail before the token matters.

## Paths

`/content` is Colab's. On Kaggle, `/kaggle/working` is the only directory whose
contents persist for the session and can be downloaded afterwards, and it holds
20 GB. `colab/workspace.py` resolves this, preferring `INDIC_DUB_WORKSPACE` if
set, then `/kaggle/working`, then `/content`, then the working directory. No
module should name a host directory directly.

Clone into `/kaggle/working`, not `/kaggle/input`, which is read only.

## Persistence, which is not what it looks like

`/kaggle/working` survives a kernel restart but **not a container replacement**.
Close the tab, hit the idle timeout, or let the session be recycled, and the
next container restores that directory only from the last *saved version's*
output. Everything from an unsaved session is gone — generated audio, reports,
and the clone itself.

The symptom is a directory holding nothing but `.virtual_documents` after a run
that clearly wrote files. Nothing failed; the container is simply a different
one.

So: **Save Version** before leaving, or treat every session as disposable and
re-clone at the top. The second is cheaper here. The clone is seconds, the pip
install is a couple of minutes, and the model weights come from the Hugging
Face cache rather than `/kaggle/working`. Only generated audio is worth saving,
and only until it has been listened to.

To carry a run's output out of a session without saving a version:

```python
import shutil
from colab import workspace

shutil.make_archive("/kaggle/working/run", "zip", workspace.root())
```

then download it from the Output panel, which needs the notebook saved anyway —
so in practice, listen inside the session.

## What does not change

The install order does. IndicF5 first, this repo's pins second, because IndicF5
drags numpy back to 1.x and breaks transformers silently — see the comment in
`colab/requirements.txt`. Restart the session after installing, before importing
anything.
