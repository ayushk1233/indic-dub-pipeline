"""
Where checkpoints go (FINETUNE_PLAN §7d). Every checkpoint uploads to its own prefix and LATEST is
written last, naming that prefix and its hashes, so an upload killed halfway leaves the previous
LATEST pointing at complete files (plan Decision 2). LocalStore is a folder with the same protocol,
used by the CPU checks; HfStore is a private Hugging Face repo.
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path

from train import checkpoint


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _put(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)

    def upload_dir(self, local_dir, prefix):
        files = sorted(p for p in Path(local_dir).iterdir() if p.is_file())
        files.sort(key=lambda p: p.name == checkpoint.SUMS)          # sums last
        for p in files:
            self._put(p, self.root / prefix / p.name)

    def write_text(self, name, text):
        tmp = self.root / (name + ".part")
        tmp.write_text(text)
        tmp.replace(self.root / name)

    def read_text(self, name):
        p = self.root / name
        return p.read_text() if p.exists() else None

    def download_dir(self, prefix, local_dir):
        local = Path(local_dir)
        shutil.rmtree(local, ignore_errors=True)
        shutil.copytree(self.root / prefix, local)
        return local

    def list_prefixes(self, parent):
        d = self.root / parent
        return sorted(f"{parent}/{p.name}" for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def delete_prefix(self, prefix):
        shutil.rmtree(self.root / prefix, ignore_errors=True)


class HfStore:
    """A private model repo. The token comes from the caller (Kaggle secrets), never from config."""

    def __init__(self, repo_id, token, api=None, download=None):
        from huggingface_hub import HfApi, snapshot_download

        self.repo_id, self.token = repo_id, token
        self.api = api or HfApi(token=token)
        self._download = download or snapshot_download

    def upload_dir(self, local_dir, prefix):
        self.api.upload_folder(folder_path=str(local_dir), path_in_repo=prefix, repo_id=self.repo_id,
                               repo_type="model", commit_message=f"checkpoint {prefix}")

    def write_text(self, name, text):
        self.api.upload_file(path_or_fileobj=text.encode(), path_in_repo=name, repo_id=self.repo_id,
                             repo_type="model", commit_message=f"update {name}")

    def read_text(self, name):
        if not self.api.file_exists(repo_id=self.repo_id, filename=name, repo_type="model"):
            return None
        root = self._download(repo_id=self.repo_id, allow_patterns=[name], token=self.token)
        return (Path(root) / name).read_text()

    def download_dir(self, prefix, local_dir):
        root = Path(self._download(repo_id=self.repo_id, allow_patterns=[f"{prefix}/*"], token=self.token))
        local = Path(local_dir)
        shutil.rmtree(local, ignore_errors=True)
        shutil.copytree(root / prefix, local)
        return local

    def list_prefixes(self, parent):
        files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="model")
        return sorted({"/".join(f.split("/")[:2]) for f in files if f.startswith(parent + "/")})

    def delete_prefix(self, prefix):
        self.api.delete_folder(path_in_repo=prefix, repo_id=self.repo_id, repo_type="model")


def read_latest(store):
    text = store.read_text("LATEST")
    return json.loads(text) if text else None


def publish_latest(store, latest):
    store.write_text("LATEST", json.dumps(latest, sort_keys=True))


def download_checkpoint(store, latest, local_dir) -> Path:
    local = store.download_dir(latest["prefix"], local_dir)
    if checkpoint.verify(local) != latest["hashes"]:
        raise checkpoint.CheckpointError(f"{latest['prefix']}: files do not match LATEST's hashes")
    return local


class Uploader:
    """Uploads in a background thread, in submission order; LATEST only after its files are up."""

    def __init__(self, store, keep=2, retries=3, backoff=5.0, sleep=time.sleep):
        self.store, self.keep = store, keep
        self.retries, self.backoff, self._sleep = retries, backoff, sleep
        self._q, self._errors = queue.Queue(), []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, local_dir, prefix, latest=None):
        self._q.put((Path(local_dir), prefix, latest))

    def _run(self):
        while True:
            job = self._q.get()
            try:
                if job is None:
                    return
                self._with_retries(job)
            except Exception as e:  # surfaced by wait()
                self._errors.append(e)
            finally:
                self._q.task_done()

    def _with_retries(self, job):
        local_dir, prefix, latest = job
        for attempt in range(self.retries):
            try:
                self.store.upload_dir(local_dir, prefix)
                if latest is not None:
                    publish_latest(self.store, latest)
                    for old in self.store.list_prefixes("ckpt")[:-self.keep]:
                        self.store.delete_prefix(old)
                return
            except Exception:
                if attempt == self.retries - 1:
                    raise
                self._sleep(self.backoff * 2 ** attempt)     # transient Hub errors (5xx, rate limit)

    def wait(self):
        self._q.join()
        if self._errors:
            raise self._errors.pop(0)

    def close(self):
        self._q.put(None)
        self._thread.join()
