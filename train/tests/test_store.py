import json

import pytest
import torch

from train import checkpoint
from train.checkpoint import CheckpointError
from train.store import HfStore, LocalStore, Uploader, download_checkpoint, publish_latest, read_latest

STATE = {"update": 1, "config_hash": "c", "manifest_sha256": "m", "base_sha256": "b"}


def _ckpt(path, value, update):
    hashes = checkpoint.save_checkpoint(path, trainable={"w": torch.full((2,), value)}, ema={"w": torch.zeros(2)},
                                        optim_state=None, state=dict(STATE, update=update))
    prefix = f"ckpt/update_{update:07d}"
    return prefix, {"prefix": prefix, "update": update, "hashes": hashes}


@pytest.mark.c7
def test_round_trip_hashes_match(tmp_path):
    store = LocalStore(tmp_path / "hub")
    prefix, latest = _ckpt(tmp_path / "last", 1.0, 10)
    up = Uploader(store); up.start(); up.submit(tmp_path / "last", prefix, latest); up.wait(); up.close()
    assert read_latest(store) == latest
    got = download_checkpoint(store, read_latest(store), tmp_path / "dl")
    assert checkpoint.verify(got) == latest["hashes"]


class FlakyStore(LocalStore):
    """Dies after copying `survive` files of the next upload: a session killed mid-upload."""

    def __init__(self, root, survive):
        super().__init__(root)
        self.survive = survive

    def _put(self, src, dst):
        if self.survive == 0:
            raise OSError("session killed")
        self.survive -= 1
        super()._put(src, dst)


@pytest.mark.c7
def test_kill_during_upload_keeps_the_previous_latest(tmp_path):
    store = FlakyStore(tmp_path / "hub", survive=10**6)
    p1, l1 = _ckpt(tmp_path / "last", 1.0, 10)
    up = Uploader(store); up.start(); up.submit(tmp_path / "last", p1, l1); up.wait()
    p2, l2 = _ckpt(tmp_path / "last", 2.0, 20)
    store.survive = 2
    up.submit(tmp_path / "last", p2, l2)
    with pytest.raises(OSError, match="killed"):
        up.wait()
    up.close()
    assert read_latest(store) == l1
    got = download_checkpoint(store, read_latest(store), tmp_path / "dl")
    assert torch.equal(checkpoint.load_checkpoint(got, config_hash="c", manifest_sha="m", base_sha="b")["trainable"]["w"],
                       torch.ones(2))


def test_download_mismatching_latest_hashes_refuses(tmp_path):
    store = LocalStore(tmp_path / "hub")
    prefix, latest = _ckpt(tmp_path / "last", 1.0, 10)
    store.upload_dir(tmp_path / "last", prefix)
    publish_latest(store, dict(latest, hashes=dict(latest["hashes"], **{"state.json": "0" * 64})))
    with pytest.raises(CheckpointError, match="LATEST"):
        download_checkpoint(store, read_latest(store), tmp_path / "dl")


def test_uploader_keeps_two_newest_checkpoints_and_all_epochs(tmp_path):
    store = LocalStore(tmp_path / "hub")
    up = Uploader(store, keep=2); up.start()
    for u in (10, 20, 30):
        prefix, latest = _ckpt(tmp_path / "last", float(u), u)
        up.submit(tmp_path / "last", prefix, latest); up.wait()
    epoch, _ = _ckpt(tmp_path / "epoch_0", 0.0, 5)
    up.submit(tmp_path / "epoch_0", "epoch_0", None); up.wait(); up.close()
    assert store.list_prefixes("ckpt") == ["ckpt/update_0000020", "ckpt/update_0000030"]
    assert store.read_text("epoch_0/SHA256SUMS") is not None


class FakeApi:
    def __init__(self):
        self.calls, self.files = [], {}

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, commit_message):
        self.calls.append(("folder", path_in_repo))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.calls.append(("file", path_in_repo)); self.files[path_in_repo] = path_or_fileobj

    def file_exists(self, repo_id, filename, repo_type):
        return filename in self.files


def test_hf_store_uploads_folder_before_latest_and_reads_missing_latest_as_none(tmp_path):
    api = FakeApi()
    store = HfStore("me/private-run", token="t", api=api)
    assert read_latest(store) is None
    prefix, latest = _ckpt(tmp_path / "last", 1.0, 10)
    store.upload_dir(tmp_path / "last", prefix)
    publish_latest(store, latest)
    assert api.calls == [("folder", prefix), ("file", "LATEST")]
    assert json.loads(api.files["LATEST"].decode())["prefix"] == prefix


class OnceFlakyStore(LocalStore):
    def __init__(self, root, failures):
        super().__init__(root)
        self.failures = failures

    def _put(self, src, dst):
        if self.failures:
            self.failures -= 1
            raise OSError("503 from the Hub")
        super()._put(src, dst)


def test_uploader_retries_transient_failures(tmp_path):
    # Review #6: one transient Hub error must not kill a 12 h session.
    store = OnceFlakyStore(tmp_path / "hub", failures=2)
    prefix, latest = _ckpt(tmp_path / "last", 1.0, 10)
    up = Uploader(store, retries=3, backoff=0.0); up.start()
    up.submit(tmp_path / "last", prefix, latest); up.wait(); up.close()
    assert read_latest(store) == latest


def test_uploader_gives_up_after_its_retries(tmp_path):
    store = OnceFlakyStore(tmp_path / "hub", failures=10)
    prefix, latest = _ckpt(tmp_path / "last", 1.0, 10)
    up = Uploader(store, retries=3, backoff=0.0); up.start()
    up.submit(tmp_path / "last", prefix, latest)
    with pytest.raises(OSError, match="503"):
        up.wait()
    up.close()
