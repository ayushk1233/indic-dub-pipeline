from prep.globe_slice import select


def _rows(n_speakers=40, per=30, dur=6.0):
    return [{"speaker_id": f"S{s:03d}", "transcript": "hello there", "duration": dur, "accent": "India",
             "shard": "train-00000-of-00108.parquet", "row_group": s, "row_in_group": i}
            for s in range(n_speakers) for i in range(per)]


def test_split_is_by_speaker():
    out = select(_rows(), train_hours=0.5, val_clips=40)
    assert not {r["speaker_id"] for r in out["train"]} & {r["speaker_id"] for r in out["val"]}
    assert len(out["val"]) == 40


def test_duration_window_and_cap():
    rows = _rows(per=400, dur=6.0) + [dict(_rows(1)[0], duration=2.0, row_in_group=999),
                                      dict(_rows(1)[0], duration=25.0, row_in_group=998)]
    out = select(rows, train_hours=100, val_clips=10, cap_s=1200)
    kept = out["train"] + out["val"]
    assert all(3.0 <= r["duration"] <= 20.0 for r in kept)
    per = {}
    for r in out["train"]:
        per[r["speaker_id"]] = per.get(r["speaker_id"], 0) + r["duration"]
    assert max(per.values()) <= 1200


def test_selection_is_deterministic():
    assert select(_rows(), train_hours=0.5, val_clips=40) == select(_rows(), train_hours=0.5, val_clips=40)
