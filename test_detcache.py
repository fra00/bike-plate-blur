#!/usr/bin/env python3
"""Unit tests for detection-cache range replay."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plates.detcache import DetectionCache, _range_compatible


def test_full_cache_accepts_prefix_and_mid_clip():
    stored = {"start": None, "end": None}
    assert _range_compatible(stored, {"start": None, "end": None})
    assert _range_compatible(stored, {"start": None, "end": 30.0})
    assert _range_compatible(stored, {"start": 310.0, "end": 340.0})


def test_clip_cache_rejects_other_range():
    stored = {"start": 0.0, "end": 30.0}
    assert _range_compatible(stored, {"start": 0.0, "end": 30.0})
    assert not _range_compatible(stored, {"start": 310.0, "end": 340.0})


def test_replay_index_offsets_mid_clip():
    cache = DetectionCache.__new__(DetectionCache)
    cache.reading = True
    cache.stored_start = None
    cache.stored_end = None
    cache.meta = {"start": 310.0, "end": 340.0}
    assert cache.replay_index(0, 29.97) == int(round(310.0 * 29.97))
    assert cache.replay_index(10, 29.97) == int(round(310.0 * 29.97)) + 10


def test_replay_index_no_offset_for_prefix():
    cache = DetectionCache.__new__(DetectionCache)
    cache.reading = True
    cache.stored_start = None
    cache.stored_end = None
    cache.meta = {"start": None, "end": 30.0}
    assert cache.replay_index(7, 29.97) == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("all passed")
