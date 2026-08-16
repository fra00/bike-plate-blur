#!/usr/bin/env python3
"""Unit tests for offline past+future zone bridging."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plates.offline_zones import (
    build_offline_zones,
    merge_tracker_with_offline_fills,
    motion_gate_ok,
    offline_zone_stats,
)


class _FakeCache:
    def __init__(self, data: dict):
        self._data = data

    def frame_indices(self):
        return sorted(self._data)

    def get(self, fi):
        return self._data.get(fi)


def _moto(x1, y1, x2, y2, conf=0.9):
    return (3, x1, y1, x2, y2, conf)


def _plate(x1, y1, x2, y2, conf=0.5, src="sahi"):
    return (x1, y1, x2, y2, conf, src)


def test_motion_gate_allows_small_travel():
    a = (100, 100, 140, 130)
    b = (110, 105, 150, 135)
    assert motion_gate_ok(a, b, gap_frames=5, vehicle=(50, 50, 200, 400),
                          max_disp_px=80.0, max_disp_frac=0.35)


def test_motion_gate_rejects_teleport():
    a = (100, 100, 140, 130)
    b = (900, 100, 940, 130)
    assert not motion_gate_ok(a, b, gap_frames=3, vehicle=(50, 50, 200, 400),
                              max_disp_px=80.0, max_disp_frac=0.35)


def test_bridge_fills_short_gap():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        10: ([_plate(120, 380, 160, 420)], [veh]),
        15: ([_plate(135, 385, 175, 425)], [veh]),
    })
    zones = build_offline_zones(
        cache, max_gap_frames=30, max_disp_px=80.0, max_disp_frac=0.35,
        conf_floor=0.15, moto_only=True, min_blur_box_h_frac=0.0,
        frame_height=1080,
    )
    assert 10 in zones and 15 in zones
    for f in range(11, 15):
        assert f in zones, f"expected bridged frame {f}"
        assert any(z[5] == "bridge" for z in zones[f])
    st = offline_zone_stats(zones)
    assert st["bridged_zones"] == 4
    assert st["raw_zones"] == 2


def test_bridge_skips_gap_too_long():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        10: ([_plate(120, 380, 160, 420)], [veh]),
        50: ([_plate(130, 385, 170, 425)], [veh]),
    })
    zones = build_offline_zones(
        cache, max_gap_frames=10, max_disp_px=80.0, max_disp_frac=0.35,
        conf_floor=0.15, moto_only=True, min_blur_box_h_frac=0.0,
        frame_height=1080,
    )
    assert 10 in zones and 50 in zones
    assert 30 not in zones
    assert offline_zone_stats(zones)["bridged_zones"] == 0


def test_bridge_skips_implausible_jump():
    cache = _FakeCache({
        10: ([_plate(100, 380, 140, 420)], [_moto(60, 200, 200, 520)]),
        15: ([_plate(900, 380, 940, 420)], [_moto(860, 200, 1000, 520)]),
    })
    zones = build_offline_zones(
        cache, max_gap_frames=30, max_disp_px=40.0, max_disp_frac=0.10,
        conf_floor=0.15, moto_only=True, min_blur_box_h_frac=0.0,
        frame_height=1080,
    )
    assert 10 in zones and 15 in zones
    assert all(f not in zones for f in range(11, 15))
    assert offline_zone_stats(zones)["bridged_zones"] == 0


def test_merge_keeps_tracker_when_present():
    tracker = [(100, 100, 140, 130, -2.0, "anchor", 1)]
    offline = [(200, 200, 240, 230, 0.3, "bridge", 2)]
    out = merge_tracker_with_offline_fills(tracker, offline)
    assert out == tracker


def test_merge_fills_only_bridges_on_blank():
    offline = [
        (100, 100, 140, 130, 0.4, "offline", 1),
        (110, 110, 150, 140, 0.3, "bridge", 1),
    ]
    out = merge_tracker_with_offline_fills([], offline)
    assert len(out) == 1
    assert out[0][5] == "bridge"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("all passed")
