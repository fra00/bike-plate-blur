#!/usr/bin/env python3
"""Unit tests for offline plate-in-vehicle interpolation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plates.track import (
    MotoSizeGate,
    build_zones,
    hold_vehicles,
    motion_gate_ok,
    moto_box_size,
    moto_size_is_large,
    zone_stats,
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


def _car(x1, y1, x2, y2, conf=0.9):
    return (2, x1, y1, x2, y2, conf)


def _plate(x1, y1, x2, y2, conf=0.5, src="crop"):
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
    zones = build_zones(
        cache, max_gap_frames=30, max_disp_px=80.0, max_disp_frac=0.35,
        conf_floor=0.15, min_moto_h_frac=0.0, frame_height=1080,
    )
    assert 10 in zones and 15 in zones
    for f in range(11, 15):
        assert f in zones, f"expected bridged frame {f}"
        assert any(z[5] == "bridge" for z in zones[f])
    st = zone_stats(zones)
    assert st["bridged_zones"] == 4
    assert st["observed_zones"] == 2


def test_bridge_skips_gap_too_long():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        10: ([_plate(120, 380, 160, 420)], [veh]),
        50: ([_plate(130, 385, 170, 425)], [veh]),
    })
    zones = build_zones(
        cache, max_gap_frames=10, max_disp_px=80.0, max_disp_frac=0.35,
        conf_floor=0.15, min_moto_h_frac=0.0, frame_height=1080,
    )
    assert 10 in zones and 50 in zones
    assert 30 not in zones
    assert zone_stats(zones)["bridged_zones"] == 0


def test_bridge_skips_implausible_jump():
    cache = _FakeCache({
        10: ([_plate(100, 380, 140, 420)], [_moto(60, 200, 200, 520)]),
        15: ([_plate(900, 380, 940, 420)], [_moto(860, 200, 1000, 520)]),
    })
    zones = build_zones(
        cache, max_gap_frames=30, max_disp_px=40.0, max_disp_frac=0.10,
        conf_floor=0.15, min_moto_h_frac=0.0, frame_height=1080,
    )
    assert 10 in zones and 15 in zones
    assert all(f not in zones for f in range(11, 15))
    assert zone_stats(zones)["bridged_zones"] == 0


def test_standalone_plate_is_dropped():
    cache = _FakeCache({
        10: ([_plate(500, 500, 560, 530, 0.9)], [_moto(80, 200, 220, 520)]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert zones == {}


def test_sahi_cache_plates_are_ignored():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        5: ([_plate(120, 380, 160, 420, 0.6, src="sahi")], [veh]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert zones == {}


def test_car_plate_is_kept():
    veh = _car(100, 200, 400, 500)
    cache = _FakeCache({
        5: ([_plate(180, 360, 280, 400, 0.6)], [veh]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert 5 in zones
    assert zones[5][0][:4] == (180, 360, 280, 400)


def test_compact_plate_wins_over_fender_blob():
    veh = _moto(80, 200, 220, 520)
    blob = _plate(90, 300, 200, 500, 0.45)      # tall blob, rh ≈ 0.62
    plate = _plate(120, 380, 165, 430, 0.40)    # compact rear plate
    cache = _FakeCache({10: ([blob, plate], [veh])})
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert len(zones[10]) == 1
    assert zones[10][0][:4] == plate[:4]


def test_tiny_moto_is_skipped():
    # 80 px tall / 60 px wide at 1080p is below 0.1065 * 1080 ≈ 115
    veh = _moto(100, 900, 160, 980)
    cache = _FakeCache({
        3: ([_plate(120, 940, 145, 965, 0.5)], [veh]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.1065,
                        frame_height=1080)
    assert zones == {}


def test_leaned_wide_moto_is_kept():
    # 90 px tall, 200 px wide: height-only gate would drop, max(w,h) keeps.
    veh = _moto(100, 900, 300, 990)
    cache = _FakeCache({
        3: ([_plate(180, 940, 220, 975, 0.5)], [veh]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.1065,
                        frame_height=1080)
    assert 3 in zones


def test_moto_box_size_is_max_side():
    assert moto_box_size((0, 0, 200, 90)) == 200
    assert moto_box_size((0, 0, 80, 150)) == 150


def test_moto_size_hysteresis_levels():
    t = 115.0
    assert moto_size_is_large(150, t, None) is True
    assert moto_size_is_large(100, t, None) is False
    # Stay large until below 0.85 * 115 ≈ 97.75
    assert moto_size_is_large(100, t, True) is True
    assert moto_size_is_large(90, t, True) is False
    # Become large only at or above 1.15 * 115 ≈ 132.25
    assert moto_size_is_large(120, t, False) is False
    assert moto_size_is_large(140, t, False) is True


def test_moto_size_gate_holds_then_drops():
    gate = MotoSizeGate(min_size=115.0)
    tall = (3, 100, 800, 180, 950, 0.6)   # 80 x 150
    mid = (3, 100, 850, 180, 950, 0.6)    # 80 x 100, IoU with tall ≈ 0.67
    short = (3, 100, 860, 180, 950, 0.6)  # 80 x 90
    assert (100, 800, 180, 950) in gate.update([tall])
    assert (100, 850, 180, 950) in gate.update([mid])
    assert (100, 860, 180, 950) not in gate.update([short])


def test_size_hysteresis_keeps_plate_across_flicker():
    # YOLO 150 px then 100 px: hard 115 px cut would drop frame 2.
    v1 = _moto(100, 800, 180, 950)
    v2 = _moto(100, 850, 180, 950)
    cache = _FakeCache({
        10: ([_plate(120, 875, 160, 905, 0.5)], [v1]),
        11: ([_plate(120, 895, 160, 925, 0.5)], [v2]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.1065,
                        frame_height=1080)
    assert 10 in zones and 11 in zones


def test_hold_vehicles_fills_short_yolo_miss():
    cache = _FakeCache({
        10: ([], [_moto(100, 400, 250, 580, 0.6)]),
        11: ([], []),
        12: ([], []),
        13: ([], [_moto(110, 405, 260, 585, 0.5)]),
    })
    held = hold_vehicles(cache, max_gap_frames=15)
    motos_11 = [v for v in held[11] if v[0] == 3]
    motos_12 = [v for v in held[12] if v[0] == 3]
    assert len(motos_11) == 1 and len(motos_12) == 1
    assert motos_11[0][1] == 103  # lerp 100 -> 110


def test_hold_vehicles_skips_teleport():
    cache = _FakeCache({
        10: ([], [_moto(100, 400, 200, 550, 0.6)]),
        11: ([], []),
        12: ([], [_moto(900, 400, 1000, 550, 0.6)]),
    })
    held = hold_vehicles(cache, max_gap_frames=15, max_disp_px=40.0,
                         max_disp_frac=0.05)
    assert [v for v in held[11] if v[0] == 3] == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("all passed")
