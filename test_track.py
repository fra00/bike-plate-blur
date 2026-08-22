#!/usr/bin/env python3
"""Unit tests for offline plate-in-vehicle interpolation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plates.track import (
    MotoSizeGate,
    _collect_dets,
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


def _truck(x1, y1, x2, y2, conf=0.9):
    return (7, x1, y1, x2, y2, conf)


def test_tiny_plate_side_is_dropped():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        5: ([_plate(120, 380, 130, 390, 0.9)], [veh]),  # 10x10
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert zones == {}


def test_readable_plate_above_floor_is_kept():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        5: ([_plate(120, 380, 146, 410, 0.5)], [veh]),  # 26x30
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert 5 in zones
    assert zones[5][0][:4] == (120, 380, 146, 410)


def test_larger_plate_wins_over_blob():
    veh = _moto(80, 200, 220, 520)
    blob = _plate(120, 400, 143, 417, 0.50)   # 23x17
    real = _plate(118, 378, 152, 420, 0.20)   # 34x42
    cache = _FakeCache({10: ([blob, real], [veh])})
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert len(zones[10]) == 1
    assert zones[10][0][:4] == real[:4]


def test_nearest_plate_beats_larger_top_case():
    """Same moto: last plate is the real one; a bigger top-case blob appears."""
    veh = _moto(1656, 660, 1878, 924)
    real = _plate(1692, 792, 1745, 839, 0.70)
    blob = _plate(1748, 712, 1825, 784, 0.22)  # larger, farther (bauletto)
    cache = _FakeCache({
        10: ([real], [veh]),
        11: ([blob, real], [veh]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert zones[11][0][:4] == real[:4]


def test_size_jump_recovers_to_larger_plate():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        10: ([_plate(125, 392, 145, 408, 0.5)], [veh]),  # 20x16 slipped in
        11: ([_plate(120, 378, 154, 420, 0.4)], [veh]),  # 34x42 real
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert 10 in zones and 11 in zones
    assert zones[11][0][:4] == (120, 378, 154, 420)


def test_size_jump_blob_is_bridged():
    veh = _moto(80, 200, 220, 520)
    cache = _FakeCache({
        10: ([_plate(120, 380, 160, 420, 0.5)], [veh]),       # 40x40
        11: ([_plate(125, 392, 145, 408, 0.6)], [veh]),       # 20x16 blob
        12: ([_plate(122, 382, 162, 422, 0.5)], [veh]),       # 40x40
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert 10 in zones and 12 in zones and 11 in zones
    assert zones[11][0][5] == "bridge"
    assert zones[10][0][:4] == (120, 380, 160, 420)
    assert zones[12][0][:4] == (122, 382, 162, 422)


def test_low_conf_shrink_is_bridged_not_kept():
    """5:31-style: a few tiny low-conf boxes between two full plates.

    Skipping a blob must not forget the last full plate, or later blobs
    become observations and a nearby small-plate track can swallow them.
    """
    veh = _moto(1300, 700, 1700, 1080)
    full_a = _plate(1340, 893, 1428, 992, 0.70)   # 88x99
    blob = _plate(1509, 988, 1546, 1022, 0.17)    # 37x34
    full_b = _plate(1588, 903, 1682, 1005, 0.55)  # 94x102
    cache = _FakeCache({
        10: ([full_a], [veh]),
        11: ([blob], [veh]),
        12: ([blob], [veh]),
        13: ([blob], [veh]),
        14: ([blob], [veh]),
        20: ([full_b], [veh]),
    })
    collected = _collect_dets(
        cache, conf_floor=0.15, filter_classes={2, 3, 5, 7},
        min_moto_h_frac=0.0, frame_height=1080, plate_mem_frames=30,
        max_area_ratio=2.5,
    )
    assert [d.box for d in collected[10]] == [full_a[:4]]
    assert [d.box for d in collected[20]] == [full_b[:4]]
    for f in range(11, 20):
        assert collected.get(f, []) == [], f"blob kept as observation on {f}"
    zones = build_zones(
        cache, max_gap_frames=30, max_disp_px=80.0, max_disp_frac=0.35,
        conf_floor=0.15, min_moto_h_frac=0.0, frame_height=1080,
        max_area_ratio=2.5,
    )
    assert zones[10][0][:4] == full_a[:4]
    assert zones[20][0][:4] == full_b[:4]
    for f in range(11, 20):
        assert f in zones, f"expected bridged frame {f}"
        assert zones[f][0][5] == "bridge"
        w = zones[f][0][2] - zones[f][0][0]
        h = zones[f][0][3] - zones[f][0][1]
        assert w * h > 4000, f"interpolated box still tiny on frame {f}: {w}x{h}"


def test_nested_inner_blob_loses_to_larger_plate():
    """5:33-style: two overlapping hits on the same plate; keep the larger."""
    veh = _moto(1540, 607, 1768, 865)
    full = _plate(1648, 725, 1699, 773, 0.34)   # 51x48
    inner = _plate(1638, 700, 1672, 726, 0.49)  # 34x26 nested
    cache = _FakeCache({
        10: ([full], [veh]),
        11: ([full, inner], [veh]),
    })
    zones = build_zones(cache, conf_floor=0.15, min_moto_h_frac=0.0)
    assert zones[11][0][:4] == full[:4]


def test_gradual_shrink_is_dropped_against_peak():
    """A plate that shrinks a little each frame must still trip max_area_ratio."""
    veh = _moto(1500, 700, 1780, 1020)
    cache = _FakeCache({
        10: ([_plate(1600, 820, 1680, 900, 0.70)], [veh]),  # 80x80
        11: ([_plate(1602, 822, 1672, 892, 0.60)], [veh]),  # 70x70
        12: ([_plate(1604, 824, 1659, 879, 0.50)], [veh]),  # 55x55
        13: ([_plate(1608, 830, 1648, 870, 0.40)], [veh]),  # 40x40 blob
        14: ([_plate(1610, 832, 1640, 862, 0.30)], [veh]),  # 30x30 blob
        20: ([_plate(1620, 818, 1698, 896, 0.65)], [veh]),  # 78x78 recovered
    })
    collected = _collect_dets(
        cache, conf_floor=0.15, filter_classes={2, 3, 5, 7},
        min_moto_h_frac=0.0, frame_height=1080, plate_mem_frames=30,
        max_area_ratio=2.5,
    )
    assert (collected[10][0].box[2] - collected[10][0].box[0]) == 80
    assert (collected[11][0].box[2] - collected[11][0].box[0]) == 70
    for f in (13, 14):
        assert collected.get(f, []) == [], f"ratcheted blob kept on {f}"
    zones = build_zones(
        cache, max_gap_frames=30, max_disp_px=80.0, max_disp_frac=0.35,
        conf_floor=0.15, min_moto_h_frac=0.0, frame_height=1080,
        max_area_ratio=2.5,
    )
    assert zones[20][0][5] != "skip"
    w = zones[13][0][2] - zones[13][0][0]
    h = zones[13][0][3] - zones[13][0][1]
    assert w * h > 3000
    assert zones[13][0][5] == "bridge"


def test_hold_vehicles_class_flip_without_overlap():
    cache = _FakeCache({
        10: ([], [_moto(100, 400, 230, 560, 0.6)]),
        11: ([], [_truck(120, 390, 240, 530, 0.4)]),
        12: ([], []),
        13: ([], [_truck(150, 400, 280, 550, 0.4)]),
        14: ([], []),
        15: ([], []),
        16: ([], [_moto(280, 430, 430, 620, 0.5)]),
    })
    held = hold_vehicles(cache, max_gap_frames=30, max_class_flip_frames=10)
    for f in range(11, 16):
        motos = [v for v in held[f] if v[0] == 3]
        assert len(motos) >= 1, f"expected held moto on frame {f}"


def test_bridge_class_flip_gap_without_vehicle_overlap():
    cache = _FakeCache({
        10: ([_plate(140, 480, 175, 515, 0.5)],
             [_moto(100, 400, 230, 560, 0.6)]),
        11: ([], [_truck(120, 390, 240, 530, 0.4)]),
        12: ([], []),
        13: ([], []),
        14: ([], []),
        15: ([], []),
        16: ([_plate(320, 510, 360, 555, 0.5)],
             [_moto(280, 430, 430, 620, 0.5)]),
    })
    zones = build_zones(
        cache, max_gap_frames=30, max_disp_px=80.0, max_disp_frac=0.35,
        conf_floor=0.15, min_moto_h_frac=0.0, frame_height=1080,
    )
    assert 10 in zones and 16 in zones
    for f in range(11, 16):
        assert f in zones, f"expected bridged frame {f}"
        assert any(z[5] == "bridge" for z in zones[f])


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


def test_hold_vehicles_drops_low_conf_moto_via_cache_wrap():
    from plates.detect import MotoConfCache
    cache = _FakeCache({
        10: ([], [_moto(1400, 760, 1550, 930, 0.23)]),
        11: ([], [_moto(1402, 762, 1552, 932, 0.22)]),
    })
    wrapped = MotoConfCache(cache, 0.30)
    held = hold_vehicles(wrapped, max_gap_frames=15)
    assert [v for v in held[10] if v[0] == 3] == []
    assert [v for v in held[11] if v[0] == 3] == []


def test_hold_does_not_relabel_low_conf_truck_as_moto():
    """A bus/truck at 0.22 next to a real moto must not become motorcycle 0.22."""
    cache = _FakeCache({
        10: ([], [_moto(1588, 826, 1695, 931, 0.44),
                  _truck(1396, 765, 1542, 832, 0.22)]),
        11: ([], [_moto(1587, 829, 1690, 930, 0.46),
                  _truck(1389, 762, 1534, 829, 0.22)]),
    })
    held = hold_vehicles(cache, max_gap_frames=15, moto_min_conf=0.30)
    for f in (10, 11):
        motos = [v for v in held[f] if v[0] == 3]
        assert all(v[5] >= 0.30 for v in motos), motos
        assert any(v[1] > 1500 for v in motos)
        assert not any(v[1] < 1450 and v[5] < 0.30 for v in motos)


def test_hold_does_not_invent_second_moto_before_it_exists():
    """Close bike then a different smaller bike later: no ghost in between."""
    close = _moto(1710, 648, 1918, 930, 0.83)
    close_next = _moto(1720, 655, 1918, 930, 0.83)
    later = _moto(1662, 702, 1764, 804, 0.83)
    cache = _FakeCache({
        10: ([], [close]),
        11: ([], [close_next]),
        12: ([], [close_next]),
        20: ([], [later]),
    })
    held = hold_vehicles(cache, max_gap_frames=15, max_disp_px=80.0,
                         max_disp_frac=0.35)
    for f in (11, 12):
        motos = [v for v in held[f] if v[0] == 3]
        assert len(motos) == 1, f"expected only YOLO moto on {f}, got {motos}"


def test_hold_vehicles_forks_merged_moto_split():
    """YOLO one wide box for two bikes, then only the left, then both."""
    wide = _moto(100, 400, 400, 700, 0.6)
    left = _moto(100, 400, 250, 700, 0.6)
    right = _moto(270, 420, 400, 680, 0.5)
    cache = _FakeCache({
        10: ([], [wide]),
        11: ([], [left]),
        12: ([], [left]),
        13: ([], [left, right]),
    })
    held = hold_vehicles(cache, max_gap_frames=15, max_disp_px=80.0,
                         max_disp_frac=0.35)
    for f in (11, 12):
        motos = [v for v in held[f] if v[0] == 3]
        assert len(motos) >= 2, f"expected left+right holds on frame {f}, got {motos}"
        cxs = [0.5 * (v[1] + v[3]) for v in motos]
        assert min(cxs) < 200 and max(cxs) > 280


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("all passed")
