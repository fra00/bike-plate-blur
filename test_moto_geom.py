#!/usr/bin/env python3
"""Unit tests for geometric motorcycle base zones (no video, no models)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from plates.moto_geom import (
    MotoGeomSmoother,
    add_moto_base_zones,
    moto_base_zone,
)


def _box_on(frame, x1, y1, x2, y2, value=30):
    frame[y1:y2, x1:x2] = value


def _diag_blob(img, box, helmet_right: bool):
    """Dark / or \\ blob: wheels at bottom, helmet toward one side."""
    x1, y1, x2, y2 = box
    mx = (x1 + x2) // 2
    if helmet_right:
        bot = (mx - 50, y2 - 10)
        top = (mx + 50, y1 + 10)
    else:
        bot = (mx + 50, y2 - 10)
        top = (mx - 50, y1 + 10)
    cv2.line(img, bot, top, (18, 18, 18), 30)
    cv2.circle(img, bot, 20, (12, 12, 12), -1)


def _axes(z):
    assert z is not None and len(z) >= 10
    return float(z[7]), float(z[8]), float(z[9])


def test_base_zone_centred_on_box():
    img = np.full((200, 160, 3), 180, dtype=np.uint8)
    _box_on(img, 60, 20, 100, 190, 20)
    box = (40, 10, 120, 195)
    z = moto_base_zone(img, box, height_frac=1.0 / 3.0, min_height=10)
    assert z is not None
    x1, y1, x2, y2 = z[:4]
    src = z[5]
    angle, ax, ay = _axes(z)
    assert abs(angle) < 1e-6
    assert abs(ax - ay) < 1e-6
    assert src == "base"
    mid_y = 0.5 * (box[1] + box[3])
    assert abs(0.5 * (y1 + y2) - mid_y) < 6
    zcx = 0.5 * (x1 + x2)
    box_cx = 0.5 * (box[0] + box[2])
    assert abs(zcx - box_cx) < 6


def test_default_zone_is_circle_of_one_third_height():
    img = np.full((200, 160, 3), 180, dtype=np.uint8)
    _box_on(img, 60, 20, 100, 190, 20)
    box = (40, 10, 120, 195)
    z = moto_base_zone(img, box, min_height=10)
    assert z is not None
    angle, ax, ay = _axes(z)
    d = 2.0 * ay
    expected = (box[3] - box[1]) / 3.0
    assert abs(d - expected) <= 2
    assert abs(ax - ay) < 1e-6
    assert abs(angle) < 1e-6
    mid_y = 0.5 * (box[1] + box[3])
    assert abs(0.5 * (z[1] + z[3]) - mid_y) < 6


def test_base_zone_ignores_side_mass():
    """Circle stays on the box centre; lower-strip mass does not pull it."""
    img = np.full((200, 220, 3), 180, dtype=np.uint8)
    _box_on(img, 140, 120, 200, 195, 15)
    box = (20, 10, 200, 195)
    z = moto_base_zone(img, box, height_frac=0.2, min_height=10)
    assert z is not None
    assert z[5] == "base"
    zcx = 0.5 * (z[0] + z[2])
    box_cx = 0.5 * (box[0] + box[2])
    assert abs(zcx - box_cx) < 6


def test_zone_centre_is_half_height_from_box_bottom():
    """Y is always by2 - h/2, not an energy blob in the crop."""
    img = np.full((200, 160, 3), 180, dtype=np.uint8)
    _box_on(img, 60, 20, 100, 80, 20)   # upper mass (helmet / pack)
    _box_on(img, 55, 150, 105, 190, 15)  # wheels
    box = (40, 10, 120, 195)
    z = moto_base_zone(img, box, min_width=10, min_height=10)
    assert z is not None
    zcy = 0.5 * (z[1] + z[3])
    expected = box[3] - 0.5 * (box[3] - box[1])
    assert abs(zcy - expected) < 6


def test_smooth_h_keeps_centre_tied_to_box_bottom():
    """Helmet flicker: same by2, shorter box; cy uses smooth_h from the bottom."""
    img = np.full((240, 160, 3), 180, dtype=np.uint8)
    _box_on(img, 60, 20, 100, 220, 20)
    box = (40, 100, 120, 220)  # top jumped down; bottom stayed
    z = moto_base_zone(
        img, box, min_height=10, smooth_h=200.0,
    )
    assert z is not None
    zcy = 0.5 * (z[1] + z[3])
    # by2 - smooth_h/2 = 120, then clamped so the circle stays in the short box
    assert 118 <= zcy <= 140
    # Not the short-box midpoint (160), which would follow the jumping top
    assert zcy < 0.5 * (box[1] + box[3]) - 10


def test_slash_blob_does_not_rotate_fallback():
    """Fallback is a circle; crop energy must not tilt it."""
    img = np.full((240, 240, 3), 180, dtype=np.uint8)
    box = (20, 10, 220, 230)
    _diag_blob(img, box, helmet_right=True)
    z = moto_base_zone(img, box, min_height=10)
    assert z is not None
    assert z[5] == "base"
    assert abs(z[7]) < 1e-6
    assert abs(z[8] - z[9]) < 1e-6


def test_backslash_blob_does_not_rotate_fallback():
    img = np.full((240, 240, 3), 180, dtype=np.uint8)
    box = (20, 10, 220, 230)
    _diag_blob(img, box, helmet_right=False)
    z = moto_base_zone(img, box, min_height=10)
    assert z is not None
    assert z[5] == "base"
    assert abs(z[7]) < 1e-6


def test_smoother_damps_height_jump():
    sm = MotoGeomSmoother(alpha=0.25, alpha_pos=0.25, min_iou=0.15)
    tall = (40, 10, 120, 210)
    short = (50, 100, 110, 180)
    h1 = sm.update([tall])[tall][0]
    h2 = sm.update([short])[short][0]
    assert abs(h1 - 200) < 1
    assert 140 < h2 < 195
    assert h2 > (short[3] - short[1])


def test_add_skips_small_motos_and_cars():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    vehicles = [
        (2, 10, 10, 80, 80, 0.9),
        (3, 20, 20, 70, 90, 0.9),
        (3, 5, 5, 15, 20, 0.9),
    ]
    large = {(20, 20, 70, 90)}
    out = add_moto_base_zones(img, vehicles, [], large, enabled=True)
    assert len(out) == 1
    assert out[0][5] == "base"


def test_add_skips_all_motos_when_none_are_large():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    vehicles = [(3, 5, 5, 15, 20, 0.9), (3, 8, 8, 18, 22, 0.8)]
    out = add_moto_base_zones(img, vehicles, [], set(), enabled=True)
    assert out == []


def test_only_if_no_plate_skips_when_hit_inside():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    _box_on(img, 30, 30, 70, 90, 20)
    vehicles = [(3, 20, 20, 80, 95, 0.9)]
    large = {(20, 20, 80, 95)}
    plate = (40, 60, 55, 75, 0.4, "crop")
    out = add_moto_base_zones(
        img, vehicles, [plate], large, enabled=True, only_if_no_plate=True,
    )
    assert out == [plate]


def test_base_added_when_no_plate():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    vehicles = [(3, 20, 20, 80, 95, 0.9)]
    large = {(20, 20, 80, 95)}
    out = add_moto_base_zones(
        img, vehicles, [], large, enabled=True, only_if_no_plate=True,
    )
    assert len(out) == 1
    assert out[0][5] == "base"


if __name__ == "__main__":
    import traceback
    failed = 0
    names = [n for n in sorted(globals())
             if n.startswith("test_") and callable(globals()[n])]
    for name in names:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(names) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
