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
    bottom_mass_cx,
    energy_pose,
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


def test_upright_bottom_mass_near_centre():
    img = np.full((200, 160, 3), 180, dtype=np.uint8)
    # Tall dark bike in the middle of the box
    _box_on(img, 60, 20, 100, 190, 20)
    box = (40, 10, 120, 195)
    cx = bottom_mass_cx(img, box, strip_frac=0.30)
    box_cx = 0.5 * (box[0] + box[2])
    assert abs(cx - box_cx) < 8


def test_leaned_bottom_mass_shifts_to_wheel_side():
    img = np.full((200, 220, 3), 180, dtype=np.uint8)
    # Helmet / torso on the left, wheels / tail packed on the right-bottom.
    _box_on(img, 30, 15, 70, 80, 25)
    _box_on(img, 130, 115, 205, 195, 15)
    box = (20, 10, 210, 198)
    cx = bottom_mass_cx(img, box, strip_frac=0.30)
    box_cx = 0.5 * (box[0] + box[2])
    assert cx > box_cx + 8


def test_base_zone_centred_on_rear_when_upright():
    img = np.full((200, 160, 3), 180, dtype=np.uint8)
    _box_on(img, 60, 20, 100, 190, 20)
    box = (40, 10, 120, 195)
    z = moto_base_zone(img, box, width_frac=0.5, height_frac=0.2,
                       aspect=1.82, min_width=10, min_height=10)
    assert z is not None
    x1, y1, x2, y2 = z[:4]
    src = z[5]
    angle, ax, ay = _axes(z)
    assert (x2 - x1) > (y2 - y1)
    assert abs(angle) < 1
    assert abs(ax / ay - 1.82) < 0.12
    mid_y = 0.5 * (box[1] + box[3])
    assert abs(0.5 * (y1 + y2) - mid_y) < 6
    assert src == "base"
    zcx = 0.5 * (x1 + x2)
    box_cx = 0.5 * (box[0] + box[2])
    assert abs(zcx - box_cx) < 6


def test_default_zone_is_wide_oval_of_one_third_height():
    img = np.full((200, 160, 3), 180, dtype=np.uint8)
    _box_on(img, 60, 20, 100, 190, 20)
    box = (40, 10, 120, 195)
    z = moto_base_zone(img, box, min_width=10, min_height=10, aspect=1.82)
    assert z is not None
    x1, y1, x2, y2 = z[:4]
    angle, ax, ay = _axes(z)
    zh = 2.0 * ay
    expected = (box[3] - box[1]) / 3.0
    assert abs(zh - expected) <= 2
    assert abs((2.0 * ax) / zh - 1.82) < 0.12
    assert abs(angle) < 1
    mid_y = 0.5 * (box[1] + box[3])
    assert abs(0.5 * (y1 + y2) - mid_y) < 6


def test_base_zone_labelled_lean_when_mass_offset():
    img = np.full((200, 220, 3), 180, dtype=np.uint8)
    _box_on(img, 140, 120, 200, 195, 15)
    box = (20, 10, 200, 195)
    z = moto_base_zone(
        img, box, width_frac=0.35, height_frac=0.2,
        min_width=10, min_height=10, snap_frac=0.04, max_shift_frac=0.35,
    )
    assert z is not None
    assert z[5] in ("base", "base_lean")


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
        img, box, min_width=10, min_height=10, aspect=1.82, smooth_h=200.0,
    )
    assert z is not None
    zcy = 0.5 * (z[1] + z[3])
    # by2 - smooth_h/2 = 120, then clamped so the oval stays in the short box
    assert 118 <= zcy <= 140
    # Not the short-box midpoint (160), which would follow the jumping top
    assert zcy < 0.5 * (box[1] + box[3]) - 10


def test_slash_blob_rotates_major_axis():
    """Helmet right of wheels (/): major axis tilts (non-zero)."""
    img = np.full((240, 240, 3), 180, dtype=np.uint8)
    box = (20, 10, 220, 230)
    _diag_blob(img, box, helmet_right=True)
    _cx, _cy, angle = energy_pose(img, box, snap_angle=8.0)
    assert abs(angle) > 10
    z = moto_base_zone(img, box, min_width=10, min_height=10, snap_angle=8.0)
    assert z is not None
    assert z[5] == "base_lean"
    assert abs(z[7]) > 10


def test_backslash_blob_rotates_major_axis():
    """Helmet left of wheels (\\): major axis tilts the opposite way from /."""
    img_slash = np.full((240, 240, 3), 180, dtype=np.uint8)
    img_back = np.full((240, 240, 3), 180, dtype=np.uint8)
    box = (20, 10, 220, 230)
    _diag_blob(img_slash, box, helmet_right=True)
    _diag_blob(img_back, box, helmet_right=False)
    ang_slash = energy_pose(img_slash, box, snap_angle=8.0)[2]
    ang_back = energy_pose(img_back, box, snap_angle=8.0)[2]
    assert abs(ang_slash) > 10
    assert abs(ang_back) > 10
    assert ang_slash * ang_back < 0
    z = moto_base_zone(img_back, box, min_width=10, min_height=10, snap_angle=8.0)
    assert z is not None
    assert z[5] == "base_lean"
    assert z[7] * ang_slash < 0


def test_smoother_damps_height_jump():
    sm = MotoGeomSmoother(alpha=0.25, alpha_pos=0.25, min_iou=0.15)
    tall = (40, 10, 120, 210)
    short = (50, 100, 110, 180)
    h1 = sm.update([tall])[tall][0]
    h2 = sm.update([short])[short][0]
    assert abs(h1 - 200) < 1
    assert 140 < h2 < 195
    assert h2 > (short[3] - short[1])


def test_smoother_damps_angle_jump():
    sm = MotoGeomSmoother(alpha=1.0, alpha_pos=1.0, alpha_ang=0.28, min_iou=0.15)
    box = (40, 10, 120, 210)
    sm.update([(box, 200.0, 150.0, 80.0, 0.0)])
    ang = sm.update([(box, 200.0, 150.0, 80.0, 40.0)])[box][3]
    assert 8 < ang < 20


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
    assert out[0][5] in ("base", "base_lean")


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
