#!/usr/bin/env python3
"""Unit tests for plates.detect (no video file, no model load).

Run:  python test_detect.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from plates.detect import (
    _MOTO_CLS,
    _box_contains_centre,
    build_vehicle_crop_canvases,
    dedupe_vehicles,
    drop_low_conf_motos,
    enhance_crop_contrast,
    filter_moto_plate_geometry,
    letterbox_to_square,
    moto_rear_roi,
    plate_in_moto_geometry_ok,
    MotoConfCache,
    unletterbox_xyxy,
)


def test_letterbox_preserves_aspect_and_roundtrips_box():
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox_to_square(img, size=640)
    assert canvas.shape[:2] == (640, 640)
    x1, y1, x2, y2 = unletterbox_xyxy(
        pad_x, pad_y, pad_x + 100 * scale, pad_y + 200 * scale,
        scale, pad_x, pad_y, 100, 200,
    )
    assert (x1, y1, x2, y2) == (0, 0, 100, 200)


def test_letterbox_empty_image_returns_canvas():
    img = np.zeros((0, 10, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox_to_square(img, size=64)
    assert canvas.shape[:2] == (64, 64)
    assert scale == 1.0


def test_enhance_crop_contrast_keeps_shape():
    img = np.full((40, 60, 3), 80, dtype=np.uint8)
    img[10:30, 15:45] = 40
    out = enhance_crop_contrast(img, clip=2.0, grid=4)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert int(out.std()) >= int(img.std())


def test_letterbox_caps_upscale_at_max_scale():
    img = np.zeros((100, 50, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox_to_square(
        img, size=640, max_scale=2.0)
    assert canvas.shape[:2] == (640, 640)
    assert abs(scale - 2.0) < 1e-6
    # 50x100 * 2 = 100x200, centred on 640
    x1, y1, x2, y2 = unletterbox_xyxy(
        pad_x, pad_y, pad_x + 50 * scale, pad_y + 100 * scale,
        scale, pad_x, pad_y, 50, 100,
    )
    assert (x1, y1, x2, y2) == (0, 0, 50, 100)


def test_letterbox_downscales_large_crop_to_fit():
    img = np.zeros((2000, 400, 3), dtype=np.uint8)
    canvas, scale, _, _ = letterbox_to_square(
        img, size=1280, max_scale=2.0)
    assert canvas.shape[:2] == (1280, 1280)
    assert abs(scale - 1280 / 2000) < 1e-6


def test_moto_rear_roi_starts_at_bottom_frac():
    rx1, ry1, rx2, ry2 = moto_rear_roi(100, 200, 180, 400, bottom_frac=0.28,
                                       side_pad_frac=0.05)
    assert ry1 == 200 + int(200 * 0.28)
    assert ry2 == 400
    assert rx1 < 100 and rx2 > 180


def test_moto_rear_roi_clamps_to_frame():
    rx1, ry1, rx2, ry2 = moto_rear_roi(
        0, 0, 50, 100, bottom_frac=0.5, side_pad_frac=0.2,
        frame_w=80, frame_h=100,
    )
    assert rx1 >= 0 and ry1 >= 0
    assert rx2 <= 80 and ry2 <= 100


def test_plate_in_moto_geometry_accepts_anywhere_inside_box():
    box = (100, 100, 200, 300)
    mid = (140, 200, 170, 230)
    top = (140, 105, 170, 125)
    bottom = (140, 270, 170, 295)
    outside = (140, 310, 170, 330)
    assert plate_in_moto_geometry_ok(mid, box)
    assert plate_in_moto_geometry_ok(top, box)
    assert plate_in_moto_geometry_ok(bottom, box)
    assert not plate_in_moto_geometry_ok(outside, box)


def test_box_contains_centre_with_expand():
    box = (100, 100, 200, 200)
    inside = (140, 140, 160, 160)
    just_outside = (210, 140, 230, 160)
    assert _box_contains_centre(inside, box, 0.15)
    assert not _box_contains_centre(just_outside, box, 0.15)
    near_edge = (198, 140, 220, 160)  # centre ~209, expand 15 px → inside
    assert _box_contains_centre(near_edge, box, 0.15)


def test_filter_keeps_in_box_moto_plates_anywhere():
    box = (100, 100, 200, 300)
    vehicles = [(3, *box, 0.8)]
    mid = (140, 200, 170, 230, 0.5, "crop_moto")
    top = (140, 105, 170, 125, 0.5, "crop_moto")
    bottom = (140, 270, 170, 295, 0.5, "crop_moto")
    kept, rej = filter_moto_plate_geometry([mid, top, bottom], vehicles,
                                           collect_rejected=True)
    assert len(kept) == 3
    assert rej == []


def test_filter_rejects_crop_far_from_every_vehicle():
    vehicles = [(3, 100, 100, 200, 220, 0.8)]
    far = (500, 500, 520, 520, 0.40, "crop_moto")
    kept, rej = filter_moto_plate_geometry([far], vehicles, collect_rejected=True)
    assert kept == []
    assert len(rej) == 1 and rej[0][5] == "rejected"


def test_filter_keeps_standalone_sahi_outside_moto():
    vehicles = [(3, 100, 100, 200, 220, 0.8)]
    sahi = (500, 500, 560, 520, 0.50, "sahi")
    kept, rej = filter_moto_plate_geometry([sahi], vehicles, collect_rejected=True)
    assert kept == [sahi]
    assert rej == []


def test_filter_keeps_plate_that_also_overlaps_a_car():
    moto = (3, 100, 100, 200, 300, 0.8)
    car = (2, 150, 80, 400, 280, 0.9)
    plate = (160, 110, 190, 140, 0.6, "crop")  # top of moto, but overlaps car
    kept, rej = filter_moto_plate_geometry([plate], [moto, car],
                                           collect_rejected=True)
    assert kept == [plate]
    assert rej == []


def test_dedupe_merges_two_boxes_on_same_moto():
    full = (3, 100, 200, 180, 420, 0.66)
    rider = (3, 110, 210, 170, 340, 0.24)
    out = dedupe_vehicles([full, rider])
    assert len(out) == 1
    cls, x1, y1, x2, y2, conf = out[0]
    assert cls == 3
    assert conf == 0.66
    assert (x1, y1, x2, y2) == (100, 200, 180, 420)


def test_dedupe_keeps_two_separate_motos():
    left = (3, 100, 200, 180, 400, 0.7)
    right = (3, 400, 200, 480, 400, 0.6)
    out = dedupe_vehicles([left, right])
    assert len(out) == 2


def test_dedupe_does_not_merge_car_and_moto():
    moto = (3, 100, 200, 180, 400, 0.7)
    car = (2, 110, 210, 170, 380, 0.8)
    out = dedupe_vehicles([moto, car])
    assert len(out) == 2


def test_drop_low_conf_motos_keeps_cars_and_confident_bikes():
    bus_as_moto = (3, 1400, 760, 1550, 850, 0.23)
    real_moto = (3, 1588, 826, 1695, 931, 0.44)
    car = (2, 100, 200, 400, 500, 0.22)
    bus = (5, 1396, 765, 1542, 832, 0.54)
    out = drop_low_conf_motos([bus_as_moto, real_moto, car, bus], 0.30)
    assert bus_as_moto not in out
    assert real_moto in out
    assert car in out
    assert bus in out


def test_moto_conf_cache_filters_get():
    class _Inner:
        def get(self, fi):
            return (
                [(100, 100, 140, 130, 0.5, "crop")],
                [(3, 80, 50, 180, 250, 0.23), (2, 10, 20, 200, 180, 0.8)],
            )

        def frame_indices(self):
            return [0]

    wrapped = MotoConfCache(_Inner(), 0.30)
    plates, vehs = wrapped.get(0)
    assert plates[0][:4] == (100, 100, 140, 130)
    assert [v[0] for v in vehs] == [2]


def test_build_vehicle_crop_canvases_moto_emits_rear_and_full():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[50:250, 80:180] = 40
    vehicles = [(_MOTO_CLS, 80, 50, 180, 250, 0.9)]
    crops = build_vehicle_crop_canvases(
        frame, vehicles, plate_crop_imgsz=1280, crop_clahe=False)
    sources = [c[5] for c in crops]
    assert sources == ["crop_moto", "crop"]
    assert all(c[0].shape[:2] == (1280, 1280) for c in crops)


def test_build_vehicle_crop_canvases_car_is_single_full_box():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    vehicles = [(2, 10, 20, 200, 180, 0.8)]
    crops = build_vehicle_crop_canvases(
        frame, vehicles, plate_crop_imgsz=64, crop_clahe=False)
    assert len(crops) == 1
    assert crops[0][5] == "crop"
    assert crops[0][0].shape[:2] == (64, 64)
    assert crops[0][9] == 2


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
