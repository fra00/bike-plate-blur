#!/usr/bin/env python3
"""Unit tests for redaction helpers in blur_plates.py.

Run:  python -m pytest test_blur_plates_unit.py -v
      (or without pytest:  python test_blur_plates_unit.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

import blur_plates as bp


def test_clamp_rect_max_untouched_when_small():
    r = (10, 20, 200, 120)
    out = bp._clamp_rect_max(r, 1920, 1080, max_frac=0.20)
    assert out == r


def test_clamp_rect_max_shrinks_oversized_box():
    r = (100, 100, 1600, 900)
    out = bp._clamp_rect_max(r, 1920, 1080, max_frac=0.20)
    w, h = out[2] - out[0], out[3] - out[1]
    assert w <= 1920 * 0.20 + 1
    assert h <= 1080 * 0.20 + 1
    assert abs((out[0] + out[2]) / 2 - 850) <= 1
    assert abs((out[1] + out[3]) / 2 - 500) <= 1


def test_clamp_rect_max_keeps_extra_fields():
    r = (0, 0, 1920, 1080, 0.9, "sahi")
    out = bp._clamp_rect_max(r, 1920, 1080, max_frac=0.20)
    assert len(out) == 6
    assert out[4] == 0.9 and out[5] == "sahi"


def test_apply_blur_single_pass_changes_region():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = frame.copy()
    before = frame[30:70, 30:70].std()
    bp.apply_blur(out, [(30, 30, 70, 70)], blur_strength=35, padding=0)
    after = out[30:70, 30:70].std()
    assert after < before * 0.9
    assert (out[5, 5] == frame[5, 5]).all()
    assert (out[95, 95] == frame[95, 95]).all()


def test_blur_kernel_grows_for_close_plate():
    from plates.redact import _blur_kernel
    assert _blur_kernel(25, 40, 40) == 25
    assert _blur_kernel(35, 145, 145) >= 51


def test_apply_blur_respects_frame_bounds():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    out = frame.copy()
    bp.apply_blur(out, [(-20, -20, 20, 20), (40, 40, 99, 99)],
                  blur_strength=35, padding=8)
    assert out.shape == frame.shape


def test_apply_blur_circle_blurs_centre_not_corners():
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    out = frame.copy()
    bp.apply_blur_circle(out, [(40, 40, 80, 80)], blur_strength=35, padding=0)
    assert not (out[60, 60] == frame[60, 60]).all()
    assert (out[42, 42] == frame[42, 42]).all()
    assert (out[10, 10] == frame[10, 10]).all()


def test_apply_blur_circle_respects_frame_bounds():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    out = frame.copy()
    bp.apply_blur_circle(out, [(-20, -20, 20, 20), (40, 40, 99, 99)],
                         blur_strength=35, padding=8)
    assert out.shape == frame.shape


def _synthetic_rotated_plate():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    plate = np.full((140, 140, 3), 200, dtype=np.uint8)
    plate[25:115, 25:115] = 30
    h, w = plate.shape[:2]
    canvas = 220
    canvas_img = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    c0 = (canvas - w) // 2
    canvas_img[c0:c0 + h, c0:c0 + w] = plate
    M = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), 25.0, 1.0)
    rotated = cv2.warpAffine(canvas_img, M, (canvas, canvas))
    x0, y0 = 960 - canvas // 2, 540 - canvas // 2
    frame[y0:y0 + canvas, x0:x0 + canvas] = rotated
    return frame


def test_estimate_plate_quad_finds_rotated_plate():
    frame = _synthetic_rotated_plate()
    quad = bp.estimate_plate_quad(frame, (700, 300, 1220, 780))
    assert quad is not None
    assert quad.shape == (4, 2)
    bw = quad[:, 0].max() - quad[:, 0].min()
    bh = quad[:, 1].max() - quad[:, 1].min()
    assert 100 <= bw <= 210
    assert 100 <= bh <= 210
    cx = quad[:, 0].mean()
    cy = quad[:, 1].mean()
    assert abs(cx - 960) <= 35
    assert abs(cy - 540) <= 35
    assert bw > 150


def test_estimate_plate_quad_returns_none_on_blank():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert bp.estimate_plate_quad(frame, (700, 300, 1220, 780)) is None
    assert bp.estimate_plate_quad(frame, (0, 0, 10, 10)) is None


def test_apply_blur_rotated_blurs_only_inside_quad():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:] = (120, 120, 120)
    quad = np.array([[70, 70], [130, 70], [130, 130], [70, 130]], dtype=np.int32)
    out = bp.apply_blur_rotated(frame.copy(), [quad], blur_strength=61, padding=0)
    frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
    frame2[:] = (10, 10, 10)
    frame2[80, 100] = (250, 250, 250)
    out2 = bp.apply_blur_rotated(frame2.copy(), [quad], blur_strength=61, padding=0)
    assert out2[80, 100].max() < 250
    assert out2[10, 10].tolist() == [10, 10, 10]
    assert out[80, 100].tolist() == [120, 120, 120]


def test_apply_blur_rotated_respects_frame_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (10, 10, 10)
    quad = np.array([[-20, -20], [120, -20], [120, 120], [-20, 120]], dtype=np.int32)
    out = bp.apply_blur_rotated(frame.copy(), [quad], blur_strength=61, padding=8)
    assert out.shape == frame.shape
    assert out[50, 50].tolist() == [10, 10, 10]


def test_align_quad_reorders_rotated_permutation():
    ref = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
    new = np.array([[100, 50], [0, 50], [0, 0], [100, 0]], dtype=np.float64)
    out = bp._align_quad(ref, new)
    assert np.max(np.abs(out - ref)) < 1e-9


def test_match_quad_state_finds_nearby_rect_and_rejects_far_one():
    hist = [{"rect": (100, 100, 200, 200), "quad": None, "ratio": 0.9,
             "active": True}]
    assert bp._match_quad_state(hist, (105, 105, 195, 195)) == 0
    assert bp._match_quad_state(hist, (800, 800, 900, 900)) == -1


def _sq(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def test_quad_state_update_holds_and_glides_when_quad_lost():
    prev_quad = _sq(100, 100, 200, 200)
    st = {"rect": (0, 0, 200, 200), "quad": prev_quad.copy(),
          "ratio": 1.0, "active": True}
    qb, use = bp._quad_state_update(st, (20, 0, 220, 200), None, 200 * 200)
    assert use is True
    assert np.allclose(qb[:, 0] - prev_quad[:, 0], 20.0)
    assert np.allclose(qb[:, 1] - prev_quad[:, 1], 0.0)
    assert st["rect"] == (20, 0, 220, 200)
    assert st["active"] is True


def test_quad_state_update_weak_ratio_holds_instead_of_snapping_to_rect():
    prev_quad = _sq(0, 0, 100, 100)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 1.0, "active": True}
    weak = _sq(40, 40, 60, 60)
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100)
    assert use is True
    assert np.allclose(qb, prev_quad)


def test_quad_state_update_rejects_top_case_jump():
    prev_quad = _sq(0, 25, 100, 75)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.5, "active": True}
    top_case = _sq(0, 0, 100, 25)
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), top_case, 100 * 100)
    assert use is True
    assert np.allclose(qb, prev_quad)


def test_quad_state_update_top_case_corrected_by_plate():
    prev_quad = _sq(0, 0, 100, 25)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.25, "active": True}
    plate = _sq(0, 25, 100, 75)
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), plate, 100 * 100)
    assert use is True
    assert np.allclose(qb.mean(axis=0), [50.0, 42.5], atol=1.0)
    assert st["ratio"] >= 0.40


def test_quad_state_update_minimum_active_duration_and_fallback():
    prev_quad = _sq(0, 30, 100, 70)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.4, "active": True}
    weak = _sq(0, 35, 100, 65)
    for _ in range(4):
        qb, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                        min_active=5)
        assert use is True
        assert np.allclose(qb, prev_quad)
    _, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                   min_active=5)
    assert use is True
    assert np.allclose(st["quad"], prev_quad)
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                    min_active=5)
    assert use is True
    assert float(qb[:, 1].max() - qb[:, 1].min()) > 40.0
    for _ in range(4):
        _, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                       min_active=5)
    assert use is False


def test_quad_state_update_enter_threshold_when_inactive():
    prev_quad = _sq(0, 0, 100, 100)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.5, "active": False}
    _, use = bp._quad_state_update(st, (0, 0, 100, 100),
                                   _sq(25, 25, 75, 75), 100 * 100)
    assert use is False
    _, use = bp._quad_state_update(st, (0, 0, 100, 100),
                                   _sq(0, 0, 100, 100), 100 * 100)
    assert use is True


def test_quad_state_update_faster_ema_while_zone_moves():
    prev_quad = _sq(0, 0, 100, 100)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 1.0, "active": True}
    new_quad = _sq(20, 0, 120, 100)
    qb, use = bp._quad_state_update(st, (20, 0, 120, 100), new_quad, 100 * 100)
    assert use is True
    assert np.allclose(qb[:, 0] - prev_quad[:, 0], 13.0)
    assert np.allclose(qb[:, 1] - prev_quad[:, 1], 0.0)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK  {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {name}: {exc}")
    if failed:
        sys.exit(1)
    print("all passed")
