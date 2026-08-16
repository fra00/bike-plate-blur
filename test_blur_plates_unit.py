#!/usr/bin/env python3
"""Unit tests for the modified logic in blur_plates.py.

Run:  python -m pytest test_blur_plates_unit.py -v
      (or without pytest:  python test_blur_plates_unit.py)

Covers the changes made for the moto pipeline:
  - _clamp_rect_max (max blur box size)
  - apply_blur (single-pass, lighter)
  - VehicleTrack: ema_update, clamp_ar, set_plate_anchor / predict_anchored,
    anchor_rect, fallback_rect
  - PlateHistory: predict_rect displacement sanity cap,
    predict_rect_lk displacement cap
  - SceneTracker: anchor-only mode for motorcycles (no plate detection)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

import blur_plates as bp


# ─── _clamp_rect_max ──────────────────────────────────────────────────────────

def test_clamp_rect_max_untouched_when_small():
    r = (10, 20, 200, 120)  # 190x100 in a 1920x1080 frame
    out = bp._clamp_rect_max(r, 1920, 1080, max_frac=0.20)
    assert out == r


def test_clamp_rect_max_shrinks_oversized_box():
    # 1500x800 covers ~78% x 74% of frame → must shrink toward centre
    r = (100, 100, 1600, 900)
    out = bp._clamp_rect_max(r, 1920, 1080, max_frac=0.20)
    w, h = out[2] - out[0], out[3] - out[1]
    assert w <= 1920 * 0.20 + 1
    assert h <= 1080 * 0.20 + 1
    # centre preserved
    assert abs((out[0] + out[2]) / 2 - 850) <= 1
    assert abs((out[1] + out[3]) / 2 - 500) <= 1


def test_clamp_rect_max_keeps_extra_fields():
    r = (0, 0, 1920, 1080, 0.9, "sahi")
    out = bp._clamp_rect_max(r, 1920, 1080, max_frac=0.20)
    assert len(out) == 6
    assert out[4] == 0.9 and out[5] == "sahi"


# ─── apply_blur ───────────────────────────────────────────────────────────────

def test_apply_blur_single_pass_changes_region():
    # textured region (noise) — a flat-white region is unchanged by any blur
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = frame.copy()
    before = frame[30:70, 30:70].std()
    bp.apply_blur(out, [(30, 30, 70, 70)], blur_strength=35, padding=0)
    after = out[30:70, 30:70].std()
    # smoothing must visibly reduce the noise inside the region
    assert after < before * 0.9
    # pixels outside the region untouched
    assert (out[5, 5] == frame[5, 5]).all()
    assert (out[95, 95] == frame[95, 95]).all()


def test_apply_blur_respects_frame_bounds():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    out = frame.copy()
    bp.apply_blur(out, [(-20, -20, 20, 20), (40, 40, 99, 99)],
                  blur_strength=35, padding=8)
    assert out.shape == frame.shape


def test_apply_blur_round_covers_rect_corners():
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    out = frame.copy()
    bp.apply_blur_round(out, [(40, 40, 80, 80)], blur_strength=35, padding=0)
    # ellipse centre must be blurred
    assert not (out[60, 60] == frame[60, 60]).all()
    # rect corners must ALSO be blurred (√2 axes): Italian plate characters sit
    # in those corners and were left readable by the old inscribed ellipse.
    assert not (out[42, 42] == frame[42, 42]).all()
    # well outside the expanded ellipse: untouched
    assert (out[10, 10] == frame[10, 10]).all()


def test_apply_blur_round_respects_frame_bounds():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    out = frame.copy()
    bp.apply_blur_round(out, [(-20, -20, 20, 20), (40, 40, 99, 99)],
                        blur_strength=35, padding=8)
    assert out.shape == frame.shape


# ─── VehicleTrack.ema_update ─────────────────────────────────────────────────

def test_ema_update_first_call_initializes():
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    e = vt.ema_update((0, 0, 100, 100), 0.6)
    assert e == (50.0, 50.0, 100.0, 100.0)


def test_ema_update_blends():
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    vt.ema_update((0, 0, 100, 100), 0.6)          # init at (50, 50, 100, 100)
    e = vt.ema_update((100, 100, 200, 200), 0.6)  # new centre (150, 150)
    # alpha=0.6 → 0.6*150 + 0.4*50 = 110
    assert abs(e[0] - 110.0) < 1e-6
    assert abs(e[1] - 110.0) < 1e-6
    # width: 0.6*100 + 0.4*100 = 100
    assert abs(e[2] - 100.0) < 1e-6


# ─── VehicleTrack.clamp_ar ────────────────────────────────────────────────────

def test_clamp_ar_wide_car_plate_not_touched_without_constraint():
    # clamp_ar with moto range 0.9–1.3 on a wide box must widen height
    vt = bp.VehicleTrack((0, 0, 200, 100), 0, cls=3)
    out = vt.clamp_ar((0, 0, 200, 100), ar_min=0.9, ar_max=1.3)
    ar = (out[2] - out[0]) / max(1e-6, out[3] - out[1])
    assert 0.9 <= ar <= 1.3 + 0.01
    # centre preserved
    assert abs((out[0] + out[2]) / 2 - 100) <= 1
    assert abs((out[1] + out[3]) / 2 - 50) <= 1


def test_clamp_ar_square_box_untouched():
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    out = vt.clamp_ar((0, 0, 100, 100), ar_min=0.9, ar_max=1.3)
    assert out == (0, 0, 100, 100)


# ─── VehicleTrack anchor ──────────────────────────────────────────────────────

def test_predict_anchored_none_without_anchor():
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    assert vt.predict_anchored() is None


def test_predict_anchored_follows_vehicle_box():
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    vt.set_plate_anchor((40, 70, 60, 90))        # centre (50, 80), size 20x20
    # vehicle moves to (200, 100)-(300, 200)
    vt.update_box((200, 100, 300, 200), 1)
    out = vt.predict_anchored(max_expand=0)
    # plate centre should track vehicle-relative offset: dx=0.5, dy=0.8
    cx = (out[0] + out[2]) / 2
    cy = (out[1] + out[3]) / 2
    assert abs(cx - 250) <= 1
    assert abs(cy - 180) <= 1


def test_predict_anchored_stays_inside_vehicle_clamp():
    # anchor at the far edge of the vehicle; prediction must stay within
    # the vehicle box expanded by 10%
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    vt.set_plate_anchor((95, 95, 99, 99))        # corner anchor
    vt.update_box((200, 100, 300, 200), 1)
    out = vt.predict_anchored(max_expand=0)
    vx1, vy1, vx2, vy2 = 200, 100, 300, 200
    pad_x = (vx2 - vx1) * 0.10
    pad_y = (vy2 - vy1) * 0.10
    assert out[0] >= int(vx1 - pad_x)
    assert out[1] >= int(vy1 - pad_y)
    assert out[2] <= int(vx2 + pad_x)
    assert out[3] <= int(vy2 + pad_y)


# ─── VehicleTrack.anchor_rect (anchor-only moto mode) ─────────────────────────

def test_anchor_rect_geometry():
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    r = vt.anchor_rect(frac=0.30, y_frac=0.60, pad_frac=0.05)
    x1, y1, x2, y2 = r
    side = 100 * 0.30
    pad = side * 0.05
    # centred horizontally, centred vertically at y=60
    assert abs((x1 + x2) / 2 - 50) <= 1
    assert abs((y1 + y2) / 2 - 60) <= 2
    # roughly square
    assert abs((x2 - x1) - (y2 - y1)) <= 4


def test_anchor_rect_clamped_to_frame():
    vt = bp.VehicleTrack((0, 0, 50, 50), 0, cls=3)
    r = vt.anchor_rect(frac=0.30, y_frac=0.60, pad_frac=0.05)
    assert r[0] >= 0 and r[1] >= 0


# ─── VehicleTrack.fallback_rect ───────────────────────────────────────────────

def test_fallback_rect_is_bottom_strip():
    vt = bp.VehicleTrack((0, 0, 100, 200), 0, cls=3)
    vt.fallback_frac     = 0.40   # SceneTracker copies these onto the track
    vt.fallback_pad_frac = 0.25   # (they are not set in __init__)
    r = vt.fallback_rect()
    x1, y1, x2, y2 = r
    h = y2 - y1
    # strip = 40% of vehicle height, padded 25% on BOTH top and bottom
    assert 0.40 * 200 + 2 * 200 * 0.40 * 0.25 >= h >= 0.40 * 200
    assert y2 >= 200                          # bottom edge reaches the box bottom
    assert y1 >= 0.40 * 200 - 25 - 2          # strip starts near the bottom half
    assert x1 >= 0 and x2 >= x1


# ─── PlateHistory.predict_rect (displacement sanity cap) ─────────────────────

def test_predict_rect_caps_runaway_extrapolation():
    ph = bp.PlateHistory(max_history=15)
    # fast motion: 100 px/frame
    ph.record(0, 0, 0, 100, 100, 0.9)
    ph.record(1, 100, 100, 200, 200, 0.9)
    ph.record(2, 200, 200, 300, 300, 0.9)
    # extrapolate 20 frames ahead at 100 px/frame → 2000 px, way over cap 40
    out = ph.predict_rect(22, max_expand=20, max_px_per_frame=40.0)
    assert out is None


def test_predict_rect_allowed_within_cap():
    ph = bp.PlateHistory(max_history=15)
    ph.record(0, 0, 0, 100, 100, 0.9)
    ph.record(1, 5, 5, 105, 105, 0.9)            # 5 px/frame
    out = ph.predict_rect(3, max_expand=20, max_px_per_frame=40.0)
    assert out is not None


def test_predict_rect_none_without_history():
    ph = bp.PlateHistory(max_history=15)
    assert ph.predict_rect(5, max_expand=20, max_px_per_frame=40.0) is None


# ─── PlateHistory.predict_rect_lk (LK displacement cap) ──────────────────────

def test_predict_rect_lk_rejects_latched_points():
    ph = bp.PlateHistory(max_history=15)
    ph.record(0, 0, 0, 100, 100, 0.9)
    rng = np.random.default_rng(0)
    gray0 = rng.integers(0, 255, (300, 300), dtype=np.uint8)   # textured bg
    gray0[10:90, 10:90] = rng.integers(0, 255, (80, 80), dtype=np.uint8)
    ph.refresh_lk(gray0, 10, 10, 90, 90)
    # next frame: the same texture appears 200px away (background latch)
    gray1 = gray0.copy()
    gray1[:] = 0
    gray1[210:290, 210:290] = gray0[10:90, 10:90]
    out = ph.predict_rect_lk(gray1, max_expand=20, max_px_per_frame=40.0)
    assert out is None


def test_predict_rect_lk_accepts_small_motion():
    ph = bp.PlateHistory(max_history=15)
    ph.record(0, 0, 0, 100, 100, 0.9)
    rng = np.random.default_rng(1)
    gray0 = rng.integers(0, 255, (300, 300), dtype=np.uint8)
    gray0[10:90, 10:90] = rng.integers(0, 255, (80, 80), dtype=np.uint8)
    ph.refresh_lk(gray0, 10, 10, 90, 90)
    gray1 = np.roll(gray0, shift=(3, 3), axis=(0, 1))   # small 3px shift
    out = ph.predict_rect_lk(gray1, max_expand=20, max_px_per_frame=40.0)
    assert out is not None


# ─── SceneTracker anchor-only mode for motorcycles ────────────────────────────

def _make_tracker_with_moto():
    tr = bp.SceneTracker(max_gap_frames=10, fallback_min_frames=1,
                         moto_anchor=True, moto_anchor_frac=0.30,
                         moto_anchor_y=0.60, moto_anchor_pad=0.05)
    vt = bp.VehicleTrack((100, 100, 200, 160), 0, cls=3)   # motorcycle
    vt.id = 1
    vt.frames_seen = 1
    tr._track_dict[1] = vt
    return tr, vt


def test_anchor_mode_emits_zone_for_moto():
    tr, vt = _make_tracker_with_moto()
    plates, _ = tr.update([], [])    # no vehicle dets, no plate dets this frame
    anchors = [p for p in plates if len(p) > 5 and p[5] == "anchor"]
    assert len(anchors) == 1


def test_anchor_mode_swallows_plate_detections():
    tr, vt = _make_tracker_with_moto()
    # a plate detection overlapping the moto box: must be claimed, not passed
    plates, _ = tr.update([], [(110, 120, 150, 160, 0.9, "sahi")])
    anchors = [p for p in plates if len(p) > 5 and p[5] == "anchor"]
    sahi = [p for p in plates if len(p) > 5 and p[5] == "sahi"]
    assert len(anchors) == 1
    assert len(sahi) == 0


def test_anchor_mode_disabled_keeps_detections():
    tr = bp.SceneTracker(max_gap_frames=10, fallback_min_frames=1,
                         moto_anchor=False)
    vt = bp.VehicleTrack((100, 100, 200, 160), 0, cls=3)
    vt.id = 1
    vt.frames_seen = 1
    tr._track_dict[1] = vt
    plates, _ = tr.update([], [(110, 120, 150, 160, 0.9, "sahi")])
    sahi = [p for p in plates if len(p) > 5 and p[5] == "sahi"]
    assert len(sahi) == 1
    assert not any(len(p) > 5 and p[5] == "anchor" for p in plates)


# ─── SceneTracker: anchor + EMA state survive updates ────────────────────────

def test_anchor_mode_track_keeps_working_across_frames():
    tr, vt = _make_tracker_with_moto()
    for _ in range(3):
        plates, _ = tr.update([], [])
        assert any(len(p) > 5 and p[5] == "anchor" for p in plates)


# ─── Anti-flicker: EMA on the anchor zone ─────────────────────────────────────

def test_zone_ema_smooths_dimensional_jump():
    vt = bp.VehicleTrack((100, 100, 200, 200), 0, cls=3)   # 100 px tall
    z1 = vt.anchor_rect(frac=0.30, y_frac=0.60, pad_frac=0.05)
    vt.zone_ema_update(z1, alpha=0.6)
    # vehicle grows suddenly (moto approaching): 100 → 400 px tall
    vt.box = (100, 100, 200, 400)
    z2 = vt.anchor_rect(frac=0.30, y_frac=0.60, pad_frac=0.05)
    vt.zone_ema_update(z2, alpha=0.6)
    ex, ey, ew, eh = vt.zone_ema
    # EMA side after one step: 0.6 * 120 + 0.4 * 30 = 84  (vs raw 120)
    assert 60 <= eh < 120
    # second step converges toward the target but never snaps
    vt.zone_ema_update(z2, alpha=0.6)
    assert vt.zone_ema[3] > eh


def test_zone_ema_glides_not_snaps_on_jump():
    vt = bp.VehicleTrack((0, 0, 100, 100), 0, cls=3)
    z1 = vt.anchor_rect(frac=0.30, y_frac=0.60, pad_frac=0.05)
    vt.zone_ema_update(z1, alpha=0.6)
    vt.box = (500, 0, 600, 100)   # moto teleports 500 px right
    z2 = vt.anchor_rect(frac=0.30, y_frac=0.60, pad_frac=0.05)
    vt.zone_ema_update(z2, alpha=0.6)
    cx = vt.zone_ema[0]
    # after one EMA step the zone centre is between the old and new position
    assert 50 < cx < 550


def test_predict_box_follows_velocity():
    vt = bp.VehicleTrack((100, 100, 200, 200), 0, cls=3)
    vt.update_box((120, 100, 220, 200), 1)     # +20 px/frame rightward
    vt.update_box((140, 100, 240, 200), 2)
    pred = vt.predict_box(4, max_disp=40.0)
    assert pred is not None
    assert pred[0] > 140                       # extrapolated further right
    assert (pred[2] - pred[0]) == 100          # size preserved


def test_predict_box_none_without_velocity():
    vt = bp.VehicleTrack((100, 100, 200, 200), 0, cls=3)
    assert vt.predict_box(5, max_disp=40.0) is None


def test_predict_box_none_on_implausible_velocity():
    vt = bp.VehicleTrack((100, 100, 200, 200), 0, cls=3)
    vt.update_box((700, 100, 800, 200), 1)    # 600 px jump in one frame
    vt.update_box((1300, 100, 1400, 200), 2)
    assert vt.predict_box(3, max_disp=40.0) is None


# ─── Anti-flicker: ghost zone persists after track deletion ──────────────────

def test_ghost_zone_emitted_after_track_dies():
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_ghost_frames=6,
                         moto_anchor_frac=0.30, moto_anchor_y=0.60,
                         moto_anchor_pad=0.05)
    vt = bp.VehicleTrack((100, 100, 200, 160), 0, cls=3)
    vt.id = 1
    vt.frames_seen = 1
    tr._track_dict[1] = vt
    tr.update([], [])           # frame 0: zone recorded (EMA init)
    tr.update([], [])           # frame 1: missed
    tr.update([], [])           # frame 2: missed → miss_count 3 > max_gap 2 → track dies, ghost saved
    counts = []
    for _ in range(7):          # ghost persists moto_ghost_frames=6, then dies
        plates, _ = tr.update([], [])
        counts.append(sum(1 for p in plates if len(p) > 5 and p[5] == "anchor"))
    assert counts == [1, 1, 1, 1, 1, 1, 0]


def test_ghost_zone_dropped_when_track_resumes():
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_ghost_frames=6,
                         moto_anchor_frac=0.30, moto_anchor_y=0.60,
                         moto_anchor_pad=0.05)
    vt = bp.VehicleTrack((100, 100, 200, 160), 0, cls=3)
    vt.id = 1
    vt.frames_seen = 1
    tr._track_dict[1] = vt
    tr.update([], [])           # frame 0: zone recorded
    tr.update([], [])           # frame 1: missed
    tr.update([], [])           # frame 2: missed → track dies, ghost saved
    plates, _ = tr.update([(3, 100, 100, 200, 160, 0.9)], [])   # moto re-detected
    anchors = [p for p in plates if len(p) > 5 and p[5] == "anchor"]
    # ghost must be dropped in favour of the live track's single zone
    assert len(anchors) == 1


# ─── Close-range motorcycle handling ─────────────────────────────────────────

def _anchors(plates):
    return [p for p in plates if len(p) > 5 and p[5] == "anchor"]


def test_close_moto_low_conf_accepted():
    # A huge box (close-up moto) at low confidence must still start a track
    # instantly (promoted), so the anchor zone appears from the first frame.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_anchor_frac=0.30,
                         moto_anchor_y=0.60, moto_anchor_pad=0.05)
    plates, _ = tr.update(
        [(3, 900, 100, 1500, 900, 0.22)], [],   # 800px tall on a 1080p frame
        frame_size=(1920, 1080),
    )
    anchors = _anchors(plates)
    assert len(anchors) == 1
    assert len(tr._track_dict) == 1


def test_small_box_low_conf_rejected():
    # A small box at the same low confidence must NOT start a track.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_anchor_frac=0.30,
                         moto_anchor_y=0.60, moto_anchor_pad=0.05)
    plates, _ = tr.update(
        [(3, 100, 100, 300, 200, 0.22)], [],   # 100px tall → not close
        frame_size=(1920, 1080),
    )
    assert len(_anchors(plates)) == 0
    assert len(tr._track_dict) == 0


def test_close_moto_zone_frozen_while_missed():
    # Close moto: while the detector misses it, the box must NOT move on
    # Kalman prediction (no drift, no blur displacement). The zone EMA may
    # converge for a couple of frames but settles; the ghost that follows the
    # track's death keeps the settled zone exactly.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_ghost_frames=6,
                         moto_anchor_frac=0.30, moto_anchor_y=0.60,
                         moto_anchor_pad=0.05)
    tr.update([(3, 900, 100, 1500, 900, 0.9)], [], frame_size=(1920, 1080))
    tr.update([(3, 950, 100, 1550, 900, 0.9)], [], frame_size=(1920, 1080))
    tr.update([], [], frame_size=(1920, 1080))   # f3: missed
    p5, _ = tr.update([], [], frame_size=(1920, 1080))   # f4: missed (still alive)
    t = list(tr._track_dict.values())[0]
    assert t.box == (943, 100, 1543, 900)        # box frozen at last real box
    assert t.last_detected_frame < tr.frame_idx
    p6, _ = tr.update([], [], frame_size=(1920, 1080))   # f5: track dies → ghost saved
    p7, _ = tr.update([], [], frame_size=(1920, 1080))   # f6: ghost frame
    a5, a6 = _anchors(p5)[0], _anchors(p6)[0]
    assert abs((a5[0] + a5[2]) - (a6[0] + a6[2])) <= 1   # zone settled
    a7 = _anchors(p7)[0]
    assert (a6[0] + a6[2]) == (a7[0] + a7[2])            # ghost frozen too
    assert (a6[1] + a6[3]) == (a7[1] + a7[3])


def test_close_moto_zone_wider_than_tall():
    # Widened zone: a close/leaning moto gets a horizontally wider blur box
    # covering the plate's lean offset.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_anchor_frac=0.30,
                         moto_anchor_y=0.60, moto_anchor_pad=0.05)
    plates, _ = tr.update(
        [(3, 900, 100, 1500, 900, 0.9)], [], frame_size=(1920, 1080),
    )
    z = _anchors(plates)[0]
    assert (z[2] - z[0]) > (z[3] - z[1])


def test_zone_extended_to_right_edge():
    # Box cut at the right frame edge → the zone extends all the way to the
    # edge so the visible part of the plate stays covered.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_anchor_frac=0.30,
                         moto_anchor_y=0.60, moto_anchor_pad=0.05)
    plates, _ = tr.update(
        [(3, 1500, 200, 1916, 900, 0.9)], [], frame_size=(1920, 1080),
    )
    z = _anchors(plates)[0]
    assert z[2] == 1919              # extended to fw - 1


def test_ghost_frozen_for_close_moto():
    # Close moto track dies with velocity recorded: the ghost must NOT drift
    # (velocity extrapolation is frozen for close-ups).
    vt = bp.VehicleTrack((900, 100, 1500, 900), 0, cls=3)   # 800px tall → close
    vt.update_box((950, 100, 1550, 900), 1)                 # +50px/frame velocity
    vt.id = 1
    vt.frames_seen = 1
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_ghost_frames=6,
                         moto_anchor_frac=0.30, moto_anchor_y=0.60,
                         moto_anchor_pad=0.05)
    tr._track_dict[1] = vt
    tr.update([], [], frame_size=(1920, 1080))   # frame 2: zone recorded
    tr.update([], [], frame_size=(1920, 1080))   # frame 3: missed
    tr.update([], [], frame_size=(1920, 1080))   # frame 4: missed → track dies, ghost saved
    p5, _ = tr.update([], [], frame_size=(1920, 1080))   # ghost frame 1
    p6, _ = tr.update([], [], frame_size=(1920, 1080))   # ghost frame 2
    a5, a6 = _anchors(p5)[0], _anchors(p6)[0]
    assert (a5[0] + a5[2]) == (a6[0] + a6[2])           # no drift
    assert (a5[1] + a5[3]) == (a6[1] + a6[3])


# ─── Near-moto zone geometry (plate below box centre, min zone size) ─────────

def test_near_moto_zone_slides_down():
    # A near moto's YOLO box includes the rider's head → the plate is below
    # the box centre, so the zone centre must sit lower than moto_anchor_y.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_anchor_frac=0.30,
                         moto_anchor_y=0.50, moto_anchor_pad=0.05)
    plates, _ = tr.update(
        [(3, 900, 100, 1500, 900, 0.9)], [],   # 800px on 1080p → near+close
        frame_size=(1920, 1080),
    )
    z = _anchors(plates)[0]
    cy = (z[1] + z[3]) * 0.5
    assert cy > 100 + 0.50 * 800            # below the box centre line
    assert cy < 100 + 0.72 * 800 + 5        # within the y_max bound


def test_near_moto_zone_min_side_floor():
    # A small fragmented close-up box must never produce a tiny zone: the
    # floor (moto_zone_min_side) keeps the blur at plate size.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_anchor_frac=0.13,
                         moto_anchor_y=0.50, moto_anchor_pad=0.02)
    plates, _ = tr.update(
        [(3, 1500, 400, 1700, 600, 0.9)], [],   # 200px box → 26px zone without floor
        frame_size=(1920, 1080),
    )
    z = _anchors(plates)[0]
    assert (z[2] - z[0]) >= 40
    assert (z[3] - z[1]) >= 40


def test_reborn_track_seeds_zone_from_ghost():
    # When a dead moto track is re-acquired, the new track must start from the
    # ghost zone (no jump/blink of the blur position).
    vt = bp.VehicleTrack((900, 100, 1500, 700), 0, cls=3)
    vt.update_box((950, 100, 1550, 700), 1)
    vt.id = 1
    vt.frames_seen = 1
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=True, moto_ghost_frames=6,
                         moto_anchor_frac=0.30, moto_anchor_y=0.60,
                         moto_anchor_pad=0.05)
    tr._track_dict[1] = vt
    tr.update([], [], frame_size=(1920, 1080))   # zone recorded
    tr.update([], [], frame_size=(1920, 1080))   # missed
    tr.update([], [], frame_size=(1920, 1080))   # missed → track dies, ghost saved
    ghost_zone = list(tr._ghost_zones.values())[0][0]
    plates, _ = tr.update([(3, 1000, 100, 1600, 700, 0.9)], [],
                          frame_size=(1920, 1080))   # moto re-acquired
    new_track = list(tr._track_dict.values())[0]
    assert new_track.zone_ema is not None
    # zone EMA seeded from the ghost → the emitted zone matches the ghost zone
    z = _anchors(plates)[0]
    gz = ghost_zone
    assert abs((z[0] + z[2]) * 0.5 - gz[0]) <= 5
    assert abs((z[1] + z[3]) * 0.5 - gz[1]) <= 5


# ─── Plate-confirmed zone upgrade (moto anchor + plate detection) ─────────────

def _moto_tracker(**kw):
    base = dict(max_gap_frames=2, fallback_min_frames=1,
                moto_anchor=True, moto_anchor_frac=0.30,
                moto_anchor_y=0.60, moto_anchor_pad=0.05,
                moto_plate_conf=0.30, moto_plate_promote_frames=2)
    base.update(kw)
    return bp.SceneTracker(**base)


def test_moto_plate_promotes_zone_after_confirmed_frames():
    # A plate detected inside the moto for 2 consecutive frames at conf >=
    # moto_plate_conf replaces the geometric anchor zone with the plate zone.
    tr = _moto_tracker()
    tr.update([(3, 900, 100, 1500, 700, 0.9)], [(1100, 300, 1300, 400, 0.8)],
              frame_size=(1920, 1080))   # frame 1: streak 1, no promotion yet
    plates, _ = tr.update([(3, 900, 100, 1500, 700, 0.9)],
                          [(1100, 300, 1300, 400, 0.8)],
                          frame_size=(1920, 1080))   # frame 2: promotion kicks in
    anchors = _anchors(plates)
    assert len(anchors) == 1
    z = anchors[0][:4]
    # zone centred on the plate (1200, 350), not on the geometric anchor
    assert abs((z[0] + z[2]) * 0.5 - 1200) <= 25
    assert abs((z[1] + z[3]) * 0.5 - 350) <= 25
    assert (z[2] - z[0]) < 300   # much smaller than the geometric zone


def test_moto_plate_single_frame_emits_immediately():
    # One isolated plate detection must emit the plate zone IMMEDIATELY —
    # privacy-first: the plate is covered from the first frame the model
    # sees it. The EMA smoothing absorbs single-frame jitter.
    tr = _moto_tracker()
    plates, _ = tr.update([(3, 900, 100, 1500, 700, 0.9)],
                          [(1100, 300, 1300, 400, 0.8)],
                          frame_size=(1920, 1080))
    anchors = _anchors(plates)
    assert len(anchors) == 1
    z = anchors[0][:4]
    # zone centred on the plate (1200, 350), not on the geometric anchor (y=460)
    assert abs((z[0] + z[2]) * 0.5 - 1200) <= 25
    assert abs((z[1] + z[3]) * 0.5 - 350) <= 25
    assert (z[2] - z[0]) < 300   # plate zone, not the geometric zone


def test_moto_plate_weak_conf_never_promotes():
    # conf below moto_plate_conf must never promote, even over many frames.
    tr = _moto_tracker()
    for _ in range(4):
        plates, _ = tr.update([(3, 900, 100, 1500, 700, 0.9)],
                              [(1100, 300, 1300, 400, 0.2)],
                              frame_size=(1920, 1080))
    anchors = _anchors(plates)
    z = anchors[0][:4]
    # still the geometric zone: centred low on the box (near moto → y≈526),
    # widened, not pulled up to the plate
    assert abs((z[1] + z[3]) * 0.5 - 526) <= 40
    assert (z[2] - z[0]) > 280   # widened geometric zone, not the 200px plate


def test_moto_plate_helper_holds_then_decays_without_detection():
    # After emission, the plate zone is HELD (extrapolated) while the plate is
    # missing; only once the hold window (moto_plate_hold_frames) expires does
    # the zone fall back toward the geometric anchor (still guaranteed).
    tr = _moto_tracker(moto_plate_hold_frames=5)
    tr.update([(3, 900, 100, 1500, 700, 0.9)], [(1100, 300, 1300, 400, 0.8)],
              frame_size=(1920, 1080))   # detection: plate zone emitted
    for _ in range(3):
        plates, _ = tr.update([(3, 900, 100, 1500, 700, 0.9)], [],
                              frame_size=(1920, 1080))   # plate gone, hold active
    anchors = _anchors(plates)
    z = anchors[0][:4]
    cy = (z[1] + z[3]) * 0.5
    # still held on the plate position (y=350), not decayed to anchor (y≈526)
    assert abs(cy - 350) < abs(cy - 526)
    for _ in range(8):
        plates, _ = tr.update([(3, 900, 100, 1500, 700, 0.9)], [],
                              frame_size=(1920, 1080))   # hold expired
    anchors = _anchors(plates)
    z = anchors[0][:4]
    cy = (z[1] + z[3]) * 0.5
    # decayed: moving back toward the geometric anchor (y≈526), away from the
    # promoted plate position (y=350)
    assert abs(cy - 526) < abs(cy - 350)


def _synthetic_rotated_plate():
    """1920x1080 dark frame with a bright rotated plate (white border, dark
    interior) centred near (960, 540)."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    plate = np.full((140, 140, 3), 200, dtype=np.uint8)      # bright plate
    plate[25:115, 25:115] = 30                                # dark interior
    h, w = plate.shape[:2]
    canvas = 220                                            # room for the 25° tilt
    canvas_img = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    c0 = (canvas - w) // 2                                  # centre the plate
    canvas_img[c0:c0 + h, c0:c0 + w] = plate
    M = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), 25.0, 1.0)
    rotated = cv2.warpAffine(canvas_img, M, (canvas, canvas))
    x0, y0 = 960 - canvas // 2, 540 - canvas // 2
    frame[y0:y0 + canvas, x0:x0 + canvas] = rotated
    return frame


def test_estimate_plate_quad_finds_rotated_plate():
    frame = _synthetic_rotated_plate()
    # zone generously around the plate (matches the geometric anchor zone)
    quad = bp.estimate_plate_quad(frame, (700, 300, 1220, 780))
    assert quad is not None
    assert quad.shape == (4, 2)
    # quad must sit on the plate: tight bbox close to the 140px plate
    bw = quad[:, 0].max() - quad[:, 0].min()
    bh = quad[:, 1].max() - quad[:, 1].min()
    assert 100 <= bw <= 210
    assert 100 <= bh <= 210
    # centre close to the plate centre (960, 540)
    cx = quad[:, 0].mean()
    cy = quad[:, 1].mean()
    assert abs(cx - 960) <= 35
    assert abs(cy - 540) <= 35
    # rotated: bounding box must be wider than the plate's own width (25° tilt)
    assert bw > 150


def test_estimate_plate_quad_returns_none_on_blank():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert bp.estimate_plate_quad(frame, (700, 300, 1220, 780)) is None
    # also on a tiny zone (< 16px side)
    assert bp.estimate_plate_quad(frame, (0, 0, 10, 10)) is None


def test_apply_blur_rotated_blurs_only_inside_quad():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:] = (120, 120, 120)                       # uniform grey everywhere
    quad = np.array([[70, 70], [130, 70], [130, 130], [70, 130]], dtype=np.int32)
    out = bp.apply_blur_rotated(frame.copy(), [quad], blur_strength=61, padding=0)
    # uniform grey inside → blur is a no-op visually; use a textured probe:
    frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
    frame2[:] = (10, 10, 10)
    frame2[80, 100] = (250, 250, 250)                # single bright pixel inside quad
    out2 = bp.apply_blur_rotated(frame2.copy(), [quad], blur_strength=61, padding=0)
    assert out2[80, 100].max() < 250                 # smoothed inside
    assert out2[10, 10].tolist() == [10, 10, 10]     # untouched outside
    assert out[80, 100].tolist() == [120, 120, 120]  # uniform input → uniform output


def test_apply_blur_rotated_respects_frame_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (10, 10, 10)
    # quad protruding beyond the frame on all sides
    quad = np.array([[-20, -20], [120, -20], [120, 120], [-20, 120]], dtype=np.int32)
    out = bp.apply_blur_rotated(frame.copy(), [quad], blur_strength=61, padding=8)
    assert out.shape == frame.shape
    assert out[50, 50].tolist() == [10, 10, 10]      # uniform → no visual change


def test_align_quad_reorders_rotated_permutation():
    # Same geometry, but minAreaRect may hand the corners in a rotated/
    # flipped order between frames; _align_quad must re-order to match ref.
    ref  = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
    new  = np.array([[100, 50], [0, 50], [0, 0], [100, 0]], dtype=np.float64)
    out  = bp._align_quad(ref, new)
    # pointwise order must match ref (small distance to ref's ordering)
    assert np.max(np.abs(out - ref)) < 1e-9


def test_match_quad_state_finds_nearby_rect_and_rejects_far_one():
    hist = [{"rect": (100, 100, 200, 200), "quad": None, "ratio": 0.9,
             "active": True}]
    # same zone → match
    assert bp._match_quad_state(hist, (105, 105, 195, 195)) == 0
    # far away → no match
    assert bp._match_quad_state(hist, (800, 800, 900, 900)) == -1


def _sq(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def test_quad_state_update_holds_and_glides_when_quad_lost():
    # Active quad covering a 200x200 zone; the zone then moves +20 px in x.
    prev_quad = _sq(100, 100, 200, 200)
    st = {"rect": (0, 0, 200, 200), "quad": prev_quad.copy(),
          "ratio": 1.0, "active": True}
    qb, use = bp._quad_state_update(st, (20, 0, 220, 200), None, 200 * 200)
    assert use is True
    # Held quad must glide with the zone instead of freezing in place
    # (that was the "blur slides off the plate in lean/curve" defect).
    assert np.allclose(qb[:, 0] - prev_quad[:, 0], 20.0)
    assert np.allclose(qb[:, 1] - prev_quad[:, 1], 0.0)
    assert st["rect"] == (20, 0, 220, 200)
    assert st["active"] is True


def test_quad_state_update_weak_ratio_holds_instead_of_snapping_to_rect():
    # Quad was active at full coverage; a weak estimate (covers 4% of the
    # zone) must NOT drop to the oversized anchor rect — the held quad stays.
    prev_quad = _sq(0, 0, 100, 100)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 1.0, "active": True}
    weak = _sq(40, 40, 60, 60)
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100)
    assert use is True
    assert np.allclose(qb, prev_quad)


def test_quad_state_update_rejects_top_case_jump():
    # A horizontal shape above the plate (top case / bauletto inside the
    # widened zone) must NOT replace the plate quad — it is neither close
    # to the held quad nor near the zone centre, and its coverage is low.
    prev_quad = _sq(0, 25, 100, 75)   # plate at the zone centre
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.5, "active": True}
    top_case = _sq(0, 0, 100, 25)     # strip above the centre
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), top_case, 100 * 100)
    assert use is True
    assert np.allclose(qb, prev_quad)   # lock did not move to the top case


def test_quad_state_update_top_case_corrected_by_plate():
    # Asymmetric gate: even if the current lock is wrong (top case), a
    # plate-like estimate at the zone centre still replaces it.
    prev_quad = _sq(0, 0, 100, 25)     # locked on the top case
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.25, "active": True}
    plate = _sq(0, 25, 100, 75)        # real plate at the centre
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), plate, 100 * 100)
    assert use is True
    # alpha 0.8 relocation blend: 0.2*top + 0.8*plate → centre y = 42.5
    assert np.allclose(qb.mean(axis=0), [50.0, 42.5], atol=1.0)
    assert st["ratio"] >= 0.40


def test_quad_state_update_minimum_active_duration_and_fallback():
    # Active quad at 40% coverage; a weak-but-blendable estimate (30%) must
    # survive `min_active` frames as a held glide before the smooth fallback
    # expansion toward the anchor-sized rect, and only then give up to it.
    prev_quad = _sq(0, 30, 100, 70)
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.4, "active": True}
    weak = _sq(0, 35, 100, 65)
    for _ in range(4):                 # frames 1-4: held, shape unchanged
        qb, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                        min_active=5)
        assert use is True
        assert np.allclose(qb, prev_quad)
    _, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                   min_active=5)
    assert use is True                 # frame 5: still held
    assert np.allclose(st["quad"], prev_quad)
    qb, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                    min_active=5)
    assert use is True                 # frame 6: fallback expansion starts
    assert float(qb[:, 1].max() - qb[:, 1].min()) > 40.0
    for _ in range(4):
        _, use = bp._quad_state_update(st, (0, 0, 100, 100), weak, 100 * 100,
                                       min_active=5)
    assert use is False                # quad given up → rect takes over


def test_quad_state_update_enter_threshold_when_inactive():
    prev_quad = _sq(0, 0, 100, 100)
    # Inactive state: a mid-coverage estimate (ratio 0.5) must not activate.
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 0.5, "active": False}
    _, use = bp._quad_state_update(st, (0, 0, 100, 100),
                                   _sq(25, 25, 75, 75), 100 * 100)
    assert use is False
    # Full-coverage estimate activates.
    _, use = bp._quad_state_update(st, (0, 0, 100, 100),
                                   _sq(0, 0, 100, 100), 100 * 100)
    assert use is True


def test_quad_state_update_faster_ema_while_zone_moves():
    prev_quad = _sq(0, 0, 100, 100)
    # Zone shifts +20 px in x, quad estimate follows: the blend must weight
    # the new estimate more (alpha 0.65) so the blur does not lag a lean.
    st = {"rect": (0, 0, 100, 100), "quad": prev_quad.copy(),
          "ratio": 1.0, "active": True}
    new_quad = _sq(20, 0, 120, 100)
    qb, use = bp._quad_state_update(st, (20, 0, 120, 100), new_quad, 100 * 100)
    assert use is True
    # Expected: 0.35 * 0 + 0.65 * 20 = 13 px shift (0.5 blend would give 10).
    assert np.allclose(qb[:, 0] - prev_quad[:, 0], 13.0)
    assert np.allclose(qb[:, 1] - prev_quad[:, 1], 0.0)


# ─── Emission displacement clamp (emit_max_disp) ─────────────────────────────
# A plate confirmed at (x, y) cannot legitimately jump to (x+500, y+500) a few
# frames later: every zone emitted for a track is clamped so the blur centre
# travels at most emit_max_disp px/frame from the last emitted zone.

def _emit_tracker(**kw):
    base = dict(max_gap_frames=10, fallback_min_frames=1,
                moto_anchor=True, moto_anchor_frac=0.30,
                moto_anchor_y=0.60, moto_anchor_pad=0.05)
    base.update(kw)
    tr = bp.SceneTracker(**base)
    vt = bp.VehicleTrack((100, 100, 200, 160), 0, cls=3)
    vt.id = 1
    vt.frames_seen = 1
    tr._track_dict[1] = vt
    return tr, vt


def test_emit_clamp_first_emission_is_free():
    # No reference zone yet — the first emission of a track must pass as-is
    # (a newly visible vehicle's plate may legitimately be anywhere).
    tr, _ = _emit_tracker(emit_max_disp=80.0)
    plates, _ = tr.update([], [])
    z = [p for p in plates if len(p) > 5 and p[5] == "anchor"][0][:4]
    assert z == _emit_expected_zone(tr)


def _emit_expected_zone(tr):
    # anchor_rect for box (100,100,200,160): frac 0.30, y 0.60, pad 0.05
    side = max(0.30 * 60, tr.moto_zone_min_side, 0.45 * 60)
    cy = 100 + 60 * 0.60
    pad = side * 0.05
    return (max(0, int(150 - side * 0.5 - pad)),
            max(0, int(cy - side * 0.5 - pad)),
            int(150 + side * 0.5 + pad),
            int(cy + side * 0.5 + pad))


def test_emit_clamp_blocks_vehicle_teleport():
    # The vehicle box teleports 600 px right (detector glitch / runaway box):
    # the blur must NOT follow it — it stays within emit_max_disp of the last
    # emitted zone.
    tr, vt = _emit_tracker(emit_max_disp=80.0)
    plates, _ = tr.update([], [])                  # frame 0: zone emitted free
    first = [p for p in plates if len(p) > 5 and p[5] == "anchor"][0][:4]
    fc = (first[0] + first[2]) * 0.5
    vt.box = (700, 100, 800, 160)                  # teleport +600 px in x
    plates, _ = tr.update([], [])
    z = [p for p in plates if len(p) > 5 and p[5] == "anchor"][0][:4]
    nc = (z[0] + z[2]) * 0.5
    assert abs(nc - fc) <= 80 + 1                  # travelled at most 80 px
    assert nc < 700                                # never reached the target


def test_emit_clamp_allows_slow_motion():
    tr, vt = _emit_tracker(emit_max_disp=80.0)
    tr.update([], [])
    vt.box = (120, 100, 220, 160)                  # +20 px/frame: plausible
    plates, _ = tr.update([], [])
    z = [p for p in plates if len(p) > 5 and p[5] == "anchor"][0][:4]
    assert abs((z[0] + z[2]) * 0.5 - 170) <= 25    # zone simply follows


def test_emit_clamp_glides_toward_target():
    # Repeated over-eager predictions converge to the target at the capped
    # speed instead of teleporting in one frame.
    tr, vt = _emit_tracker(emit_max_disp=80.0)
    tr.update([], [])
    vt.box = (700, 100, 800, 160)
    seen = []
    for _ in range(3):
        plates, _ = tr.update([], [])
        z = [p for p in plates if len(p) > 5 and p[5] == "anchor"][0][:4]
        seen.append((z[0] + z[2]) * 0.5)
    # each frame the blur makes progress toward the target but stays bounded
    assert seen[1] - seen[0] <= 80 + 1
    assert seen[2] - seen[1] <= 80 + 1
    assert seen[2] < 700


def test_emit_clamp_limits_far_detection_on_car():
    # A far in-box detection jumping 480 px in one frame must be clamped:
    # it is detection jitter / false positive, not plate motion.
    tr = bp.SceneTracker(max_gap_frames=2, fallback_min_frames=1,
                         moto_anchor=False, emit_max_disp=80.0)
    vt = bp.VehicleTrack((0, 0, 600, 160), 0, cls=2)
    vt.id = 1
    vt.frames_seen = 1
    tr._track_dict[1] = vt
    plates, _ = tr.update([], [(20, 20, 80, 60, 0.9, "sahi")])    # free
    plates, _ = tr.update([], [(500, 20, 560, 60, 0.9, "sahi")])  # 480px jump
    sahi = [p for p in plates if len(p) > 5 and p[5] == "sahi"]
    assert len(sahi) == 1
    nc = (sahi[0][0] + sahi[0][2]) * 0.5
    assert abs(nc - 50) <= 80 + 1


def test_emit_clamp_blocks_pred_racing_from_runaway_box():
    # Repro of the 27-29 s montage case: a plate confirmed on a car, then the
    # runaway vehicle box drags the anchored prediction across the frame —
    # the emitted pred zone must stay within the cap of the last emission.
    tr = bp.SceneTracker(max_gap_frames=10, fallback_min_frames=1,
                         moto_anchor=False, emit_max_disp=80.0)
    vt = bp.VehicleTrack((800, 630, 1200, 850), 0, cls=2)
    vt.id = 1
    vt.frames_seen = 1
    tr._track_dict[1] = vt
    tr.update([], [(860, 764, 902, 790, 0.9, "sahi")])   # plate confirmed
    vt.box = (700, 630, 1100, 850)                       # box slides left
    plates, _ = tr.update([], [])                        # plate missed → pred
    preds = [p for p in plates if len(p) > 5 and p[5] == "pred"]
    assert len(preds) == 1
    nc = (preds[0][0] + preds[0][2]) * 0.5
    assert abs(nc - 881) <= 81


# ─── BoxFilter (constant-velocity Kalman) ─────────────────────────────────────

def _moving_boxes(n, v=10.0, x0=400.0, y0=500.0, w=60.0, h=40.0):
    """n boxes travelling at v px/frame along x (exact, noise-free)."""
    out = []
    for i in range(n):
        cx, cy = x0 + v * i, y0
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def test_boxfilter_tracks_constant_velocity_without_lag():
    boxes = _moving_boxes(10, v=10.0)
    f = bp.BoxFilter(boxes[0])
    for b in boxes[1:]:
        f.predict()
        f.update(b)
    truth_cx = (boxes[-1][0] + boxes[-1][2]) * 0.5
    assert abs(f.centre[0] - truth_cx) < 1.5
    assert abs(f.velocity[0] - 10.0) < 1.0


def test_boxfilter_beats_ema_cascade_on_moving_target():
    # The two chained EMAs settle ~1.33 * v behind a constant-velocity target;
    # the filter estimates the velocity instead, so it stays on the plate.
    boxes = _moving_boxes(12, v=10.0)
    f = bp.BoxFilter(boxes[0])
    track = bp.VehicleTrack(boxes[0], 0)
    for b in boxes[1:]:
        f.predict()
        f.update(b)
        cx, cy, w, h = track.ema_update(b, 0.6)
        track.zone_ema_update((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 0.6)
    truth_cx = (boxes[-1][0] + boxes[-1][2]) * 0.5
    kalman_err = abs(f.centre[0] - truth_cx)
    ema_err    = abs(track.zone_ema[0] - truth_cx)
    assert kalman_err < 2.0
    assert ema_err > 8.0
    assert kalman_err < ema_err / 4


def test_boxfilter_predicts_through_gap():
    boxes = _moving_boxes(8, v=12.0)
    f = bp.BoxFilter(boxes[0])
    for b in boxes[1:]:
        f.predict()
        f.update(b)
    cx_before = f.centre[0]
    for _ in range(3):
        f.predict()
    assert abs((f.centre[0] - cx_before) - 36.0) < 4.0
    assert f.since_update == 3


def test_boxfilter_low_confidence_moves_estimate_less():
    boxes = _moving_boxes(6, v=0.0)
    strong = bp.BoxFilter(boxes[0])
    weak   = bp.BoxFilter(boxes[0])
    for b in boxes[1:]:
        for f in (strong, weak):
            f.predict()
        strong.update(b)
        weak.update(b)
    jump = (500, 480, 560, 520)      # same box shifted ~100 px right
    strong.predict(); strong.update(jump, conf=0.95)
    weak.predict();   weak.update(jump, conf=0.15)
    base = (boxes[0][0] + boxes[0][2]) * 0.5
    assert abs(strong.centre[0] - base) > abs(weak.centre[0] - base)


def test_boxfilter_uncertainty_grows_during_gap():
    f = bp.BoxFilter((100, 100, 160, 140))
    f.predict(); f.update((110, 100, 170, 140))
    sigmas = []
    for _ in range(5):
        f.predict()
        sigmas.append(f.position_sigma())
    assert sigmas == sorted(sigmas)
    assert sigmas[-1] > sigmas[0]


def test_boxfilter_gate_rejects_teleport_accepts_plausible():
    boxes = _moving_boxes(6, v=8.0)
    f = bp.BoxFilter(boxes[0])
    for b in boxes[1:]:
        f.predict()
        f.update(b)
    f.predict()
    cx = f.centre[0]
    near = (cx - 25, 480, cx + 35, 520)          # a few px off the prediction
    far  = (cx + 470, 480, cx + 530, 520)        # 500 px teleport
    assert f.gate_distance(near) < f.gate_distance(far)
    assert f.gate_distance(far) > 5.0


def test_boxfilter_size_stays_positive_through_long_gap():
    # A shrinking plate (vehicle receding) must never produce a negative box.
    f = bp.BoxFilter((100, 100, 200, 180))
    for w in (90, 80, 70, 60, 50):
        f.predict()
        f.update((100, 100, 100 + w, 100 + int(w * 0.8)))
    for _ in range(40):
        f.predict()
    x1, y1, x2, y2 = f.box()
    assert x2 > x1 and y2 > y1


# ─── SceneTracker with zone_filter="kalman" ───────────────────────────────────

def _kalman_tracker(**kw):
    kw.setdefault("moto_anchor", False)
    kw.setdefault("fallback_min_frames", 1)
    return bp.SceneTracker(zone_filter="kalman", **kw)


def _car_track(tr, box, tid=1):
    vt = bp.VehicleTrack(box, 0, cls=2)
    vt.id = tid
    vt.frames_seen = 1
    tr._track_dict[tid] = vt
    return vt


def test_kalman_mode_is_off_by_default():
    assert bp.SceneTracker().zone_filter == "ema"


def test_kalman_zone_covers_the_detection_it_was_fed():
    tr = _kalman_tracker()
    _car_track(tr, (800, 600, 1200, 900))
    plate = (900, 780, 960, 810)
    zones, _ = tr.update([], [(*plate, 0.9, "sahi")])
    assert zones
    assert max(bp.covered_fraction(plate, z[:4]) for z in zones) >= 0.9


def test_kalman_gated_detection_still_gets_a_patch_zone():
    # Privacy first: a detection the filter refuses to follow (implausible jump)
    # must still be blurred, as a separate 'patch' zone.
    tr = _kalman_tracker(kf_gate_max=1.0)
    _car_track(tr, (800, 600, 1200, 900))
    for _ in range(3):
        tr.update([], [(900, 780, 960, 810, 0.9, "sahi")])
    jump = (1100, 780, 1160, 810)
    zones, _ = tr.update([], [(*jump, 0.9, "sahi")])
    assert any(z[5] == "patch" for z in zones)
    assert max(bp.covered_fraction(jump, z[:4]) for z in zones) >= 0.9


def test_kalman_reseeds_after_repeated_rejections():
    tr = _kalman_tracker(kf_gate_max=1.0, kf_max_rejects=2)
    vt = _car_track(tr, (800, 600, 1200, 900))
    for _ in range(3):
        tr.update([], [(900, 780, 960, 810, 0.9, "sahi")])
    for _ in range(2):
        tr.update([], [(1100, 780, 1160, 810, 0.9, "sahi")])
    # After the second rejection the filter is re-seeded on the new position,
    # so it no longer coasts near the abandoned one.
    assert abs(vt.kf.centre[0] - 1130) < 40


def test_kalman_padding_grows_at_once_and_retreats_slowly():
    tr = _kalman_tracker(kf_sigma_pad_k=1.0, kf_sigma_pad_max=40.0,
                         kf_pad_decay=1.5)
    vt = _car_track(tr, (800, 600, 1200, 900))
    for _ in range(4):
        tr.update([], [(900, 780, 960, 810, 0.9, "sahi")])
    settled = vt.kf_pad
    for _ in range(4):                       # plate missing → uncertainty grows
        tr.update([], [])
    grown = vt.kf_pad
    assert grown > settled
    tr.update([], [(900, 780, 960, 810, 0.9, "sahi")])
    assert vt.kf_pad >= grown - 1.5 - 1e-6   # retreat is rate-limited


def test_kalman_gap_zone_stays_inside_the_vehicle_box():
    # A long coast must not wander onto roadside objects.
    tr = _kalman_tracker(max_gap_frames=20)
    vt = _car_track(tr, (800, 600, 1200, 900))
    for cx in (900, 940, 980):
        tr.update([], [(cx, 780, cx + 60, 810, 0.9, "sahi")])
    for _ in range(12):
        zones, _ = tr.update([], [])
    preds = [z for z in zones if z[5] == "pred"]
    assert preds
    vx1, vy1, vx2, vy2 = vt.box
    px = (vx2 - vx1) * 0.10
    for z in preds:
        assert z[0] >= vx1 - px - 1 and z[2] <= vx2 + px + 1


def test_kalman_gap_fill_follows_the_plate_better_than_ema():
    # On frames where the plate IS detected both modes emit the detection, so
    # the difference shows up during gaps. The old path applies the last plate
    # offset to the vehicle box, which does not move when the vehicle box is
    # steady, so the zone stays behind; the filter keeps the plate's own
    # velocity and coasts with it.
    v, gap = 12, 4
    boxes = [(900 + v * i, 780, 960 + v * i, 810) for i in range(6)]
    truth = 930 + v * (len(boxes) - 1 + gap)   # centre if motion continues
    results = {}
    for mode in ("ema", "kalman"):
        # max_gap_frames high enough that the pre-seeded track survives frames
        # without vehicle detections being fed to ByteTrack.
        tr = bp.SceneTracker(zone_filter=mode, moto_anchor=False,
                             fallback_min_frames=1, max_gap_frames=30)
        _car_track(tr, (700, 600, 1500, 900))
        for b in boxes:
            tr.update([], [(*b, 0.9, "sahi")])
        for _ in range(gap):
            zones, _ = tr.update([], [])
        preds = [z for z in zones if z[5] == "pred"]
        assert preds, f"{mode} emitted no gap-fill zone"
        centre = (preds[0][0] + preds[0][2]) * 0.5
        results[mode] = abs(centre - truth)
    assert results["kalman"] < results["ema"] / 2


def test_boxfilter_noise_scale_moves_estimate_less():
    # A geometric (weak) update with noise_scale=5 must nudge far less than a
    # full-confidence detection of the same jump.
    base = (100, 100, 160, 140)
    jump = (200, 100, 260, 140)
    strong = bp.BoxFilter(base)
    weak   = bp.BoxFilter(base)
    strong.predict(); strong.update(jump, conf=1.0, noise_scale=1.0)
    weak.predict();   weak.update(jump, conf=1.0, noise_scale=5.0)
    base_cx = 130.0
    assert abs(strong.centre[0] - base_cx) > abs(weak.centre[0] - base_cx)


def test_kalman_moto_anchor_is_weak_measurement_not_size_floor():
    # Without a plate detection the Kalman moto path seeds from the geometric
    # anchor. The emitted zone must stay near plate size (≈ frac × box height),
    # not the old 0.45×box-height floor that made zones ~2× too wide.
    tr = bp.SceneTracker(
        zone_filter="kalman", moto_anchor=True,
        moto_anchor_frac=0.20, moto_anchor_y=0.50, moto_anchor_pad=0.05,
        moto_zone_min_side=40.0, fallback_min_frames=1,
        moto_near_frac=0.40,   # keep this box out of the near-widen path
        kf_anchor_meas_scale=5.0, moto_plate_pad=4,
    )
    # 90 px tall → well below near threshold; plate-sized side ≈ 18 px.
    box = (1200, 550, 1290, 640)
    zones = None
    for _ in range(3):
        zones, _ = tr.update([(3, *box, 0.7)], [], None, frame_size=(1920, 1080))
    assert zones, "expected an anchor zone from the weak geometric measurement"
    z = max(zones, key=lambda zz: (zz[2] - zz[0]) * (zz[3] - zz[1]))
    side_w = z[2] - z[0]
    # Old path blew width up to 0.45×box height (~40 px here) as a square side.
    # The new path may be taller (fender coverage) but must stay narrow enough
    # that it is not the old square blow-up.
    assert side_w < 70, f"zone width {side_w} still near the old 0.45×height floor"
    assert side_w > 12, f"zone width {side_w} collapsed below a readable plate"


def test_kalman_moto_real_plate_overrides_anchor():
    # A confident in-box plate detection must pull the filter onto the plate,
    # not leave it sitting on the geometric centre.
    from plates.common import covered_fraction
    tr = bp.SceneTracker(
        zone_filter="kalman", moto_anchor=True,
        moto_anchor_frac=0.20, moto_anchor_y=0.50, moto_anchor_pad=0.05,
        moto_plate_conf=0.30, fallback_min_frames=1,
        moto_near_frac=0.40, kf_anchor_meas_scale=5.0,
    )
    box = (1000, 400, 1200, 700)
    plate = (1140, 560, 1170, 590, 0.80, "crop")  # right of geometric centre
    for _ in range(2):
        tr.update([(3, *box, 0.7)], [], None, frame_size=(1920, 1080))
    zones, _ = tr.update([(3, *box, 0.7)], [plate], None,
                         frame_size=(1920, 1080))
    assert zones
    z = max(zones, key=lambda zz: covered_fraction(plate[:4], zz[:4]))
    zcx = (z[0] + z[2]) * 0.5
    plate_cx = 1155.0
    geo_cx = 1100.0
    assert abs(zcx - plate_cx) < abs(zcx - geo_cx)


def test_plate_geometry_gate_rejects_wheel_and_backpack():
    tr = bp.SceneTracker(zone_filter="kalman", moto_anchor=True)
    box = (1000, 400, 1200, 700)   # 200×300
    # Mid-rear plate: should pass.
    assert tr._plate_geometry_ok((1080, 520, 1120, 555), box)
    # Wheel / bottom of box: reject.
    assert not tr._plate_geometry_ok((1080, 650, 1120, 690), box)
    # Rider backpack / top of box: reject.
    assert not tr._plate_geometry_ok((1140, 420, 1170, 445), box)
    # Low fender plate on a tall adventure bike: must still be accepted.
    assert tr._plate_geometry_ok((1080, 595, 1125, 630), box)
    # Tiny noise: reject.
    assert not tr._plate_geometry_ok((1100, 530, 1105, 536), box)


def test_kalman_moto_learns_relative_pose_and_ignores_outlier():
    from plates.common import covered_fraction
    tr = bp.SceneTracker(
        zone_filter="kalman", moto_anchor=True,
        moto_anchor_frac=0.20, moto_anchor_y=0.38, moto_anchor_pad=0.05,
        moto_plate_conf=0.15, fallback_min_frames=1, moto_near_frac=0.40,
        kf_anchor_meas_scale=5.0, moto_plate_hold_frames=20,
    )
    box = (1000, 400, 1200, 700)
    good = (1080, 520, 1120, 555, 0.70, "crop")   # mid-rear
    bad  = (1140, 420, 1170, 445, 0.40, "crop")   # backpack / top
    for _ in range(3):
        tr.update([(3, *box, 0.7)], [good], None, frame_size=(1920, 1080))
    # After learning, an outlier must not yank the zone; it becomes a patch.
    zones, _ = tr.update([(3, *box, 0.7)], [bad], None, frame_size=(1920, 1080))
    assert zones
    main = [z for z in zones if z[5] != "patch"]
    patches = [z for z in zones if z[5] == "patch"]
    assert main and patches
    assert covered_fraction(good[:4], main[0][:4]) >= 0.5
    assert covered_fraction(bad[:4], main[0][:4]) < 0.5


# ─── Detection letterbox / moto ROI / geometry filter ─────────────────────────

def test_letterbox_preserves_aspect_and_roundtrips_box():
    from plates.detect import letterbox_to_square, unletterbox_xyxy
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[40:60, 80:120] = 255
    canvas, scale, pad_x, pad_y = letterbox_to_square(img, size=640)
    assert canvas.shape == (640, 640, 3)
    # Content should not fill the full square (letterboxing pads the short axis)
    assert scale == min(640 / 200, 640 / 100)
    assert abs(pad_y) > 1 or abs(pad_x) > 1
    # Map a known box through letterbox space and back
    bx1, by1 = pad_x + 80 * scale, pad_y + 40 * scale
    bx2, by2 = pad_x + 120 * scale, pad_y + 60 * scale
    ox1, oy1, ox2, oy2 = unletterbox_xyxy(
        bx1, by1, bx2, by2, scale, pad_x, pad_y, 200, 100)
    assert abs(ox1 - 80) <= 2 and abs(oy1 - 40) <= 2
    assert abs(ox2 - 120) <= 2 and abs(oy2 - 60) <= 2


def test_moto_rear_roi_starts_at_bottom_frac():
    from plates.detect import moto_rear_roi
    # box 100×200 → ROI y from 0.40 * 200 = 80
    rx1, ry1, rx2, ry2 = moto_rear_roi(
        10, 20, 110, 220, bottom_frac=0.40, side_pad_frac=0.05)
    assert ry1 == 20 + 80
    assert ry2 == 220
    assert rx1 == 10 - 5   # 5% of width 100
    assert rx2 == 110 + 5


def test_plate_in_moto_geometry_accepts_mid_rear_rejects_top():
    from plates.detect import plate_in_moto_geometry_ok, filter_moto_plate_geometry
    box = (100, 100, 200, 300)  # 100×200
    mid = (130, 200, 170, 230)  # centre ry≈0.575
    top = (130, 110, 170, 140)  # centre ry≈0.125
    assert plate_in_moto_geometry_ok(mid, box)
    assert not plate_in_moto_geometry_ok(top, box)
    vehicles = [(3, *box, 0.8)]
    kept, rej = filter_moto_plate_geometry(
        [( *mid, 0.5, "crop_moto"), (*top, 0.5, "crop_moto")],
        vehicles, collect_rejected=True)
    assert len(kept) == 1 and kept[0][5] == "crop_moto"
    assert len(rej) == 1 and rej[0][5] == "rejected"


def test_moto_too_small_gate():
    tr = bp.SceneTracker(moto_min_blur_box_h_frac=0.10)
    assert tr._moto_too_small((0, 0, 50, 90), fh=1080)    # 90 < 108
    assert not tr._moto_too_small((0, 0, 200, 400), fh=1080)


def test_moto_trusted_tight_vs_weak_fender():
    tr = bp.SceneTracker(
        moto_anchor=True, moto_plate_promote_frames=2,
        moto_max_zone_box_frac=0.35, moto_weak_fender_frac=0.90,
        moto_zone_min_side=20.0,
    )
    from plates.track import VehicleTrack
    box = (1000, 400, 1200, 900)  # tall near moto 200x500
    vt = VehicleTrack(box, 0, cls=3)
    vt.plate_rel = (0.5, 0.45, 0.18, 0.12)
    vt.moto_plate_streak = 2
    vt.moto_plate_miss = 0
    assert tr._moto_trusted(vt)
    tight = tr._moto_anchor_box(vt, box, fw=1920, fh=1080)
    tw, th = tight[2] - tight[0], tight[3] - tight[1]
    # Trusted zone must stay well under half the vehicle area
    assert (tw * th) / (200 * 500) <= 0.40

    vt.moto_plate_miss = 8  # weak / coasting
    assert not tr._moto_trusted(vt)
    weak = tr._moto_anchor_box(vt, box, fw=1920, fh=1080)
    # Weak path stretches toward fender (~0.90 of box)
    assert weak[3] >= box[1] + int(500 * 0.85)


def test_cap_moto_zone_shrinks_oversized():
    tr = bp.SceneTracker(moto_max_zone_box_frac=0.25)
    box = (0, 0, 100, 200)
    huge = (-50, -50, 150, 250)  # bigger than box
    out = tr._cap_moto_zone(huge, box)
    ow, oh = out[2] - out[0], out[3] - out[1]
    assert (ow * oh) / (100 * 200) <= 0.26


# ─── run without pytest ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{sum(1 for n in globals() if n.startswith('test_') and callable(globals()[n])) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
