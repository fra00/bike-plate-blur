"""Geometric motorcycle plate zone: wide oval, centre h/2 from the box bottom.

Height defaults to one third of the green box. Width is larger (flattened
vertically / elongated horizontally). Vertical pose is always *from the
bottom* of the YOLO box (wheels), never from the top or from an energy
centroid: the helmet edge flickers, Sobel locks onto logos / knee sliders.
Lean angle still comes from the principal axis of crop energy. Size, x and
angle are EMA-smoothed per moto.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from plates.common import iou

_MOTO_CLS = 3


def _crop_box(frame, box):
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    h_img, w_img = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img, x2), min(h_img, y2)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None
    return crop, x1c, y1c, x2c, y2c


def strip_mass_xy(frame, box, y0_frac: float, y1_frac: float):
    """Frame (cx, cy) of Sobel-energy centroid in a horizontal crop strip."""
    parsed = _crop_box(frame, box)
    if parsed is None:
        return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    crop, x1c, y1c, x2c, y2c = parsed
    ch, cw = crop.shape[:2]
    if cw < 4 or ch < 4:
        return 0.5 * (x1c + x2c), 0.5 * (y1c + y2c)
    y0f = min(1.0, max(0.0, float(y0_frac)))
    y1f = min(1.0, max(y0f + 1e-3, float(y1_frac)))
    y0 = int(ch * y0f)
    y1 = int(ch * y1f)
    if y1 <= y0:
        y1 = min(ch, y0 + 1)
    strip = crop[y0:y1, :]
    if strip.size == 0:
        return 0.5 * (x1c + x2c), float(y1c + y0)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.hypot(gx, gy)
    sigma = max(1.5, cw / 20.0)
    energy = cv2.GaussianBlur(energy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    total = float(energy.sum())
    sh, sw = energy.shape
    if total < 1e-3:
        return 0.5 * (x1c + x2c), float(y1c + y0 + 0.5 * sh)
    ys, xs = np.indices(energy.shape, dtype=np.float64)
    cx_local = float((xs * energy).sum() / total)
    cy_local = float((ys * energy).sum() / total)
    return float(x1c + cx_local), float(y1c + y0 + cy_local)


def bottom_mass_cx(frame, box, strip_frac: float = 0.30) -> float:
    """Frame-x of the lower-strip edge centroid. Box centre if the crop is empty."""
    cx, _ = strip_mass_xy(frame, box, 1.0 - min(0.95, max(0.08, float(strip_frac))), 1.0)
    return cx


def _fold_ellipse_angle(deg: float) -> float:
    """Map any orientation onto [-90, 90) — 180° is the same ellipse."""
    return (float(deg) + 90.0) % 180.0 - 90.0


def _wrap_delta(prev: float, new: float) -> float:
    return (float(new) - float(prev) + 180.0) % 360.0 - 180.0


def _crop_energy(frame, box):
    parsed = _crop_box(frame, box)
    if parsed is None:
        return None
    crop, x1c, y1c, x2c, y2c = parsed
    ch, cw = crop.shape[:2]
    if cw < 4 or ch < 4:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.hypot(gx, gy)
    sigma = max(1.5, min(cw, ch) / 20.0)
    energy = cv2.GaussianBlur(energy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return energy, gray, x1c, y1c, cw, ch


def _vert_dist(deg: float) -> float:
    """Distance of an axis to vertical, in [0, 90]."""
    return abs((float(deg) % 180.0) - 90.0)


def energy_pose(frame, box, snap_angle: float = 8.0,
                rear_along_frac: float = 0.20,
                rear_y0_frac: float = 0.45,
                rear_y1_frac: float = 0.72,
                wide_ar: float = 0.95,
                wide_y0_frac: float = 0.28,
                wide_y1_frac: float = 0.58):
    """Box-bottom centre (h/2 up) and plate-major angle.

    Vertical centre is always ``by2 - h/2``. Energy is used only for the
    lean angle (principal axis, forced near-vertical so a top-case does
    not rotate the oval 90°). Unused *rear_*/ *wide_* kwargs are kept so
    older callers do not break.
    """
    del rear_along_frac, rear_y0_frac, rear_y1_frac
    del wide_ar, wide_y0_frac, wide_y1_frac
    bx1, by1, bx2, by2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    bh = by2 - by1
    cx = 0.5 * (bx1 + bx2)
    cy = by2 - 0.5 * bh
    fallback = (cx, cy, 0.0)
    parsed = _crop_energy(frame, box)
    if parsed is None:
        return fallback
    energy, _gray, _x1c, _y1c, _cw, _ch = parsed
    w = energy.astype(np.float64)
    total = float(w.sum())
    if total < 1e-3:
        return fallback
    ys, xs = np.indices(w.shape, dtype=np.float64)
    ecx = float((xs * w).sum() / total)
    ecy = float((ys * w).sum() / total)
    dx, dy = xs - ecx, ys - ecy
    mu20 = float((w * dx * dx).sum())
    mu02 = float((w * dy * dy).sum())
    mu11 = float((w * dx * dy).sum())
    bike_deg = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
    if _vert_dist(bike_deg) > _vert_dist(bike_deg + 90.0):
        bike_deg += 90.0
    disc = math.hypot(mu20 - mu02, 2.0 * mu11)
    lam1 = 0.5 * (mu20 + mu02 + disc)
    lam2 = 0.5 * (mu20 + mu02 - disc)
    if lam1 / max(lam2, 1e-6) < 1.28:
        bike_deg = 90.0
    plate_deg = _fold_ellipse_angle(bike_deg + 90.0)
    if abs(plate_deg) < float(snap_angle):
        plate_deg = 0.0
    return float(cx), float(cy), float(plate_deg)


def lean_pose(frame, box, strip_frac: float = 0.30, snap_angle: float = 8.0,
              rear_along_frac: float = 0.20):
    """Compatibility wrapper: rear point, dummy top, plate-major angle."""
    del strip_frac
    cx, cy, angle = energy_pose(
        frame, box, snap_angle=snap_angle, rear_along_frac=rear_along_frac)
    return (cx, cy), (0.5 * (box[0] + box[2]), float(box[1])), float(angle)


def ellipse_aabb(cx, cy, ax, ay, angle):
    """Axis-aligned bounding box of a rotated ellipse."""
    th = math.radians(float(angle))
    half_w = math.hypot(ax * math.cos(th), ay * math.sin(th))
    half_h = math.hypot(ax * math.sin(th), ay * math.cos(th))
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _match_prev(prev_boxes, curr_boxes, min_iou: float) -> dict[int, int]:
    pairs = []
    for i, pb in enumerate(prev_boxes):
        for j, cb in enumerate(curr_boxes):
            ii = iou(pb, cb)
            if ii >= min_iou:
                pairs.append((ii, i, j))
    pairs.sort(key=lambda t: t[0], reverse=True)
    mapping: dict[int, int] = {}
    used_p, used_c = set(), set()
    for _, i, j in pairs:
        if i in used_p or j in used_c:
            continue
        used_p.add(i)
        used_c.add(j)
        mapping[j] = i
    return mapping


def _as_sample(item):
    """Accept a bare box or (box, h, cy, cx, angle)."""
    if item and isinstance(item[0], (tuple, list)):
        box, h, cy, cx, ang = item
        return (tuple(box), float(h), float(cy), float(cx), float(ang))
    box = tuple(item)
    return (box, float(box[3] - box[1]),
            0.5 * (box[1] + box[3]), 0.5 * (box[0] + box[2]), 0.0)


class MotoGeomSmoother:
    """EMA of blur height, centre and lean angle, matched across frames by IoU."""

    def __init__(self, alpha: float = 0.22, alpha_pos: float = 0.35,
                 alpha_ang: float = 0.28, min_iou: float = 0.15):
        self.alpha = float(alpha)
        self.alpha_pos = float(alpha_pos)
        self.alpha_ang = float(alpha_ang)
        self.min_iou = float(min_iou)
        self._prev: list[tuple[tuple, float, float, float, float]] = []

    def update(self, samples) -> dict[tuple, tuple[float, float, float, float]]:
        """Return {box: (smooth_h, smooth_cy, smooth_cx, smooth_angle)}."""
        curr = [_as_sample(s) for s in samples]
        mapping = _match_prev([p[0] for p in self._prev],
                              [c[0] for c in curr], self.min_iou)
        a_h = min(1.0, max(0.0, self.alpha))
        a_p = min(1.0, max(0.0, self.alpha_pos))
        a_a = min(1.0, max(0.0, self.alpha_ang))
        out: dict[tuple, tuple[float, float, float, float]] = {}
        new_state = []
        for j, (box, h, cy, cx, ang) in enumerate(curr):
            if j in mapping:
                _, ph, pcy, pcx, pang = self._prev[mapping[j]]
                h = a_h * h + (1.0 - a_h) * ph
                cy = a_p * cy + (1.0 - a_p) * pcy
                cx = a_p * cx + (1.0 - a_p) * pcx
                ang = _fold_ellipse_angle(pang + a_a * _wrap_delta(pang, ang))
            out[box] = (h, cy, cx, ang)
            new_state.append((box, h, cy, cx, ang))
        self._prev = new_state
        return out


def moto_base_zone(
    frame,
    box,
    *,
    width_frac: float = 0.42,
    height_frac: float = 1.0 / 3.0,
    aspect: float = 1.82,
    min_width: float = 36.0,
    min_height: float = 22.0,
    strip_frac: float = 0.30,
    snap_frac: float = 0.04,
    max_shift_frac: float = 0.28,
    snap_angle: float = 8.0,
    rear_along_frac: float = 0.20,
    rear_y0_frac: float = 0.45,
    rear_y1_frac: float = 0.72,
    wide_ar: float = 0.95,
    wide_y0_frac: float = 0.28,
    wide_y1_frac: float = 0.58,
    rear_nudge_frac: float | None = None,
    smooth_h: float | None = None,
    smooth_cy: float | None = None,
    smooth_cx: float | None = None,
    smooth_angle: float | None = None,
):
    """Wide oval, centre h/2 up from the box bottom, rotated with lean.

    Vertical size = box height × *height_frac* (default h/3). Horizontal
    size = vertical × *aspect*. Y is always ``by2 - use_h/2`` so a jumping
    helmet / energy blob cannot lift the oval. *width_frac* and the old
    rear/wide-band kwargs are unused (kept so callers can pass them).
    """
    del width_frac, rear_along_frac, rear_y0_frac, rear_y1_frac
    del wide_ar, wide_y0_frac, wide_y1_frac, rear_nudge_frac, smooth_cy
    bx1, by1, bx2, by2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    bw = bx2 - bx1
    bh = by2 - by1
    if bw < 8 or bh < 8:
        return None
    use_h = float(smooth_h) if smooth_h is not None else bh
    zh = min(use_h, max(float(min_height), use_h * float(height_frac)))
    zw = max(float(min_width) * 0.5, zh * float(aspect))
    ax, ay = 0.5 * zw, 0.5 * zh

    _cx0, _cy0, raw_angle = energy_pose(
        frame, (bx1, by1, bx2, by2), snap_angle=snap_angle)
    angle = float(smooth_angle) if smooth_angle is not None else raw_angle
    angle = _fold_ellipse_angle(angle)
    if abs(angle) < float(snap_angle):
        angle = 0.0

    box_cx = 0.5 * (bx1 + bx2)
    if smooth_cx is not None:
        cx = float(smooth_cx)
    else:
        mass_cx = bottom_mass_cx(frame, (bx1, by1, bx2, by2), strip_frac=strip_frac)
        max_shift = bw * float(max_shift_frac)
        cx = box_cx + max(-max_shift, min(max_shift, mass_cx - box_cx))
    max_shift = bw * float(max_shift_frac)
    shift = max(-max_shift, min(max_shift, cx - box_cx))
    cx = box_cx + shift
    # From the bottom: stable when the top of the YOLO box flickers.
    cy = by2 - 0.5 * use_h
    cy = max(by1 + ay, min(by2 - ay, cy))

    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = ellipse_aabb(cx, cy, ax, ay, angle)
    if x1 < 0:
        cx -= x1
    if y1 < 0:
        cy -= y1
    if x2 > fw:
        cx -= x2 - fw
    if y2 > fh:
        cy -= y2 - fh
    x1, y1, x2, y2 = ellipse_aabb(cx, cy, ax, ay, angle)
    span_w, span_h = x2 - x1, y2 - y1
    if span_w > fw or span_h > fh:
        scale = min(fw / max(span_w, 1e-6), fh / max(span_h, 1e-6), 1.0)
        ax *= scale
        ay *= scale
        x1, y1, x2, y2 = ellipse_aabb(cx, cy, ax, ay, angle)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    leaned = abs(angle) >= float(snap_angle) or abs(cx - 0.5 * (bx1 + bx2)) >= bw * float(snap_frac)
    src = "base_lean" if leaned else "base"
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
            1.0, src, -1, float(angle), float(ax), float(ay))


def _plate_in_moto(plate, box) -> bool:
    pcx = 0.5 * (plate[0] + plate[2])
    pcy = 0.5 * (plate[1] + plate[3])
    return box[0] <= pcx <= box[2] and box[1] <= pcy <= box[3]


def add_moto_base_zones(
    frame,
    vehicles,
    plates,
    large_boxes,
    *,
    enabled: bool = True,
    only_if_no_plate: bool = False,
    smoother: MotoGeomSmoother | None = None,
    **zone_kw,
) -> list:
    """Append geometric base zones for large motorcycles."""
    if not enabled or frame is None:
        return list(plates)
    out = list(plates)
    gated = large_boxes is not None
    large = set(large_boxes or ())
    moto_boxes = []
    for v in vehicles:
        if v[0] != _MOTO_CLS:
            continue
        box = (int(v[1]), int(v[2]), int(v[3]), int(v[4]))
        if gated and box not in large:
            continue
        moto_boxes.append(box)

    samples = []
    snap_angle = float(zone_kw.get("snap_angle", 8.0))
    strip_frac = float(zone_kw.get("strip_frac", 0.30))
    for box in moto_boxes:
        bh = float(box[3] - box[1])
        cy = float(box[3]) - 0.5 * bh
        cx = bottom_mass_cx(frame, box, strip_frac=strip_frac)
        ang = energy_pose(frame, box, snap_angle=snap_angle)[2]
        samples.append((box, bh, cy, cx, ang))

    smooth = smoother.update(samples) if smoother is not None else {}
    for box in moto_boxes:
        if only_if_no_plate and any(_plate_in_moto(p, box) for p in out):
            continue
        extra = {}
        if box in smooth:
            sh, _scy, scx, sang = smooth[box]
            extra = dict(smooth_h=sh, smooth_cx=scx, smooth_angle=sang)
        zone = moto_base_zone(frame, box, **zone_kw, **extra)
        if zone is not None:
            out.append(zone)
    return out
