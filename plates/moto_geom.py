"""Geometric motorcycle plate fallback: circle, centre h/2 from the box bottom.

Diameter is one third of the green box height. Centre is horizontally the
box midpoint and vertically ``by2 - h/2`` (from the wheels, not the helmet).
Height is EMA-smoothed per moto so YOLO box flicker is damped.

The plate model still owns in-box hits. The circle is added only when that
motorcycle has no plate zone (``only_if_no_plate``).
"""
from __future__ import annotations

from plates.common import iou

_MOTO_CLS = 3


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
    """Accept a bare box or (box, h, cy, cx). Extra fields are ignored."""
    if item and isinstance(item[0], (tuple, list)):
        box = tuple(item[0])
        h = float(item[1])
        cx = float(item[3]) if len(item) > 3 else 0.5 * (box[0] + box[2])
        return (box, h, cx)
    box = tuple(item)
    return (box, float(box[3] - box[1]), 0.5 * (box[0] + box[2]))


class MotoGeomSmoother:
    """EMA of blur height and horizontal centre, matched across frames by IoU."""

    def __init__(self, alpha: float = 0.22, alpha_pos: float = 0.35,
                 min_iou: float = 0.15, **_unused):
        self.alpha = float(alpha)
        self.alpha_pos = float(alpha_pos)
        self.min_iou = float(min_iou)
        self._prev: list[tuple[tuple, float, float]] = []

    def update(self, samples) -> dict[tuple, tuple[float, float]]:
        """Return {box: (smooth_h, smooth_cx)}."""
        curr = [_as_sample(s) for s in samples]
        mapping = _match_prev([p[0] for p in self._prev],
                              [c[0] for c in curr], self.min_iou)
        a_h = min(1.0, max(0.0, self.alpha))
        a_p = min(1.0, max(0.0, self.alpha_pos))
        out: dict[tuple, tuple[float, float]] = {}
        new_state = []
        for j, (box, h, cx) in enumerate(curr):
            if j in mapping:
                _, ph, pcx = self._prev[mapping[j]]
                h = a_h * h + (1.0 - a_h) * ph
                cx = a_p * cx + (1.0 - a_p) * pcx
            out[box] = (h, cx)
            new_state.append((box, h, cx))
        self._prev = new_state
        return out


def moto_base_zone(
    frame,
    box,
    *,
    height_frac: float = 1.0 / 3.0,
    min_height: float = 22.0,
    smooth_h: float | None = None,
    smooth_cx: float | None = None,
    **_unused,
):
    """Circle of diameter box-h × *height_frac*, centre h/2 up from the bottom."""
    bx1, by1, bx2, by2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    bw = bx2 - bx1
    bh = by2 - by1
    if bw < 8 or bh < 8:
        return None
    use_h = float(smooth_h) if smooth_h is not None else bh
    d = min(use_h, max(float(min_height), use_h * float(height_frac)))
    r = 0.5 * d
    cx = float(smooth_cx) if smooth_cx is not None else 0.5 * (bx1 + bx2)
    cy = by2 - 0.5 * use_h
    cy = max(by1 + r, min(by2 - r, cy))

    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
    if x1 < 0:
        cx -= x1
    if y1 < 0:
        cy -= y1
    if x2 > fw:
        cx -= x2 - fw
    if y2 > fh:
        cy -= y2 - fh
    if r * 2 > fw or r * 2 > fh:
        r = 0.5 * min(fw, fh, d)
    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
            1.0, "base", -1, 0.0, float(r), float(r))


def _is_base_zone(plate) -> bool:
    return len(plate) > 5 and str(plate[5]).startswith("base")


def _plate_in_moto(plate, box) -> bool:
    if _is_base_zone(plate):
        return False
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
    """Append geometric base circles for large motorcycles without a plate hit."""
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
    for box in moto_boxes:
        bh = float(box[3] - box[1])
        cx = 0.5 * (box[0] + box[2])
        samples.append((box, bh, 0.0, cx))

    smooth = smoother.update(samples) if smoother is not None else {}
    for box in moto_boxes:
        if only_if_no_plate and any(_plate_in_moto(p, box) for p in out):
            continue
        extra = {}
        if box in smooth:
            sh, scx = smooth[box]
            extra = dict(smooth_h=sh, smooth_cx=scx)
        zone = moto_base_zone(frame, box, **zone_kw, **extra)
        if zone is not None:
            out.append(zone)
    return out
