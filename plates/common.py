# ─── Small geometry / parsing helpers ───────────────────────────────────────


def time_to_seconds(time_str: str) -> float:
    """Convert MM:SS or HH:MM:SS to float seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(time_str)


def parse_region(s: str):
    """Parse 'x1,y1,x2,y2' string into a tuple of ints."""
    parts = [int(v.strip()) for v in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected x1,y1,x2,y2 but got: {s!r}")
    return tuple(parts)

def _overlaps(a, b):
    """Return True if rectangle a overlaps rectangle b."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

def covered_fraction(inner, outer) -> float:
    """Fraction of *inner*'s area that falls inside *outer*.

    Used to answer the only question that matters for privacy: is this plate
    actually hidden by the region we are about to blur?
    """
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area  = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0

def suppress_duplicate_plates(plate_rects, vehicle_boxes):
    """Keep only the highest-confidence plate inside each vehicle box.

    SAHI tiles overlap, so the same physical plate can produce 2-3 slightly
    offset detections that survive IoU-NMS. This enforces the physical
    constraint that only one plate face is visible per vehicle at a time.
    Plates not inside any vehicle box are passed through unchanged.

    Returns (kept, suppressed) so callers can visualise the dropped detections.
    """
    if not vehicle_boxes or not plate_rects:
        return plate_rects, []
    claimed    = [False] * len(plate_rects)
    kept       = []
    suppressed = []
    for vb in vehicle_boxes:
        inside = [(i, p) for i, p in enumerate(plate_rects)
                  if not claimed[i] and _overlaps(p[:4], vb)]
        if inside:
            best_i, best_p = max(inside, key=lambda ip: ip[1][4] if len(ip[1]) > 4 else 1.0)
            kept.append(best_p)
            for i, p in inside:
                claimed[i] = True
                if i != best_i:
                    suppressed.append(p)
    kept.extend(p for i, p in enumerate(plate_rects) if not claimed[i])
    return kept, suppressed

def merge_overlapping(rects, iou_thresh=0.3):
    """Merge highly overlapping rectangles (greedy NMS)."""
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept = []
    suppressed = [False] * len(rects)
    for i, r in enumerate(rects):
        if suppressed[i]:
            continue
        kept.append(r)
        for j in range(i + 1, len(rects)):
            if suppressed[j]:
                continue
            if iou(r, rects[j]) > iou_thresh:
                suppressed[j] = True
    return kept


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

def _clamp_rect_max(rect, w, h, max_frac=0.20):
    """Shrink a rect toward its centre if it covers more than max_frac
    of the frame's width or height (prevents gap-fill boxes ballooning)."""
    x1, y1, x2, y2 = rect[:4]
    bw, bh = x2 - x1, y2 - y1
    mw, mh = w * max_frac, h * max_frac
    if bw <= mw and bh <= mh:
        return rect
    scale = min(mw / max(bw, 1), mh / max(bh, 1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * scale, bh * scale
    clamped = (int(cx - nw / 2), int(cy - nh / 2),
               int(cx + nw / 2), int(cy + nh / 2))
    if len(rect) > 4:
        return (*clamped, *rect[4:])
    return clamped

def parse_color(s: str):
    """Parse 'R,G,B' (0-255) into a BGR tuple for OpenCV."""
    parts = [int(v.strip()) for v in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B but got: {s!r}")
    if any(p < 0 or p > 255 for p in parts):
        raise ValueError(f"Color components must be 0-255: {s!r}")
    r, g, b = parts
    return (b, g, r)   # OpenCV uses BGR

def _format_duration(seconds: float) -> str:
    """Human-readable duration: '12.3s', '5m23s', or '1h12m05s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(round(seconds)), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(round(seconds)), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"
