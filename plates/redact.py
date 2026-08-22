# ─── Redaction: blur / colour / image overlay ───────────────────────────────
import cv2
import numpy as np

from plates.common import _clamp_rect_max


def _blur_kernel(strength: int, roi_w: int, roi_h: int) -> int:
    """Odd Gaussian kernel. Close plates need a bigger k or letters stay readable."""
    k = max(1, int(strength)) | 1
    short = min(max(0, int(roi_w)), max(0, int(roi_h)))
    extra = (int(short * 0.42) | 1) if short else k
    return min(81, max(k, extra))


def apply_blur_feathered(frame, rects, blur_strength=61, padding=8,
                         feather=14):
    """Rectangular Gaussian blur with a soft feathered boundary.

    The forced audit patches used a hard rectangle edge, which the plate
    detector latched onto as a new "plate".  Feathering spreads the
    blurred->sharp transition over ~2*feather pixels: the patch blends into
    the surroundings and no crisp rectangular edge remains.
    """
    h, w = frame.shape[:2]
    for rect in rects:
        x1, y1 = max(0, int(rect[0]) - padding), max(0, int(rect[1]) - padding)
        x2, y2 = min(w, int(rect[2]) + padding), min(h, int(rect[3]) + padding)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        k = _blur_kernel(blur_strength, x2 - x1, y2 - y1)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        mask = np.zeros(roi.shape[:2], dtype=np.float32)
        iw, ih = mask.shape[1], mask.shape[0]
        sx1, sy1 = min(feather, iw // 2), min(feather, ih // 2)
        sx2, sy2 = max(iw - feather, iw // 2), max(ih - feather, ih // 2)
        cv2.rectangle(mask, (sx1, sy1), (sx2, sy2), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        m = mask[:, :, None]
        frame[y1:y2, x1:x2] = (blurred * m + roi * (1.0 - m)).astype(np.uint8)
    return frame

def apply_blur(frame, rects, blur_strength=61, padding=8):
    """Apply strong Gaussian blur to each rectangle region."""
    h, w = frame.shape[:2]
    for rect in rects:
        x1, y1, x2, y2 = rect[:4]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = _blur_kernel(blur_strength, x2 - x1, y2 - y1)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        frame[y1:y2, x1:x2] = blurred
    return frame

def estimate_plate_quad(frame, rect, ar_min: float = 0.8, ar_max: float = 2.5,
                        min_area_frac: float = 0.03, max_area_frac: float = 0.95,
                        max_center_shift: float = 0.45):
    """
    Refine an axis-aligned zone into the actual (possibly rotated) plate quad
    using the plate's high-contrast border (cv2.minAreaRect on the strongest
    contour inside the zone). Returns a 4x2 int array of corner points in
    full-frame coordinates, or None when no plausible plate shape is found —
    callers then fall back to the plain rectangular blur.

    This is the OpenCV-only step that gives the OBB effect (blur glued to a
    leaning plate) without retraining anything.
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(rect[0])), max(0, int(rect[1]))
    x2, y2 = min(w, int(rect[2])), min(h, int(rect[3]))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone_area = (x2 - x1) * (y2 - y1)
    if zone_area <= 0:
        return None
    zone_cx = (x2 - x1) * 0.5
    zone_cy = (y2 - y1) * 0.5
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if not (zone_area * min_area_frac <= area <= zone_area * max_area_frac):
            continue
        rect_c = cv2.minAreaRect(c)
        (cw, ch), _ = rect_c[1], rect_c[2]
        if min(cw, ch) < 6:
            continue
        ar = max(cw, ch) / max(min(cw, ch), 1e-6)
        if not (ar_min <= ar <= ar_max):
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        ccx = m["m10"] / m["m00"]
        ccy = m["m01"] / m["m00"]
        if abs(ccx - zone_cx) / (x2 - x1) > max_center_shift:
            continue
        if abs(ccy - zone_cy) / (y2 - y1) > max_center_shift:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None:
        return None
    rect_m = cv2.minAreaRect(best)
    quad = cv2.boxPoints(rect_m).astype(np.int32)
    quad[:, 0] += x1
    quad[:, 1] += y1
    return quad

def _align_quad(q_ref, q_new):
    """Re-order q_new's corner points to match q_ref's ordering so a pointwise
    EMA blend between consecutive frames makes geometric sense (minAreaRect
    does not guarantee a stable corner ordering between frames)."""
    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_new = np.asarray(q_new, dtype=np.float64)
    best, best_d = q_new, float("inf")
    for flip in (False, True):
        pts = q_new[::-1] if flip else q_new
        for r in range(4):
            cand = np.roll(pts, r, axis=0)
            d = float(np.sum((cand - q_ref) ** 2))
            if d < best_d:
                best_d = d
                best = cand
    return best


def _match_quad_state(quad_hist, rect, max_shift=0.5):
    """Find the previous-frame quad whose zone centre is closest to `rect`'s
    centre (within max_shift × the rect's max side). Returns the history index
    or -1. This is what lets a quad stay temporally stable (EMA + hysteresis)
    instead of flipping between rotated and axis-aligned shapes every frame."""
    cx = (rect[0] + rect[2]) * 0.5
    cy = (rect[1] + rect[3]) * 0.5
    diag = max(rect[2] - rect[0], rect[3] - rect[1], 1.0)
    best_i, best_d = -1, float("inf")
    for i, st in enumerate(quad_hist):
        sx = (st["rect"][0] + st["rect"][2]) * 0.5
        sy = (st["rect"][1] + st["rect"][3]) * 0.5
        d = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best_i = i
    if best_i >= 0 and best_d <= max_shift * diag:
        return best_i
    return -1

def _quad_state_update(st, zone_rect, quad, zarea,
                       enter_ratio=0.75, hold_ratio=0.40,
                       min_active=10, fallback_steps=4):
    """Update one quad_hist entry with the current frame's quad estimate.
    Returns (q_blend, use_quad).

    - EMA-blends the new quad with the previous one; the blend favours the
      new estimate (alpha 0.65 vs 0.5) while the anchor zone is moving fast
      (lean/curve), so the blur does not lag behind the plate.
    - Consistency gate (asymmetric): a new estimate is trusted only when
      (a) its centre lies close to the glided held quad (per-axis), and it
      does not shrink the coverage by more than half, or (b) it is a
      plate-like shape near the zone centre — the plate prior. So a wrong
      lock (top case / bauletto inside the widened zone) never replaces the
      plate, but a real plate relocation still replaces a wrong lock.
      Rejected estimates behave as a lost quad: hold + glide.
    - Minimum active duration: an active quad survives short weak patches
      (shadow, motion blur) for `min_active` frames before falling back.
    - Smooth fallback: when the quad is finally given up, it expands toward
      the anchor rect over `fallback_steps` frames instead of snapping, so
      the redaction shape never flashes.
    """
    prev_quad = st["quad"]
    prev_rect = st["rect"]
    was_active = st["active"]
    dx = dy = 0.0
    if prev_rect is not None:
        dx = ((zone_rect[0] + zone_rect[2]) - (prev_rect[0] + prev_rect[2])) * 0.5
        dy = ((zone_rect[1] + zone_rect[3]) - (prev_rect[1] + prev_rect[3])) * 0.5
    glided = prev_quad.astype(np.float64)
    glided[:, 0] += dx
    glided[:, 1] += dy
    q_blend = glided.astype(np.int32)
    ratio = 0.0
    fresh = False
    if quad is not None and zarea > 0:
        q_new = _align_quad(prev_quad, quad)
        qc = q_new.mean(axis=0)
        gc = glided.mean(axis=0)
        zcx = (zone_rect[0] + zone_rect[2]) * 0.5
        zcy = (zone_rect[1] + zone_rect[3]) * 0.5
        gw = float(np.max(glided[:, 0]) - np.min(glided[:, 0]))
        gh = float(np.max(glided[:, 1]) - np.min(glided[:, 1]))
        zside = max(zone_rect[2] - zone_rect[0], zone_rect[3] - zone_rect[1], 1.0)
        ratio_new = float(cv2.contourArea(
            np.ascontiguousarray(q_new, dtype=np.float32))) / zarea
        shrink = ratio_new < 0.5 * st.get("ratio", 0.0)
        consistent = (not shrink
                      and abs(qc[0] - gc[0]) <= 0.4 * max(gw, 1.0)
                      and abs(qc[1] - gc[1]) <= 0.4 * max(gh, 1.0))
        at_prior = (np.hypot(qc[0] - zcx, qc[1] - zcy) <= 0.30 * zside
                    and ratio_new >= max(hold_ratio, st.get("ratio", 0.0)))
        if at_prior and not consistent:
            # Relocation: a plate-like estimate at the plate position
            # replaces the current lock immediately (bauletto -> plate).
            alpha = 0.8
        elif consistent:
            alpha = 0.65 if max(abs(dx), abs(dy)) >= 8.0 else 0.5
        else:
            # Wrong lock (top case, shadow, edge): ignore, keep gliding.
            alpha = 0.0
        if alpha > 0.0:
            fresh = True
            q_blend = ((1.0 - alpha) * prev_quad + alpha * q_new).astype(np.int32)
            ratio = float(cv2.contourArea(
                np.ascontiguousarray(q_blend, dtype=np.float32))) / zarea
        else:
            ratio = float(cv2.contourArea(
                np.ascontiguousarray(glided, dtype=np.float32))) / zarea
    # Re-engagement rules: an inactive state (or one whose fallback expansion
    # is in progress) only comes back on a fresh estimate — never on the held
    # shape alone (the growing quad's own ratio would cancel the fallback).
    # During a fallback the fresh estimate must be strong (enter level), or
    # the state would re-lock on the same marginal quad that got given up.
    in_fallback = st.get("fallback", 0) > 0
    if in_fallback:
        strong = fresh and ratio >= enter_ratio
    else:
        threshold = hold_ratio if was_active else enter_ratio
        strong = ratio >= threshold and (fresh or was_active)
    if strong:
        st["fails"] = 0
        st["fallback"] = 0
        use_quad = True
    elif was_active and st.get("fails", 0) < min_active:
        # Short weak patch: keep the held quad gliding with the zone.
        st["fails"] = st.get("fails", 0) + 1
        q_blend = glided.astype(np.int32)
        ratio = (float(cv2.contourArea(
            np.ascontiguousarray(q_blend, dtype=np.float32))) / zarea
                 if zarea > 0 else 0.0)
        use_quad = True
    elif was_active:
        # Give up on the quad: expand it toward the anchor-sized rect, but
        # centred on the held plate position — the geometric anchor can sit
        # low on a leaning/close moto and would drag the blur off the plate.
        fall = st.get("fallback", 0) + 1
        st["fallback"] = fall
        t = fall / (fallback_steps + 1.0)
        bx1, by1 = float(glided[:, 0].min()), float(glided[:, 1].min())
        bx2, by2 = float(glided[:, 0].max()), float(glided[:, 1].max())
        cxm, cym = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        rw = zone_rect[2] - zone_rect[0]
        rh = zone_rect[3] - zone_rect[1]
        tx1 = cxm - rw * 0.5
        ty1 = cym - rh * 0.5
        tx2 = cxm + rw * 0.5
        ty2 = cym + rh * 0.5
        nx1 = bx1 + (tx1 - bx1) * t
        ny1 = by1 + (ty1 - by1) * t
        nx2 = bx2 + (tx2 - bx2) * t
        ny2 = by2 + (ty2 - by2) * t
        qb = glided.copy()
        qb[:, 0] = cxm + (qb[:, 0] - cxm) * ((nx2 - nx1) / max(bx2 - bx1, 1e-9))
        qb[:, 1] = cym + (qb[:, 1] - cym) * ((ny2 - ny1) / max(by2 - by1, 1e-9))
        q_blend = qb.astype(np.int32)
        ratio = (float(cv2.contourArea(
            np.ascontiguousarray(q_blend, dtype=np.float32))) / zarea
                 if zarea > 0 else 0.0)
        use_quad = fall < fallback_steps
    else:
        use_quad = False
    st["rect"] = zone_rect
    st["quad"] = q_blend
    st["ratio"] = ratio
    st["active"] = use_quad
    return q_blend, use_quad

def apply_blur_rotated(frame, quads, blur_strength=61, padding=8):
    """Gaussian-blur only the pixels inside each rotated quad (4x2 int array
    of full-frame corner points); everything outside is untouched."""
    h, w = frame.shape[:2]
    for quad in quads:
        q = np.asarray(quad, dtype=np.int32)
        x1 = max(0, int(q[:, 0].min()) - padding)
        y1 = max(0, int(q[:, 1].min()) - padding)
        x2 = min(w, int(q[:, 0].max()) + padding)
        y2 = min(h, int(q[:, 1].max()) + padding)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        k = _blur_kernel(blur_strength, x2 - x1, y2 - y1)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [(q - np.array([x1, y1]))], 255)
        roi[mask > 0] = blurred[mask > 0]
    return frame

def apply_blur_circle(frame, rects, blur_strength=61, padding=8):
    """Gaussian-blur pixels inside the circle inscribed in each zone AABB."""
    h, w = frame.shape[:2]
    for rect in rects:
        x1, y1, x2, y2 = (float(rect[0]), float(rect[1]),
                          float(rect[2]), float(rect[3]))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if len(rect) >= 10:
            r = max(float(rect[8]), float(rect[9])) + padding
        else:
            r = 0.5 * min(x2 - x1, y2 - y1) + padding
        if r < 1:
            continue
        x0 = max(0, int(np.floor(cx - r)))
        y0 = max(0, int(np.floor(cy - r)))
        x1c = min(w, int(np.ceil(cx + r)))
        y1c = min(h, int(np.ceil(cy + r)))
        if x1c - x0 < 2 or y1c - y0 < 2:
            continue
        roi = frame[y0:y1c, x0:x1c]
        k = _blur_kernel(blur_strength, x1c - x0, y1c - y0)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(round(cx - x0)), int(round(cy - y0))),
                   max(1, int(round(r))), 255, -1)
        roi[mask > 0] = blurred[mask > 0]
    return frame

def apply_solid_color(frame, rects, color=(0, 0, 0), padding=8):
    """Fill each rectangle region with a solid BGR colour."""
    h, w = frame.shape[:2]
    for rect in rects:
        x1, y1, x2, y2 = rect[:4]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    return frame

def apply_image_overlay(frame, rects, overlay_img, padding=8):
    """
    Paste *overlay_img* (BGR ndarray, already loaded) onto each rectangle
    region, stretched to fill the padded rect exactly.

    overlay_img may include an alpha channel (4 channels).  If alpha is
    present it is used for compositing so the corners of e.g. a circular
    logo blend with the underlying frame; otherwise the overlay covers
    the rect opaquely.
    """
    h, w = frame.shape[:2]
    has_alpha = overlay_img.ndim == 3 and overlay_img.shape[2] == 4
    for rect in rects:
        x1, y1, x2, y2 = rect[:4]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            continue
        resized = cv2.resize(overlay_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        if has_alpha:
            bgr = resized[:, :, :3].astype(np.float32)
            alpha = (resized[:, :, 3:4].astype(np.float32)) / 255.0
            roi = frame[y1:y2, x1:x2].astype(np.float32)
            blended = bgr * alpha + roi * (1.0 - alpha)
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = resized
    return frame

def apply_redaction(frame, rects, mode="blur",
                    blur_strength=61, color=(0, 0, 0),
                    overlay_img=None, padding=8, max_box_frac=0.20):
    """
    Dispatcher for the three redaction modes.

    mode = "blur"  → Gaussian blur (apply_blur)
    mode = "color" → solid colour fill (apply_solid_color)
    mode = "image" → stretched overlay image (apply_image_overlay)

    Falls back to blur if mode == "image" but no overlay_img is provided.
    """
    h, w = frame.shape[:2]
    rects = [_clamp_rect_max(r, w, h, max_box_frac) for r in rects]
    if mode == "color":
        return apply_solid_color(frame, rects, color=color, padding=padding)
    if mode == "image" and overlay_img is not None:
        return apply_image_overlay(frame, rects, overlay_img, padding=padding)
    return apply_blur(frame, rects, blur_strength=blur_strength, padding=padding)

def load_overlay_image(path: str):
    """Load an overlay image (PNG/JPG); preserves alpha channel when present."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read overlay image: {path}")
    return img
