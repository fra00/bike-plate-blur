# ─── Debug overlays (boxes, trajectories, HUD panel) ───────────────────────
import os

import cv2
import numpy as np

from plates.common import iou
from plates.constants import (
    VEHICLE_CLASSES,
    _DBG_FONT, _DBG_OWN_COLOR,
    _DBG_PLATE_COLOR, _DBG_PREDICT_COLOR, _DBG_BASE_COLOR,
    _DBG_SUPPRESSED_COLOR,
    _DBG_VEHICLE_COLOR, _DBG_VEHICLE_SMALL_COLOR,
    _DD_BLUE, _DD_DIM, _DD_F_BODY, _DD_F_DISP, _DD_F_MONO, _DD_F_MONOB,
    _DD_FONT_DIR, _DD_GHOST, _DD_GREEN, _DD_HUD_BG, _DD_HUD_W,
    _DD_RED, _DD_SOURCE_COLOR, _DD_VOID_BG,
    _DD_WHITE,
)
from plates.redact import (
    apply_redaction,
    apply_blur_circle,
)

_MOTO_CLS = 3


def _is_base_zone(rect) -> bool:
    return len(rect) > 5 and str(rect[5]).startswith("base")


def _zone_circle(rect):
    x1, y1, x2, y2 = rect[:4]
    cx = 0.5 * (float(x1) + float(x2))
    cy = 0.5 * (float(y1) + float(y2))
    if len(rect) >= 10:
        r = max(float(rect[8]), float(rect[9]))
    else:
        r = 0.5 * min(float(x2) - float(x1), float(y2) - float(y1))
    return cx, cy, r


def _dbg_circle(img, rect, color, label, thickness=2):
    cx, cy, r = _zone_circle(rect)
    cx, cy = int(round(cx)), int(round(cy))
    ir = max(1, int(round(r)))
    cv2.circle(img, (cx, cy), ir, color, thickness)
    fs = max(0.55, ir / 180)
    (tw, th), _ = cv2.getTextSize(label, _DBG_FONT, fs, 2)
    pad = 6
    lx = min(max(0, cx - tw // 2), img.shape[1] - tw - pad * 2)
    by1 = max(0, cy - ir - th - pad * 2)
    cv2.rectangle(img, (lx, by1), (lx + tw + pad * 2, by1 + th + pad * 2), color, -1)
    cv2.putText(img, label, (lx + pad, by1 + th + pad), _DBG_FONT, fs,
                (255, 255, 255), 2, cv2.LINE_AA)


def _vehicle_box_style(cls, x1, y1, x2, y2, conf, frame_h, min_moto_h_frac,
                       moto_large_boxes=None):
    """Green if the vehicle can enter blur zones; magenta if moto is too small."""
    name = VEHICLE_CLASSES.get(cls, str(cls))
    label = f"{name} {conf:.2f}"
    if cls != _MOTO_CLS or min_moto_h_frac <= 0:
        return _DBG_VEHICLE_COLOR, label
    box = (int(x1), int(y1), int(x2), int(y2))
    if moto_large_boxes is not None:
        is_large = box in moto_large_boxes
    else:
        size = max(x2 - x1, y2 - y1)
        is_large = size >= min_moto_h_frac * float(frame_h)
    if not is_large:
        return _DBG_VEHICLE_SMALL_COLOR, f"{label} small"
    return _DBG_VEHICLE_COLOR, label


def _dbg_box(img, x1, y1, x2, y2, color, label, thickness=3):
    h_img, w_img = img.shape[:2]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    fs = max(0.6, (x2 - x1) / 500)
    (tw, th), _ = cv2.getTextSize(label, _DBG_FONT, fs, 2)
    pad = 6
    # Flip label below the box when it would clip above the frame top
    if y1 - th - pad * 2 >= 0:
        bg_y1, bg_y2, ty = y1 - th - pad * 2, y1, y1 - pad
    else:
        bg_y1, bg_y2, ty = y2, y2 + th + pad * 2, y2 + th + pad
    # Clamp label right edge to frame width
    lx = min(x1, w_img - tw - pad * 2)
    cv2.rectangle(img, (lx, bg_y1), (lx + tw + pad * 2, bg_y2), color, -1)
    cv2.putText(img, label, (lx + pad, ty), _DBG_FONT, fs, (255, 255, 255), 2, cv2.LINE_AA)

def draw_debug_overlay(frame, plate_rects, all_vehicles, own_plate_region=None,
                       blur_padding=8, blur_strength=61,
                       redact_mode="blur", redact_color=(0, 0, 0),
                       overlay_img=None, observed_plates=None,
                       min_moto_h_frac=0.08333, moto_large_boxes=None):
    """
    Returns a debug frame that shows detections plus production redaction:
      - Green box    = vehicle YOLO (moto large enough for blur zones)
      - Magenta box  = motorcycle below the size gate (max side + hysteresis)
      - Blue box     = plate-model hit that became a blur zone
      - Grey box     = plate-model hit that tracking dropped (skip)
      - Cyan circle  = geometric moto zone (centre h/2 from box bottom, height h/3)
      - Yellow box   = interpolated plate (gap fill, not a detection)
      - Orange box   = own-plate fixed region
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ── Step 1: apply the real redaction so the frame looks like production ──
    all_rects = list(plate_rects)
    if own_plate_region:
        all_rects.append(own_plate_region)
    other = [r for r in all_rects if not _is_base_zone(r)]
    base = [r for r in all_rects if _is_base_zone(r)]
    if other:
        vis = apply_redaction(vis, other, mode=redact_mode,
                              blur_strength=blur_strength,
                              color=redact_color,
                              overlay_img=overlay_img,
                              padding=blur_padding)
    if base:
        vis = apply_blur_circle(
            vis, base, blur_strength=blur_strength, padding=blur_padding)

    # ── Step 2: vehicle boxes (green = usable, magenta = moto too small) ──
    for vi, (cls, x1, y1, x2, y2, conf) in enumerate(all_vehicles):
        color, label = _vehicle_box_style(
            cls, x1, y1, x2, y2, conf, h, min_moto_h_frac, moto_large_boxes)
        _dbg_box(vis, x1, y1, x2, y2, color, label, thickness=3)

    # ── Step 3: plate-model boxes ────────────────────────────────────────────
    # Blue = a cache hit that became a blur zone. Grey = detector fired but
    # tracking dropped it (tiny / nested blob, small moto, size jump).
    raw = observed_plates if observed_plates is not None else [
        r for r in plate_rects
        if (r[5] if len(r) > 5 else "sahi") != "bridge"
    ]
    used = [
        r for r in plate_rects
        if (r[5] if len(r) > 5 else "") not in ("bridge", "own")
        and not str(r[5] if len(r) > 5 else "").startswith("base")
    ]
    for rect in raw:
        x1, y1, x2, y2 = (int(rect[i]) for i in range(4))
        conf = rect[4] if len(rect) > 4 else None
        source = rect[5] if len(rect) > 5 else "sahi"
        if source in ("bridge", "own") or str(source).startswith("base"):
            continue
        kept = any(iou((x1, y1, x2, y2), z[:4]) >= 0.3 for z in used)
        if kept:
            label = f"plate {conf:.2f}" if conf is not None else "plate"
            _dbg_box(vis, x1, y1, x2, y2, _DBG_PLATE_COLOR, label, thickness=2)
        else:
            label = f"skip {conf:.2f}" if conf is not None else "skip"
            _dbg_box(vis, x1, y1, x2, y2, _DBG_SUPPRESSED_COLOR, label,
                     thickness=1)

    # Yellow: interpolated zones (not a model hit)
    for rect in plate_rects:
        source = rect[5] if len(rect) > 5 else "sahi"
        if source != "bridge":
            continue
        x1, y1, x2, y2 = rect[:4]
        _dbg_box(vis, x1, y1, x2, y2, _DBG_PREDICT_COLOR, "interpolated", thickness=2)

    for rect in plate_rects:
        source = rect[5] if len(rect) > 5 else ""
        if not str(source).startswith("base"):
            continue
        _dbg_circle(vis, rect, _DBG_BASE_COLOR, "base", thickness=2)

    # ── Step 4: own-plate fixed region (orange) ───────────────────────────────
    if own_plate_region:
        ox1, oy1, ox2, oy2 = own_plate_region
        _dbg_box(vis, ox1, oy1, ox2, oy2, _DBG_OWN_COLOR, "own plate", thickness=3)

    return vis

def _dd_have_pil_fonts() -> bool:
    """All brand fonts present? Falls back to OpenCV Hershey if missing."""
    return all(os.path.exists(p) for p in
               (_DD_F_DISP, _DD_F_BODY, _DD_F_MONO, _DD_F_MONOB))

def _dd_tag(img, x, y, text, bg_bgr, fg_bgr=_DD_WHITE, fs=0.5, pad=4):
    """Solid pill-style tag for plate / vehicle labels."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    h_img, w_img = img.shape[:2]
    # Draw above the box if there's room, otherwise below
    if y - th - pad * 2 >= 0:
        bg_y1, bg_y2, ty = y - th - pad * 2, y, y - pad
    else:
        bg_y1, bg_y2, ty = y, y + th + pad * 2, y + th + pad
    lx = min(x, w_img - tw - pad * 2)
    cv2.rectangle(img, (lx, bg_y1), (lx + tw + pad * 2, bg_y2), bg_bgr, -1)
    cv2.putText(img, text, (lx + pad, ty), cv2.FONT_HERSHEY_DUPLEX, fs,
                fg_bgr, 1, cv2.LINE_AA)

def _dd_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=8, gap=4):
    """Dashed rectangle for predicted (gap-fill) plates."""
    step = dash + gap
    for ix in range(x1, x2, step):
        cv2.line(img, (ix, y1), (min(ix+dash, x2), y1), color, thickness)
        cv2.line(img, (ix, y2), (min(ix+dash, x2), y2), color, thickness)
    for iy in range(y1, y2, step):
        cv2.line(img, (x1, iy), (x1, min(iy+dash, y2)), color, thickness)

def draw_extended_overlay(frame, plate_rects, all_vehicles,
                          blur_padding=8, blur_strength=61,
                          redact_mode="blur", redact_color=(0, 0, 0),
                          overlay_img=None, rejected_plates=None,
                          min_moto_h_frac=0.08333, moto_large_boxes=None):
    """
    DEBUG DATA - mode A.  Returns a frame with the actual redaction applied
    plus source-tagged plate boxes and vehicle boxes.
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ── Apply real redaction so the user sees production output ───────────
    if plate_rects:
        other = [r for r in plate_rects if not _is_base_zone(r)]
        base = [r for r in plate_rects if _is_base_zone(r)]
        if other:
            vis = apply_redaction(vis, other,
                                  mode=redact_mode,
                                  blur_strength=blur_strength,
                                  color=redact_color,
                                  overlay_img=overlay_img,
                                  padding=blur_padding)
        if base:
            vis = apply_blur_circle(
                vis, base, blur_strength=blur_strength, padding=blur_padding)

    # ── Rejected detections (grey, dashed) — below confidence threshold ───
    # Drawn first so accepted/predicted boxes render on top of them.
    for rect in (rejected_plates or []):
        x1, y1, x2, y2 = rect[:4]
        conf = rect[4] if len(rect) > 4 else None
        _dd_dashed_rect(vis, x1, y1, x2, y2, _DD_GHOST, thickness=1)
        label = f"REJ {conf:.2f}" if conf is not None else "REJ"
        _dd_tag(vis, x1, y1, label, _DD_GHOST, fs=0.4)

    # ── Vehicle boxes (green = usable, magenta = moto too small) ───────────
    for vi, (cls, x1, y1, x2, y2, conf) in enumerate(all_vehicles):
        color, label = _vehicle_box_style(
            cls, x1, y1, x2, y2, conf, h, min_moto_h_frac, moto_large_boxes)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        _dd_tag(vis, x1, y1, label, color)

    # ── Plate boxes with source tag ───────────────────────────────────────
    src_counts = {"sahi": 0, "crop": 0, "crop_moto": 0, "bridge": 0, "own": 0}
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf   = rect[4] if len(rect) > 4 else None
        source = rect[5] if len(rect) > 5 else "sahi"
        count_key = "crop" if source.startswith("crop") else source
        src_counts[count_key] = src_counts.get(count_key, 0) + 1
        color = _DD_SOURCE_COLOR.get(source, _DD_GREEN)

        if source == "bridge":
            _dd_dashed_rect(vis, x1, y1, x2, y2, color, thickness=2)
            _dd_tag(vis, x1, y2 + 20, "BRIDGE  interpolated", color,
                    fg_bgr=(0, 0, 0), fs=0.45)
        elif str(source).startswith("base"):
            cx, cy, r = _zone_circle(rect)
            cv2.circle(
                vis,
                (int(round(cx)), int(round(cy))),
                max(1, int(round(r))),
                color, 2,
            )
            _dd_tag(vis, x1, y1, "BASE", color, fg_bgr=(0, 0, 0))
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if conf is not None and conf >= 0:
                _dd_tag(vis, x1, y1, f"{source.upper()} {conf:.2f}", color,
                        fg_bgr=(0, 0, 0))
            else:
                _dd_tag(vis, x1, y1, source.upper(), color, fg_bgr=(0, 0, 0))

    # ── Top status strip ──────────────────────────────────────────────────
    strip_h = 38
    cv2.rectangle(vis, (0, 0), (w, strip_h), _DD_VOID_BG, -1)
    cv2.rectangle(vis, (0, strip_h - 1), (w, strip_h), _DD_BLUE, 1)
    strip_text = (f"DEBUG DATA   VEH {len(all_vehicles)}   "
                  f"PLT {len(plate_rects)} "
                  f"(obs {src_counts['sahi'] + src_counts['crop'] + src_counts.get('crop_moto', 0)}, "
                  f"bridge {src_counts['bridge']}, own {src_counts['own']})   "
                  f"REJ {len(rejected_plates or [])}")
    cv2.putText(vis, strip_text, (12, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, _DD_WHITE, 1, cv2.LINE_AA)

    return vis

def _dd_font(path, size, variation=None):
    """Load PIL font; honours OpenType variation axes (e.g. Orbitron Bold)."""
    from PIL import ImageFont
    fnt = ImageFont.truetype(path, size)
    if variation is not None:
        try:
            fnt.set_variation_by_name(variation)
        except OSError:
            pass
    return fnt

def draw_hud_panel(frame_h, telemetry):
    """
    DEBUG DATA - mode B.  Renders a 320×frame_h BGR side-panel with:
      ULTRA-style header, FRAME / VEHICLES / PLATES / TRACKS sections and
      a TIMINGS (ms) block.  Falls back to a Hershey-rendered panel if the
      brand TTFs aren't installed locally.
    """
    if not _dd_have_pil_fonts():
        return _draw_hud_panel_fallback(frame_h, telemetry)

    from PIL import Image, ImageDraw

    hud = np.full((frame_h, _DD_HUD_W, 3), _DD_HUD_BG, dtype=np.uint8)
    # Left-edge accent bar (Data Blue)
    cv2.rectangle(hud, (0, 0), (3, frame_h), _DD_BLUE, -1)

    pil = Image.fromarray(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)).convert("RGBA")
    over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)

    fnt_display = _dd_font(_DD_F_DISP,  22, "Bold")
    fnt_body    = _dd_font(_DD_F_BODY,  20)
    fnt_mono    = _dd_font(_DD_F_MONO,  17)
    fnt_mono_b  = _dd_font(_DD_F_MONOB, 22)
    fnt_small   = _dd_font(_DD_F_DISP,  14, "Bold")

    # Cursor (vertical) and helpers — use a mutable container so nested fns
    # can advance the cursor without leaning on `nonlocal`.
    x0   = 18
    yp   = [18]

    # PIL uses RGB so colours need a flip from our BGR brand constants.
    def _rgb(bgr): return (bgr[2], bgr[1], bgr[0])
    BLUE_RGB  = _rgb(_DD_BLUE)  + (255,)
    RED_RGB   = _rgb(_DD_RED)   + (255,)
    WHITE_RGB = _rgb(_DD_WHITE) + (255,)
    DIM_RGB   = _rgb(_DD_DIM)   + (255,)
    GREEN_RGB = _rgb(_DD_GREEN) + (255,)

    def section(title, color):
        draw.text((x0, yp[0]), title, font=fnt_body, fill=color)
        yp[0] += 24

    def kv(label, value, value_color=WHITE_RGB):
        draw.text((x0, yp[0]),       label, font=fnt_mono,   fill=DIM_RGB)
        draw.text((x0 + 120, yp[0]), value, font=fnt_mono_b, fill=value_color)
        yp[0] += 22

    def bar(label, val_ms, max_ms, color):
        draw.text((x0,         yp[0]),     label,           font=fnt_mono,   fill=DIM_RGB)
        draw.text((x0 + 230,   yp[0] - 1), f"{val_ms:5.0f}", font=fnt_mono_b, fill=WHITE_RGB)
        bx, by, bw_, bh_ = x0 + 70, yp[0] + 5, 150, 10
        draw.rectangle([(bx, by), (bx + bw_, by + bh_)], fill=(40, 40, 50, 255))
        fill_w = int(bw_ * min(1.0, val_ms / max_ms))
        draw.rectangle([(bx, by), (bx + fill_w, by + bh_)], fill=color)
        yp[0] += 22

    # ── Header ─────────────────────────────────────────────────────────────
    draw.text((x0, yp[0]), "DEBUG  DATA", font=fnt_display, fill=WHITE_RGB)
    yp[0] += 32
    draw.rectangle([(x0, yp[0]), (_DD_HUD_W - 18, yp[0] + 2)], fill=BLUE_RGB)
    yp[0] += 18

    # ── FRAME ─────────────────────────────────────────────────────────────
    section("FRAME", BLUE_RGB)
    frame_idx    = telemetry.get("frame_num", 0)
    total_frames = telemetry.get("total_frames", 0)
    fps_target   = telemetry.get("fps_target", 30.0)
    timings      = telemetry.get("timings", {})
    ts_sec       = frame_idx / fps_target if fps_target else 0
    ts_min, ts_s = divmod(ts_sec, 60)
    total_ms     = sum(timings.values()) if timings else 0
    inst_fps     = (1000.0 / total_ms) if total_ms > 0 else 0
    kv("idx",  f"{frame_idx:05d} / {total_frames}")
    kv("time", f"{int(ts_min):02d}:{ts_s:06.3f}")
    kv("fps",  f"{inst_fps:5.1f}",
       value_color=GREEN_RGB if inst_fps >= fps_target * 0.5 else RED_RGB)
    yp[0] += 8

    # ── VEHICLES ──────────────────────────────────────────────────────────
    vehicles = telemetry.get("vehicles", [])
    tracks   = telemetry.get("tracks", [])
    section("VEHICLES", BLUE_RGB)
    kv("found",   f"{len(vehicles)}")
    kv("tracked", f"{len(tracks)}")
    yp[0] += 8

    # ── PLATES ────────────────────────────────────────────────────────────
    plates = telemetry.get("plates", [])
    by_src = {"sahi": 0, "crop": 0, "bridge": 0, "own": 0}
    for r in plates:
        s = r[5] if len(r) > 5 else "sahi"
        if s.startswith("crop"):
            s = "crop"
        by_src[s] = by_src.get(s, 0) + 1
    section("PLATES", BLUE_RGB)
    kv("obs", f"{by_src.get('sahi', 0) + by_src.get('crop', 0)}",
       value_color=WHITE_RGB if (by_src.get('sahi', 0) + by_src.get('crop', 0)) else DIM_RGB)
    kv("bridge", f"{by_src.get('bridge', 0)}",
       value_color=WHITE_RGB if by_src.get("bridge") else DIM_RGB)
    kv("own", f"{by_src.get('own', 0)}",
       value_color=WHITE_RGB if by_src.get("own") else DIM_RGB)
    yp[0] += 8

    # ── TRACKS list (up to 6 rows) ────────────────────────────────────────
    section("TRACKS", RED_RGB)
    if not tracks:
        draw.text((x0, yp[0]), "—  no active tracks",
                  font=fnt_mono, fill=DIM_RGB)
        yp[0] += 24
    else:
        for tid in tracks[:6]:
            label = f"#{getattr(tid, 'id', tid)}"
            draw.text((x0, yp[0]), label,
                      font=fnt_mono_b, fill=WHITE_RGB)
            yp[0] += 22
        if len(tracks) > 6:
            draw.text((x0, yp[0]), f"+ {len(tracks) - 6} more...",
                      font=fnt_mono, fill=DIM_RGB)
            yp[0] += 22
    yp[0] += 4

    # ── TIMINGS (ms) ──────────────────────────────────────────────────────
    section("TIMINGS (ms)", RED_RGB)
    detect_ms = timings.get("detect", 0)
    track_ms  = timings.get("track",  0)
    render_ms = timings.get("render", 0)
    max_ms    = max(60.0, detect_ms * 1.1)
    bar("detect", detect_ms, max_ms, RED_RGB)
    bar("track",  track_ms,  max_ms, RED_RGB)
    bar("render", render_ms, max_ms, RED_RGB)
    yp[0] += 4
    draw.rectangle([(x0, yp[0]), (_DD_HUD_W - 18, yp[0] + 1)],
                   fill=(70, 70, 80, 255))
    yp[0] += 8
    bar("TOTAL",  total_ms, max(80.0, total_ms * 1.1), BLUE_RGB)

    # ── Footer ────────────────────────────────────────────────────────────
    draw.text((x0, frame_h - 30), "DSDT.X / AI VISION",
              font=fnt_small, fill=DIM_RGB)

    pil.alpha_composite(over)
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def _draw_hud_panel_fallback(frame_h, telemetry):
    """OpenCV-only HUD when brand TTFs aren't available."""
    hud = np.full((frame_h, _DD_HUD_W, 3), _DD_HUD_BG, dtype=np.uint8)
    cv2.rectangle(hud, (0, 0), (3, frame_h), _DD_BLUE, -1)
    y = 30
    cv2.putText(hud, "DEBUG DATA", (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                _DD_WHITE, 2, cv2.LINE_AA)
    y += 26
    cv2.line(hud, (18, y), (_DD_HUD_W - 18, y), _DD_BLUE, 2); y += 18

    timings = telemetry.get("timings", {})
    total_ms = sum(timings.values()) if timings else 0
    for line in (
        f"frame  {telemetry.get('frame_num', 0):>5d}/{telemetry.get('total_frames', 0)}",
        f"VEH    {len(telemetry.get('vehicles', []))}",
        f"PLT    {len(telemetry.get('plates', []))}",
        f"TRK    {len(telemetry.get('tracks', []))}",
        "",
        f"detect {timings.get('detect', 0):6.1f} ms",
        f"track  {timings.get('track',  0):6.1f} ms",
        f"render {timings.get('render', 0):6.1f} ms",
        f"TOTAL  {total_ms:6.1f} ms",
    ):
        cv2.putText(hud, line, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                    _DD_WHITE, 1, cv2.LINE_AA)
        y += 24
    return hud
