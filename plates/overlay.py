# ─── Debug overlays (boxes, trajectories, HUD panel) ───────────────────────
import os

import cv2
import numpy as np

from plates.common import iou
from plates.constants import (
    VEHICLE_CLASSES,
    _DBG_BLUR_COLOR, _DBG_FONT, _DBG_GHOST_COLOR, _DBG_OWN_COLOR,
    _DBG_PLATE_COLOR, _DBG_PREDICT_COLOR, _DBG_SUPPRESSED_COLOR,
    _DBG_VEHICLE_COLOR,
    _DD_BLUE, _DD_DIM, _DD_F_BODY, _DD_F_DISP, _DD_F_MONO, _DD_F_MONOB,
    _DD_FONT_DIR, _DD_GHOST, _DD_GREEN, _DD_HUD_BG, _DD_HUD_LINE, _DD_HUD_W,
    _DD_ORANGE, _DD_RED, _DD_SOURCE_COLOR, _DD_TRAIL, _DD_VOID_BG,
    _DD_WHITE, _DD_YELLOW,
)
from plates.redact import apply_redaction


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
                       blur_padding=8, blur_strength=61, tracker=None,
                       suppressed_plates=None,
                       redact_mode="blur", redact_color=(0, 0, 0),
                       overlay_img=None):
    """
    Returns a debug frame that shows exactly what the production output will look like:
      - Redaction (blur / solid colour / image overlay) is applied to all detected
        regions, matching production output exactly
      - Blue box     = vehicle detection  (label: class conf | #id Nf detected)
      - Teal box     = tracked vehicle whose detector dropped this frame (label: gap N/M)
      - Green box    = raw plate detection boundary
      - Yellow box   = tracker-predicted plate (gap fill)
      - Red box      = padded redaction region (what was actually erased)
      - Orange box   = own-plate fixed region
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ── Step 1: apply the real redaction so the frame looks like production ──
    all_rects = list(plate_rects)
    if own_plate_region:
        all_rects.append(own_plate_region)
    if all_rects:
        vis = apply_redaction(vis, all_rects, mode=redact_mode,
                              blur_strength=blur_strength,
                              color=redact_color,
                              overlay_img=overlay_img,
                              padding=blur_padding)

    # ── Step 2: build a lookup of track data keyed by closest vehicle box ─────
    # Maps each track to its detected vehicle (if any) so we can annotate labels.
    track_by_vehicle = {}   # index into all_vehicles → VehicleTrack
    ghost_tracks     = []   # tracks whose vehicle wasn't detected this frame
    if tracker is not None:
        for track in tracker.tracks:
            if track.miss_count == 0:
                # Find the all_vehicles entry closest to this track's box
                best_vi, best_iou = None, 0.0
                for vi, (_, vx1, vy1, vx2, vy2, _) in enumerate(all_vehicles):
                    score = iou(track.box, (vx1, vy1, vx2, vy2))
                    if score > best_iou:
                        best_iou, best_vi = score, vi
                if best_vi is not None and best_iou > 0.1:
                    track_by_vehicle[best_vi] = track
            else:
                ghost_tracks.append(track)

    # ── Step 3: detected vehicle boxes (blue) ────────────────────────────────
    for vi, (cls, x1, y1, x2, y2, conf) in enumerate(all_vehicles):
        track = track_by_vehicle.get(vi)
        if track:
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f} | #{track.id} {track.frames_seen}f"
        else:
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
        _dbg_box(vis, x1, y1, x2, y2, _DBG_VEHICLE_COLOR, label, thickness=3)

    # ── Step 4: ghost vehicle boxes — tracked but not detected this frame ─────
    for track in ghost_tracks:
        x1, y1, x2, y2 = track.box
        label = f"#{track.id} gap {track.miss_count}/{tracker.max_gap_frames} | {track.frames_seen}f"
        _dbg_box(vis, x1, y1, x2, y2, _DBG_GHOST_COLOR, label, thickness=2)

    # ── Step 3: plate boxes — green (detected), yellow (predicted), red (blur) ─
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf = rect[4] if len(rect) > 4 else None

        if conf is not None and conf < 0:      # tracker gap fill (SAHI missed this frame)
            color = _DBG_PREDICT_COLOR
            label = "gap fill"
        else:
            color = _DBG_PLATE_COLOR
            label = f"plate {conf:.2f}" if conf is not None else "plate"

        # Colour-coded boundary
        _dbg_box(vis, x1, y1, x2, y2, color, label, thickness=2)

        # Red: padded region that was blurred
        px1 = max(0, x1 - blur_padding)
        py1 = max(0, y1 - blur_padding)
        px2 = min(w, x2 + blur_padding)
        py2 = min(h, y2 + blur_padding)
        cv2.rectangle(vis, (px1, py1), (px2, py2), _DBG_BLUR_COLOR, 4)

    # ── Step 4: own-plate fixed region (orange) ───────────────────────────────
    if own_plate_region:
        ox1, oy1, ox2, oy2 = own_plate_region
        _dbg_box(vis, ox1, oy1, ox2, oy2, _DBG_OWN_COLOR, "own plate", thickness=3)

    # ── Step 5: suppressed duplicates (grey, thin) ────────────────────────────
    for rect in (suppressed_plates or []):
        x1, y1, x2, y2 = rect[:4]
        conf  = rect[4] if len(rect) > 4 else None
        label = f"dup {conf:.2f}" if conf is not None else "dup"
        _dbg_box(vis, x1, y1, x2, y2, _DBG_SUPPRESSED_COLOR, label, thickness=1)

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

def draw_extended_overlay(frame, plate_rects, all_vehicles, tracker=None,
                          blur_padding=8, blur_strength=61,
                          redact_mode="blur", redact_color=(0, 0, 0),
                          overlay_img=None, rejected_plates=None):
    """
    DEBUG DATA - mode A.  Returns a frame with the actual redaction applied
    plus rich annotations: source-tagged plate boxes, vehicle boxes with
    track IDs, ghost tracks, plate trajectory trails, and a top HUD strip
    summarising what's in this frame.
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ── Apply real redaction so the user sees production output ───────────
    if plate_rects:
        vis = apply_redaction(vis, plate_rects,
                              mode=redact_mode,
                              blur_strength=blur_strength,
                              color=redact_color,
                              overlay_img=overlay_img,
                              padding=blur_padding)

    # ── Rejected detections (grey, dashed) — below confidence threshold ───
    # Drawn first so accepted/predicted boxes render on top of them.
    for rect in (rejected_plates or []):
        x1, y1, x2, y2 = rect[:4]
        conf = rect[4] if len(rect) > 4 else None
        _dd_dashed_rect(vis, x1, y1, x2, y2, _DD_GHOST, thickness=1)
        label = f"REJ {conf:.2f}" if conf is not None else "REJ"
        _dd_tag(vis, x1, y1, label, _DD_GHOST, fs=0.4)

    # ── Vehicle boxes (blue) with track-aware rich tag ────────────────────
    track_by_vidx = {}
    ghost_tracks  = []
    if tracker is not None:
        for tr in tracker.tracks:
            if tr.miss_count == 0:
                # Find closest current vehicle box for labelling
                best_vi, best_iou = None, 0.0
                for vi, (_, vx1, vy1, vx2, vy2, _) in enumerate(all_vehicles):
                    score = iou(tr.box, (vx1, vy1, vx2, vy2))
                    if score > best_iou:
                        best_iou, best_vi = score, vi
                if best_vi is not None and best_iou > 0.1:
                    track_by_vidx[best_vi] = tr
            else:
                ghost_tracks.append(tr)

    for vi, (cls, x1, y1, x2, y2, conf) in enumerate(all_vehicles):
        tr = track_by_vidx.get(vi)
        if tr:
            label = (f"{VEHICLE_CLASSES[cls]} {conf:.2f} | "
                     f"#TRK{tr.id} age:{tr.frames_seen}f miss:{tr.miss_count}")
        else:
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), _DD_BLUE, 2)
        _dd_tag(vis, x1, y1, label, _DD_BLUE)

    # ── Ghost tracks (detector missed this frame, tracker still holds) ────
    for tr in ghost_tracks:
        x1, y1, x2, y2 = tr.box
        label = f"GHOST #{tr.id} gap {tr.miss_count}/{tracker.max_gap_frames}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), _DD_GHOST, 1)
        _dd_tag(vis, x1, y1, label, _DD_GHOST, fs=0.45)

    # ── Plate trajectory trails — fading cyan line per track ──────────────
    if tracker is not None:
        for tr in tracker.tracks:
            if not tr.plate.has_history:
                continue
            # Use the centres of the recorded plate positions
            pts = [(int(d[1]), int(d[2])) for d in tr.plate._data[-15:]]
            for i in range(len(pts) - 1):
                alpha = 0.25 + 0.75 * (i / max(1, len(pts) - 1))
                c = tuple(int(v * alpha) for v in _DD_TRAIL)
                cv2.line(vis, pts[i], pts[i+1], c, 2, cv2.LINE_AA)

    # ── Plate boxes with source tag ───────────────────────────────────────
    src_counts = {"sahi": 0, "crop": 0, "pred": 0, "own": 0, "fallback": 0,
                  "anchor": 0}
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf   = rect[4] if len(rect) > 4 else None
        source = rect[5] if len(rect) > 5 else "sahi"
        src_counts[source] = src_counts.get(source, 0) + 1
        color = _DD_SOURCE_COLOR.get(source, _DD_GREEN)

        if source == "pred":
            _dd_dashed_rect(vis, x1, y1, x2, y2, color, thickness=2)
            _dd_tag(vis, x1, y2 + 20, "PRED  gap-fill", color,
                    fg_bgr=(0, 0, 0), fs=0.45)
        elif source == "fallback":
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            _dd_tag(vis, x1, y1, "FALLBACK  bottom strip", color,
                    fg_bgr=(0, 0, 0), fs=0.45)
        elif source == "anchor":
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            _dd_tag(vis, x1, y1, "ANCHOR  moto zone", color,
                    fg_bgr=(0, 0, 0), fs=0.45)
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
                  f"(SAHI {src_counts['sahi']}, crop+ {src_counts['crop']}, "
                  f"pred {src_counts['pred']}, own {src_counts['own']})   "
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
    by_src = {"sahi": 0, "crop": 0, "pred": 0, "own": 0, "fallback": 0,
              "anchor": 0}
    for r in plates:
        s = r[5] if len(r) > 5 else "sahi"
        by_src[s] = by_src.get(s, 0) + 1
    section("PLATES", BLUE_RGB)
    kv("SAHI",  f"{by_src['sahi']}",
       value_color=WHITE_RGB if by_src["sahi"] else DIM_RGB)
    kv("crop+", f"{by_src['crop']}",
       value_color=WHITE_RGB if by_src["crop"] else DIM_RGB)
    kv("pred",  f"{by_src['pred']}",
       value_color=WHITE_RGB if by_src["pred"] else DIM_RGB)
    kv("own",   f"{by_src['own']}",
       value_color=WHITE_RGB if by_src["own"] else DIM_RGB)
    yp[0] += 8

    # ── TRACKS list (up to 6 rows) ────────────────────────────────────────
    section("TRACKS", RED_RGB)
    if not tracks:
        draw.text((x0, yp[0]), "—  no active tracks",
                  font=fnt_mono, fill=DIM_RGB)
        yp[0] += 24
    else:
        for tr in tracks[:6]:
            draw.text((x0,        yp[0]), f"#{tr.id}",
                      font=fnt_mono_b, fill=WHITE_RGB)
            draw.text((x0 + 50,   yp[0]),
                      f"age {tr.frames_seen}f",
                      font=fnt_mono, fill=(180, 180, 200, 255))
            draw.text((x0 + 165,  yp[0]),
                      f"miss {tr.miss_count}",
                      font=fnt_mono, fill=(180, 180, 200, 255))
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
