# ─── Main redaction pipeline (frame loop) ──────────────────────────────────
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
from tqdm import tqdm

from plates.common import _overlaps
from plates.constants import (
    PLATE_MODEL_PATH,
    VEHICLE_CLASSES,
    VEHICLE_FILTER_MAP,
    _CUVID_DECODERS,
    _DD_HUD_W,
)
from plates.detcache import DetectionCache, build_meta
from plates.detect import detect_plates
from plates.ffmpeg import (
    build_ffmpeg_encode_lossless,
    build_ffmpeg_extract,
    estimate_frame_count,
    get_video_info,
    mux_audio,
)
from plates.models import load_models
from plates.overlay import draw_debug_overlay, draw_extended_overlay, draw_hud_panel
from plates.redact import (
    apply_blur_feathered,
    apply_blur_rotated,
    apply_blur_round,
    apply_redaction,
    estimate_plate_quad,
    load_overlay_image,
    _match_quad_state,
    _quad_state_update,
)
from plates.offline_zones import (
    build_offline_zones,
    merge_tracker_with_offline_fills,
    offline_zone_stats,
)
from plates.report import _print_run_summary
from plates.track import SceneTracker


def blur_license_plates(
    input_path: str,
    output_path: str,
    start_time: float = None,
    end_time: float = None,
    blur_strength: int = 61,
    blur_padding: int = 8,
    vehicle_conf: float = 0.3,
    plate_conf: float = 0.15,
    plate_conf_in_vehicle: float = 0.07,
    sahi_slice_size: int = 640,
    sahi_overlap: float = 0.2,
    own_plate_region: tuple = None,
    vehicle_filter: str = "all",
    preset: str = "medium",
    quality: int = 18,            # final HEVC quality (CRF / CQ) — see config.toml
    tmp_dir: str = "auto",
    debug: bool = False,
    debug_overlay: bool = False,   # extended in-frame overlay (DEBUG DATA mode)
    debug_hud: bool = False,       # side HUD panel (DEBUG DATA mode)
    tracking_enabled: bool = True,
    max_gap_frames: int = 8,
    history_frames: int = 15,
    min_vehicle_conf: float = 0.60,
    standalone_min_ar: float = 1.2,
    standalone_max_ar: float = 6.0,
    predict_expand_max: int = 20,
    detect_scale: float = 1.0,
    sharpen: bool = False,
    sharpen_amount: float = 1.5,
    sharpen_sigma: float = 1.0,
    vehicle_crop_scale: float = 1.0,
    moto_crop_scale: float = 3.0,
    moto_crop_bottom_frac: float = 0.28,
    moto_crop_side_pad_frac: float = 0.05,
    redact_mode: str = "blur",
    redact_color: tuple = (0, 0, 0),
    redact_image_path: str = None,
    max_box_frac: float = 0.15,
    predict_max_disp: float = 40.0,
    fallback_enabled: bool = True,
    fallback_frac: float = 0.40,
    fallback_pad_frac: float = 0.25,
    fallback_min_frames: int = 3,
    ema_alpha: float = 0.6,
    moto_ar_min: float = 0.9,
    moto_ar_max: float = 1.3,
    moto_anchor: bool = True,
    moto_anchor_frac: float = 0.45,
    moto_anchor_y: float = 0.70,
    moto_anchor_pad: float = 0.15,
    moto_ghost_frames: int = 6,
    moto_close_frac: float = 0.40,
    moto_close_conf: float = 0.20,
    moto_close_zone_w: float = 1.6,
    moto_edge_px: int = 4,
    moto_near_frac: float = 0.15,
    moto_zone_min_side: float = 40.0,
    moto_anchor_y_max: float = 0.72,
    moto_plate_conf: float = 0.30,
    moto_plate_promote_frames: int = 2,
    moto_plate_hold_frames: int = 15,
    moto_plate_pad: int = 10,
    moto_min_blur_box_h_frac: float = 0.10,
    moto_max_zone_box_frac: float = 0.35,
    moto_weak_fender_frac: float = 0.92,
    moto_quad_refine: bool = True,
    blur_shape: str = "rect",
    forced_fixlist: dict = None,
    emit_max_disp: float = 80.0,
    detect_cache: str = None,
    zone_filter: str = "ema",
    kf_params: dict = None,
    offline_zones: bool = False,
    offline_params: dict = None,
):
    if tmp_dir == "auto":
        tmp_dir = os.path.join(tempfile.gettempdir(), "plate-blur-tmp")
    # Capture wall-clock start so we can report total + processing-only time
    # at the end. Uses a different name from the `start_time` parameter (which
    # is the video trim start, not a timestamp).
    _run_start_ts = time.perf_counter()

    # Load the overlay image (if any) once, up front
    overlay_img = None
    if redact_mode == "image" and redact_image_path:
        overlay_img = load_overlay_image(redact_image_path)

    print(f"\n{'='*60}")
    print(f"  License Plate Redaction Tool")
    print(f"{'='*60}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    if start_time is not None:
        print(f"  Clip  : {start_time:.1f}s to {end_time:.1f}s")
    if vehicle_filter != "all":
        print(f"  Filter: {vehicle_filter} plates only")
    print(f"  Plate conf : {plate_conf} (global)  |  {plate_conf_in_vehicle} (inside vehicle boxes)")
    if own_plate_region:
        print(f"  Own plate region (always redacted): {own_plate_region}")
    print(f"  Mode  : {redact_mode}", end="")
    if redact_mode == "color":
        print(f"  (BGR {redact_color})")
    elif redact_mode == "image" and overlay_img is not None:
        print(f"  (overlay {redact_image_path})")
    else:
        print()
    if detect_scale < 1.0:
        print(f"  Detect : {detect_scale:.2f}× scale  (detection at {detect_scale*100:.0f}% res, blur at full res)")
    if sharpen:
        print(f"  Sharpen: enabled  (amount={sharpen_amount}, sigma={sharpen_sigma})")
    if vehicle_crop_scale > 1.0:
        print(f"  Crop upscale: {vehicle_crop_scale:.1f}×  (car/truck plate pass)")
    if moto_crop_scale > 1.0:
        print(f"  Moto crop  : {moto_crop_scale:.1f}x rear ROI "
              f"(y>={moto_crop_bottom_frac:.2f}, pad={moto_crop_side_pad_frac:.2f})")
    if debug:
        print(f"  Mode  : DEBUG (blur applied + detection overlay)")
    if offline_zones:
        print(f"  Zones : hybrid (online tracker + offline gap fills)")
    if tracking_enabled:
        print(f"  Track : enabled  (gap={max_gap_frames} frames, history={history_frames}, "
              f"expand_max={predict_expand_max}px)")
    print(f"{'='*60}\n")

    info = get_video_info(input_path)
    width, height, fps = info["width"], info["height"], info["fps"]
    print(f"  Video : {width}x{height} @ {fps:.3f}fps  [{info['codec']}]")

    cache = None
    if detect_cache:
        cache = DetectionCache(detect_cache, build_meta(
            input                 = os.path.basename(input_path),
            input_size            = os.path.getsize(input_path),
            start                 = start_time,
            end                   = end_time,
            width                 = width,
            height                = height,
            vehicle_conf          = vehicle_conf,
            vehicle_filter        = vehicle_filter,
            plate_conf            = plate_conf,
            plate_conf_in_vehicle = plate_conf_in_vehicle,
            sahi_slice_size       = sahi_slice_size,
            sahi_overlap          = sahi_overlap,
            detect_scale          = detect_scale,
            sharpen               = sharpen,
            sharpen_amount        = sharpen_amount,
            sharpen_sigma         = sharpen_sigma,
            vehicle_crop_scale    = vehicle_crop_scale,
            moto_crop_scale       = moto_crop_scale,
            moto_crop_bottom_frac = moto_crop_bottom_frac,
            moto_crop_side_pad_frac = moto_crop_side_pad_frac,
            moto_close_conf       = moto_close_conf,
            plate_model           = os.path.basename(PLATE_MODEL_PATH),
        ))
        if cache.reading:
            print(f"  Cache : reading detections for {cache.frames} frames from "
                  f"{os.path.basename(detect_cache)}"
                  + ("  [PARTIAL — write was interrupted]" if cache.partial else ""))
        else:
            print(f"  Cache : writing detections to {os.path.basename(detect_cache)}")

    offline_zones_map = None
    if offline_zones:
        if cache is None or not cache.reading:
            raise ValueError(
                "--offline-zones requires an existing --detect-cache JSONL "
                "(build the cache first, then re-run with --offline-zones)"
            )
        _off = dict(offline_params or {})
        offline_zones_map = build_offline_zones(
            cache,
            max_gap_frames=int(_off.get("max_gap_frames", 15)),
            max_disp_px=float(_off.get("max_disp_px", 80.0)),
            max_disp_frac=float(_off.get("max_disp_frac", 0.35)),
            conf_floor=float(_off.get("conf_floor", 0.15)),
            moto_only=bool(_off.get("moto_only", True)),
            min_blur_box_h_frac=float(
                _off.get("min_blur_box_h_frac", moto_min_blur_box_h_frac)
            ),
            frame_height=height,
            min_vehicle_iou=float(_off.get("min_vehicle_iou", 0.15)),
        )
        _st = offline_zone_stats(offline_zones_map)
        print(f"  Offline zones: {_st['frames_with_zones']} frames, "
              f"{_st['raw_zones']} raw + {_st['bridged_zones']} bridged")

    # Reading a cache replays stored detections, so the detector is never called
    # and loading it would only cost startup time and memory.
    if cache is not None and cache.reading:
        vehicle_model = plate_model = None
        device = "cpu"
        print("  Models skipped  |  detections replayed from cache\n")
    else:
        print("  Loading models...")
        vehicle_model, plate_model, device = load_models(
            plate_conf=min(plate_conf, plate_conf_in_vehicle)
        )
        print(f"  Models ready  |  vehicle detector + license plate detector\n")

    tracker = SceneTracker(
        max_gap_frames=max_gap_frames,
        history_frames=history_frames,
        min_vehicle_conf=min_vehicle_conf,
        standalone_min_ar=standalone_min_ar,
        standalone_max_ar=standalone_max_ar,
        predict_expand_max=predict_expand_max,
        predict_max_disp=predict_max_disp,
        fallback_enabled=fallback_enabled,
        fallback_frac=fallback_frac,
        fallback_pad_frac=fallback_pad_frac,
        fallback_min_frames=fallback_min_frames,
        ema_alpha=ema_alpha,
        moto_ar_min=moto_ar_min,
        moto_ar_max=moto_ar_max,
        moto_anchor=moto_anchor,
        moto_anchor_frac=moto_anchor_frac,
        moto_anchor_y=moto_anchor_y,
        moto_anchor_pad=moto_anchor_pad,
        moto_ghost_frames=moto_ghost_frames,
        moto_close_frac=moto_close_frac,
        moto_close_conf=moto_close_conf,
        moto_close_zone_w=moto_close_zone_w,
        moto_edge_px=moto_edge_px,
        moto_near_frac=moto_near_frac,
        moto_zone_min_side=moto_zone_min_side,
        moto_anchor_y_max=moto_anchor_y_max,
        moto_plate_conf=moto_plate_conf,
        moto_plate_promote_frames=moto_plate_promote_frames,
        moto_plate_hold_frames=moto_plate_hold_frames,
        moto_plate_pad=moto_plate_pad,
        moto_min_blur_box_h_frac=moto_min_blur_box_h_frac,
        moto_max_zone_box_frac=moto_max_zone_box_frac,
        moto_weak_fender_frac=moto_weak_fender_frac,
        emit_max_disp=emit_max_disp,
        zone_filter=zone_filter,
        **(kf_params or {}),
    ) if tracking_enabled else None

    frame_size     = width * height * 3
    total_frames   = estimate_frame_count(info, start_time, end_time)

    # When the HUD side-panel is enabled, output frames are wider than input.
    # The encoder is told this widened size; ffmpeg's `-s WxH` reads the raw
    # buffer at the new dimensions so the side panel survives encoding.
    enc_width  = width + _DD_HUD_W if debug_hud else width
    enc_height = height

    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False, dir=tmp_dir) as tmp:
        tmp_path = tmp.name

    try:
        # CUVID pre-check: probe one decoded frame to detect unsupported codec
        # profiles (e.g. iPhone 10-bit HEVC, some MOV variants).  CUVID failure
        # is silent — it produces 0 bytes — which would otherwise create an
        # empty intermediate and an unplayable output file.
        video_codec = info.get("codec")
        if video_codec in _CUVID_DECODERS:
            _probe_end = (start_time or 0.0) + 1.0 / fps
            _probe = subprocess.Popen(
                build_ffmpeg_extract(input_path, start_time, _probe_end, codec=video_codec),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            _probe_ok = len(_probe.stdout.read(frame_size)) >= frame_size
            _probe.stdout.close()
            _probe.wait()
            if not _probe_ok:
                print(f"  Warning: {_CUVID_DECODERS[video_codec]} decode failed for this file "
                      f"— falling back to software decode")
                video_codec = None   # None → build_ffmpeg_extract uses software path

        extract_cmd = build_ffmpeg_extract(input_path, start_time, end_time, codec=video_codec)
        encode_cmd  = build_ffmpeg_encode_lossless(enc_width, enc_height, fps, tmp_path)

        extract_proc = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
        encode_proc  = subprocess.Popen(encode_cmd,  stdin=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)

        frame_num    = 0
        total_plates = 0
        total_quads  = 0
        quad_hist    = []   # temporal quad state: [{rect, quad, ratio, active}]
        _process_start_ts = time.perf_counter()

        with tqdm(total=total_frames, unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames "
                             "[{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            try:
                while True:
                    raw = extract_proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (height, width, 3)).copy()

                    # Per-frame timing dict for the debug HUD's TIMINGS bars.
                    # Each stage is wrapped with perf_counter() deltas in ms.
                    timings = {}
                    _t0 = time.perf_counter()

                    suppressed_plates = []   # reset each frame; populated by tracker or dedup
                    cached = cache.get(frame_num) if (cache is not None
                                                      and cache.reading) else None
                    if cached is not None:
                        plates, vehicles = cached
                        rejected_plates  = []   # not cached: debug-overlay only
                    elif cache is not None and cache.reading:
                        break   # cache exhausted (partial write) — stop cleanly
                    else:
                        plates, vehicles, rejected_plates = detect_plates(
                            frame, vehicle_model, plate_model,
                            vehicle_conf=vehicle_conf,
                            vehicle_conf_floor=moto_close_conf,
                            vehicle_filter=vehicle_filter,
                            plate_conf=plate_conf,
                            plate_conf_in_vehicle=plate_conf_in_vehicle,
                            sahi_slice_size=sahi_slice_size,
                            sahi_overlap=sahi_overlap,
                            detect_scale=detect_scale,
                            sharpen=sharpen,
                            sharpen_amount=sharpen_amount,
                            sharpen_sigma=sharpen_sigma,
                            vehicle_crop_scale=vehicle_crop_scale,
                            moto_crop_scale=moto_crop_scale,
                            moto_crop_bottom_frac=moto_crop_bottom_frac,
                            moto_crop_side_pad_frac=moto_crop_side_pad_frac,
                            collect_rejected=debug_overlay,
                        )
                        if cache is not None:
                            cache.put(frame_num, plates, vehicles)

                    timings["detect"] = (time.perf_counter() - _t0) * 1000
                    _t1 = time.perf_counter()

                    _dbg = os.environ.get("AUTOCLIP_DEBUG_DETECT", "")
                    if _dbg:
                        try:
                            _dbg_from, _dbg_to, _dbg_tgt = _dbg.split(",")
                            if int(_dbg_from) <= frame_num <= int(_dbg_to):
                                print(f"DBG f={frame_num} plates=" +
                                      "; ".join(f"({p[0]},{p[1]},{p[2]},{p[3]})c={p[4]:.2f}" for p in plates) +
                                      " veh=" +
                                      "; ".join(f"cls={v[0]}({v[1]},{v[2]},{v[3]},{v[4]})c={v[5]:.2f}"
                                                for v in vehicles[:14]), flush=True)
                        except Exception:
                            pass

                    filter_classes  = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
                    vehicle_boxes   = [(v[1], v[2], v[3], v[4]) for v in vehicles
                                       if v[0] in filter_classes]

                    if tracker is not None:
                        # Tracker owns dedup: it uses plate history to gate out
                        # spatially inconsistent false positives before recording.
                        # suppress_duplicate_plates is intentionally skipped here
                        # so the tracker sees all candidates, not just the
                        # highest-confidence one chosen without temporal context.
                        plates, suppressed_plates = tracker.update(
                            vehicles, plates, frame,
                            frame_size=(width, height),
                        )
                        if offline_zones_map is not None:
                            plates = merge_tracker_with_offline_fills(
                                plates, offline_zones_map.get(frame_num),
                            )
                        if tracker.debug_zones:
                            for p in plates:
                                if len(p) > 5 and p[5] == "anchor":
                                    z = p[:4]
                                    if redact_mode == "blur" and moto_quad_refine:
                                        q = estimate_plate_quad(frame, z)
                                        zarea = (z[2] - z[0]) * (z[3] - z[1])
                                        qinfo = "quad=none"
                                        if q is not None and zarea > 0:
                                            qinfo = f"quad={cv2.contourArea(q)/zarea:.2f}"
                                    else:
                                        qinfo = "quad=off"
                                    print(f"[ZONE] f={frame_num} conf={p[4]:.2f} "
                                          f"zone=({z[0]:.0f},{z[1]:.0f},{z[2]:.0f},{z[3]:.0f}) "
                                          f"{qinfo}", flush=True)
                    elif offline_zones_map is not None:
                        # Tracking disabled but offline requested: bridges only
                        # on empty frames (no raw offline replacements).
                        plates = merge_tracker_with_offline_fills(
                            [], offline_zones_map.get(frame_num),
                        )
                        suppressed_plates = []
                    else:
                        # Tracking disabled: AR filter only — no suppression
                        vehicle_zones = vehicle_boxes
                        ar_filtered = []
                        for p in plates:
                            if any(_overlaps(p[:4], z) for z in vehicle_zones):
                                ar_filtered.append(p)
                            else:
                                x1, y1, x2, y2 = p[:4]
                                h = y2 - y1
                                if h > 0 and standalone_min_ar <= (x2-x1)/h <= standalone_max_ar:
                                    ar_filtered.append(p)
                        plates = ar_filtered

                    # Always include own-plate region
                    timings["track"] = (time.perf_counter() - _t1) * 1000
                    _t2 = time.perf_counter()

                    if own_plate_region:
                        # Tag own-plate with conf 1.0 and source 'own' so the
                        # debug overlay can label / colour it distinctly.
                        own_with_source = (*own_plate_region, 1.0, "own")
                        plates = list(plates) + [own_with_source]

                    if plates:
                        total_plates += len(plates)

                    if debug_overlay:
                        frame = draw_extended_overlay(
                            frame, plates, vehicles,
                            tracker=tracker,
                            blur_padding=blur_padding,
                            blur_strength=blur_strength,
                            redact_mode=redact_mode,
                            redact_color=redact_color,
                            overlay_img=overlay_img,
                            rejected_plates=rejected_plates,
                        )
                    elif debug:
                        frame = draw_debug_overlay(frame, plates, vehicles,
                                                   own_plate_region=own_plate_region,
                                                   blur_padding=blur_padding,
                                                   blur_strength=blur_strength,
                                                   tracker=tracker,
                                                   suppressed_plates=suppressed_plates,
                                                   redact_mode=redact_mode,
                                                   redact_color=redact_color,
                                                   overlay_img=overlay_img)
                    elif plates:
                        redact_rects = []
                        redact_quads = []
                        for p in plates:
                            if (moto_quad_refine and redact_mode == "blur"
                                    and len(p) > 5 and p[5] == "anchor"):
                                quad = estimate_plate_quad(frame, p[:4])
                                # Safety guard: the quad replaces the zone only
                                # when it covers most of it — a partial contour
                                # (leaning plate, motion blur, cut edges) would
                                # otherwise shrink the blur to half a plate.
                                # Otherwise fall back to the guaranteed rect.
                                zarea = ((p[2] - p[0]) * (p[3] - p[1]))
                                use_quad = False
                                q_blend = quad
                                i = _match_quad_state(quad_hist, p[:4])
                                if i >= 0:
                                    q_blend, use_quad = _quad_state_update(
                                        quad_hist[i], p[:4], quad, zarea)
                                elif quad is not None and zarea > 0:
                                    qc = quad.mean(axis=0)
                                    zcx = (p[0] + p[2]) * 0.5
                                    zcy = (p[1] + p[3]) * 0.5
                                    zside = max(p[2] - p[0], p[3] - p[1], 1.0)
                                    # Plate prior: only engage on a shape near
                                    # the zone centre (where the plate must
                                    # sit) — not on a top case above it.
                                    near_centre = (np.hypot(qc[0] - zcx,
                                                            qc[1] - zcy)
                                                   <= 0.30 * zside)
                                    ratio = cv2.contourArea(quad) / zarea
                                    use_quad = ratio >= 0.75 and near_centre
                                    quad_hist.append({"rect": p[:4],
                                                      "quad": quad,
                                                      "ratio": ratio,
                                                      "active": use_quad,
                                                      "fails": 0,
                                                      "fallback": 0})
                                if use_quad:
                                    redact_quads.append(q_blend)
                                    continue
                                if i >= 0 and quad_hist[i]["quad"] is not None \
                                        and quad_hist[i]["fallback"] > 0:
                                    # Memory fallback: after the quad is given
                                    # up, keep an anchor-sized rect centred on
                                    # the last plate position. The geometric
                                    # anchor can sit low (leaning/close moto)
                                    # and leave the plate visible above it.
                                    stq = quad_hist[i]
                                    qq = stq["quad"]
                                    qcx = (float(qq[:, 0].min()) + float(qq[:, 0].max())) * 0.5
                                    qcy = (float(qq[:, 1].min()) + float(qq[:, 1].max())) * 0.5
                                    rw = p[2] - p[0]
                                    rh = p[3] - p[1]
                                    fh2, fw2 = frame.shape[:2]
                                    mem = (int(qcx - rw * 0.5), int(qcy - rh * 0.5),
                                           int(qcx + rw * 0.5), int(qcy + rh * 0.5))
                                    mem = (max(0, mem[0]), max(0, mem[1]),
                                           min(fw2, mem[2]), min(fh2, mem[3]))
                                    redact_rects.append(mem)
                                    continue
                            redact_rects.append(p)
                        total_quads += len(redact_quads)
                        if redact_rects:
                            if blur_shape == "round":
                                frame = apply_blur_round(frame, redact_rects,
                                                         blur_strength=blur_strength,
                                                         padding=blur_padding)
                            else:
                                frame = apply_redaction(frame, redact_rects,
                                                        mode=redact_mode,
                                                        blur_strength=blur_strength,
                                                        color=redact_color,
                                                        overlay_img=overlay_img,
                                                        padding=blur_padding,
                                                        max_box_frac=max_box_frac)
                        if redact_quads:
                            if blur_shape == "round":
                                # round shape: ellipse the quad's bounding rect
                                qrects = [(int(q[:, 0].min()), int(q[:, 1].min()),
                                           int(q[:, 0].max()), int(q[:, 1].max()))
                                          for q in redact_quads]
                                frame = apply_blur_round(frame, qrects,
                                                         blur_strength=blur_strength,
                                                         padding=blur_padding)
                            else:
                                frame = apply_blur_rotated(frame, redact_quads,
                                                           blur_strength=blur_strength,
                                                           padding=blur_padding)

                    timings["render"] = (time.perf_counter() - _t2) * 1000

                    # ── Forced audit fixes ─────────────────────────────────
                    # Frames the audit pass flagged as still readable get a
                    # guaranteed rectangular blur from the fixlist, applied
                    # AFTER every other redaction so it always covers.
                    forced = forced_fixlist.get(frame_num) if forced_fixlist else None
                    if forced:
                        fixed_rects = [(*r, 0.99) for r in forced]
                        frame = apply_blur_feathered(frame, fixed_rects,
                                                     blur_strength=blur_strength,
                                                     padding=blur_padding + 4)

                    # HUD side panel (DEBUG DATA mode B) — appended to the right
                    if debug_hud:
                        hud = draw_hud_panel(
                            frame_h    = frame.shape[0],
                            telemetry  = {
                                "frame_num"   : frame_num,
                                "total_frames": total_frames,
                                "fps_target"  : fps,
                                "vehicles"    : vehicles,
                                "plates"      : plates,
                                "tracks"      : tracker.tracks if tracker else [],
                                "timings"     : timings,
                            },
                        )
                        frame = np.concatenate([frame, hud], axis=1)

                    encode_proc.stdin.write(frame.tobytes())
                    frame_num += 1

                    pbar.update(1)
                    pbar.set_postfix(plates=total_plates, refresh=False)

            finally:
                extract_proc.stdout.close()
                extract_proc.wait()
                encode_proc.stdin.close()
                encode_proc.wait()
                if cache is not None:
                    cache.close()

        # Wall-clock processing time = detection + draw loop, before muxing.
        _elapsed_process = time.perf_counter() - _process_start_ts

        action = "annotated" if debug else "redacted"
        print(f"\n  Processed : {frame_num} frames")
        print(f"  Detections: {total_plates} plate regions {action}")

        print("  Encoding final output (lossless HEVC + audio sync fix)...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mux_audio(tmp_path, input_path, output_path, start_time, end_time, fps,
                  total_frames=frame_num, preset=preset, tmp_dir=tmp_dir,
                  quality=quality)

        # End-of-run summary: hardware, timing, throughput, mode, settings.
        _elapsed_total = time.perf_counter() - _run_start_ts
        _print_run_summary(
            elapsed_total   = _elapsed_total,
            elapsed_process = _elapsed_process,
            device          = device,
            info            = info,
            input_path      = input_path,
            output_path     = output_path,
            frame_num       = frame_num,
            total_plates    = total_plates,
            total_quads     = total_quads,
            redact_mode     = redact_mode,
            redact_color    = redact_color,
            redact_image_path = redact_image_path,
            vehicle_filter  = vehicle_filter,
        )
        return frame_num, total_plates

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
