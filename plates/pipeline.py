# ─── Main redaction pipeline (detect → interpolate → render) ───────────────
import os
import subprocess
import tempfile
import time

import numpy as np
from tqdm import tqdm

from plates.common import _clamp_rect_max
from plates.constants import (
    PLATE_MODEL_PATH,
    VEHICLE_CLASSES,
    VEHICLE_FILTER_MAP,
    VEHICLE_MODEL_PATH,
    _CUVID_DECODERS,
    _DD_HUD_W,
)
from plates.detcache import DetectionCache, DetectionStore, build_meta
from plates.detect import detect_plates, MotoConfCache
from plates.ffmpeg import (
    build_ffmpeg_encode_lossless,
    build_ffmpeg_extract,
    estimate_frame_count,
    get_video_info,
    mux_audio,
)
from plates.models import load_models
from plates.overlay import (
    _is_base_zone,
    draw_debug_overlay,
    draw_extended_overlay,
    draw_hud_panel,
)
from plates.redact import (
    apply_blur_feathered,
    apply_blur_circle,
    apply_redaction,
    load_overlay_image,
)
from plates.report import _print_run_summary
from plates.moto_geom import MotoGeomSmoother, add_moto_base_zones
from plates.track import MotoSizeGate, build_zones, hold_vehicles, zone_stats


def _probe_cuvid(input_path, start_time, fps, video_codec, frame_size):
    """Return *video_codec* or None if CUVID cannot decode this file."""
    if video_codec not in _CUVID_DECODERS:
        return video_codec
    probe_end = (start_time or 0.0) + 1.0 / fps
    probe = subprocess.Popen(
        build_ffmpeg_extract(input_path, start_time, probe_end, codec=video_codec),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    ok = len(probe.stdout.read(frame_size)) >= frame_size
    probe.stdout.close()
    probe.wait()
    if ok:
        return video_codec
    print(f"  Warning: {_CUVID_DECODERS[video_codec]} decode failed for this file "
          f"-- falling back to software decode")
    return None


def _open_extract(input_path, start_time, end_time, codec):
    cmd = build_ffmpeg_extract(input_path, start_time, end_time, codec=codec)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _read_frame(extract_proc, frame_size, height, width):
    raw = extract_proc.stdout.read(frame_size)
    if len(raw) < frame_size:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


def _close_extract(extract_proc):
    extract_proc.stdout.close()
    extract_proc.wait()


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
    quality: int = 18,
    tmp_dir: str = "auto",
    debug: bool = False,
    debug_overlay: bool = False,
    debug_hud: bool = False,
    detect_scale: float = 1.0,
    sharpen: bool = False,
    sharpen_amount: float = 1.5,
    sharpen_sigma: float = 1.0,
    vehicle_crop_scale: float = 1.0,
    moto_crop_scale: float = 2.0,
    moto_crop_bottom_frac: float = 0.28,
    moto_crop_side_pad_frac: float = 0.05,
    plate_crop_imgsz: int = 1280,
    crop_clahe: bool = False,
    crop_clahe_clip: float = 2.0,
    crop_clahe_grid: int = 8,
    moto_close_conf: float = 0.20,
    moto_min_conf: float = 0.30,
    redact_mode: str = "blur",
    redact_color: tuple = (0, 0, 0),
    redact_image_path: str = None,
    max_box_frac: float = 0.15,
    forced_fixlist: dict = None,
    detect_cache: str = None,
    zone_params: dict = None,
):
    if tmp_dir == "auto":
        tmp_dir = os.path.join(tempfile.gettempdir(), "plate-blur-tmp")
    _run_start_ts = time.perf_counter()

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
    if crop_clahe:
        print(f"  Crop CLAHE: clip={crop_clahe_clip:.1f} grid={int(crop_clahe_grid)}")
    if sharpen:
        print(f"  Sharpen: enabled on vehicle crops (amount={sharpen_amount}, sigma={sharpen_sigma})")
    if vehicle_crop_scale > 1.0:
        print(f"  Crop canvas: {int(plate_crop_imgsz)} px  max {vehicle_crop_scale:.1f}x "
              f"(letterbox, same aspect)")
    if moto_crop_scale > 1.0:
        print(f"  Moto crop  : {moto_crop_scale:.1f}x rear ROI "
              f"(y>={moto_crop_bottom_frac:.2f}, pad={moto_crop_side_pad_frac:.2f})")
    if debug:
        print(f"  Mode  : DEBUG (blur applied + detection overlay)")
    print(f"  Zones : interpolate plates inside vehicles from the full cache")
    if moto_min_conf > 0:
        print(f"  Moto min conf: {moto_min_conf:.2f} (YOLO floor still {moto_close_conf:.2f})")
    print(f"{'='*60}\n")

    info = get_video_info(input_path)
    width, height, fps = info["width"], info["height"], info["fps"]
    print(f"  Video : {width}x{height} @ {fps:.3f}fps  [{info['codec']}]")

    meta = build_meta(
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
        plate_crop_imgsz      = int(plate_crop_imgsz),
        crop_clahe            = bool(crop_clahe),
        crop_clahe_clip       = float(crop_clahe_clip),
        crop_clahe_grid       = int(crop_clahe_grid),
        moto_close_conf       = moto_close_conf,
        plate_model           = os.path.basename(PLATE_MODEL_PATH),
        vehicle_model         = os.path.basename(VEHICLE_MODEL_PATH),
    )
    if detect_cache:
        cache = DetectionCache(detect_cache, meta)
        if cache.reading and cache.partial:
            print(f"  Cache : incomplete {os.path.basename(detect_cache)} "
                  f"({cache.frames} frames) -- re-detecting from scratch")
            cache.close()
            os.remove(detect_cache)
            cache = DetectionCache(detect_cache, meta)
        if cache.reading:
            print(f"  Cache : reading detections for {cache.frames} frames from "
                  f"{os.path.basename(detect_cache)}")
        else:
            print(f"  Cache : writing detections to {os.path.basename(detect_cache)}")
    else:
        cache = DetectionStore()
        print("  Cache : in-memory (pass --detect-cache PATH to reuse detections)")

    frame_size = width * height * 3
    total_frames = estimate_frame_count(info, start_time, end_time)
    video_codec = _probe_cuvid(input_path, start_time, fps, info.get("codec"), frame_size)

    vehicle_model = plate_model = None
    device = "cpu"
    need_detect = not cache.reading
    if need_detect:
        print("  Loading models...")
        vehicle_model, plate_model, device = load_models(
            plate_conf=min(plate_conf, plate_conf_in_vehicle)
        )
        print(f"  Models ready  |  vehicle detector + license plate detector\n")
        print("  Pass 1/2: detecting plates...")
        extract_proc = _open_extract(input_path, start_time, end_time, video_codec)
        frame_num = 0
        try:
            with tqdm(total=total_frames, unit="frame", dynamic_ncols=True,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames "
                                 "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                while True:
                    frame = _read_frame(extract_proc, frame_size, height, width)
                    if frame is None:
                        break
                    plates, vehicles, _rej = detect_plates(
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
                        plate_crop_imgsz=plate_crop_imgsz,
                        crop_clahe=crop_clahe,
                        crop_clahe_clip=crop_clahe_clip,
                        crop_clahe_grid=crop_clahe_grid,
                        moto_min_conf=moto_min_conf,
                    )
                    cache.put(frame_num, plates, vehicles)
                    frame_num += 1
                    pbar.update(1)
        finally:
            _close_extract(extract_proc)
        cache.finish_write()
        print(f"  Detected {cache.frames} frames\n")
        vehicle_model = plate_model = None
    else:
        print("  Models skipped  |  detections replayed from cache\n")

    zp = dict(zone_params or {})
    min_moto_h_frac = float(zp.get("min_moto_h_frac", 0.08333))
    moto_size_enter_frac = float(zp.get("moto_size_enter_frac", 1.15))
    moto_size_exit_frac = float(zp.get("moto_size_exit_frac", 0.85))
    min_vehicle_iou = float(zp.get("min_vehicle_iou", 0.15))
    max_gap_frames = int(zp.get("max_gap_frames", 30))
    max_disp_px = float(zp.get("max_disp_px", 80.0))
    max_disp_frac = float(zp.get("max_disp_frac", 0.35))
    max_class_flip_frames = int(zp.get("max_class_flip_frames", 10))
    min_plate_side_px = float(zp.get("min_plate_side_px", 12.0))
    max_area_ratio = float(zp.get("max_area_ratio", 2.5))
    moto_min_conf = float(zp.get("moto_min_conf", moto_min_conf))
    zone_cache = (
        MotoConfCache(cache, moto_min_conf)
        if moto_min_conf > 0 else cache
    )
    filter_classes = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
    vehicles_held = hold_vehicles(
        zone_cache,
        max_gap_frames=max_gap_frames,
        min_iou=min_vehicle_iou,
        max_disp_px=max_disp_px,
        max_disp_frac=max_disp_frac,
        max_class_flip_frames=max_class_flip_frames,
        moto_min_conf=moto_min_conf,
    )
    zones_map = build_zones(
        zone_cache,
        max_gap_frames=max_gap_frames,
        max_disp_px=max_disp_px,
        max_disp_frac=max_disp_frac,
        conf_floor=float(zp.get("conf_floor", plate_conf_in_vehicle)),
        filter_classes=filter_classes,
        min_moto_h_frac=min_moto_h_frac,
        frame_height=height,
        min_vehicle_iou=min_vehicle_iou,
        moto_size_enter_frac=moto_size_enter_frac,
        moto_size_exit_frac=moto_size_exit_frac,
        min_plate_side_px=min_plate_side_px,
        max_class_flip_frames=max_class_flip_frames,
        max_area_ratio=max_area_ratio,
    )
    _st = zone_stats(zones_map)
    print(f"  Zones : {_st['frames_with_zones']} frames, "
          f"{_st['observed_zones']} observed + {_st['bridged_zones']} interpolated")
    moto_base_blur = bool(zp.get("moto_base_blur", False))
    moto_base_if_no_plate = bool(zp.get("moto_base_if_no_plate", True))
    moto_base_kw = dict(
        height_frac=float(zp.get("moto_base_height_frac", 1.0 / 3.0)),
        min_height=float(zp.get("moto_base_min_height", 22.0)),
    )
    moto_smooth_alpha = float(zp.get("moto_base_smooth_alpha", 0.22))
    moto_smooth_alpha_pos = float(zp.get("moto_base_smooth_alpha_pos", 0.35))
    geom_smoother = (
        MotoGeomSmoother(
            alpha=moto_smooth_alpha,
            alpha_pos=moto_smooth_alpha_pos,
            min_iou=min_vehicle_iou,
        )
        if moto_base_blur else None
    )
    if moto_base_blur:
        print("  Moto base: circle h/3, centre h/2 from box bottom "
              f"EMA h a={moto_smooth_alpha:.2f} pos a={moto_smooth_alpha_pos:.2f}"
              + (" (only if no plate hit)" if moto_base_if_no_plate else ""))
    print("  Pass 2/2: applying blur...\n" if need_detect else
          "  Applying interpolated zones...\n")

    enc_width = width + _DD_HUD_W if debug_hud else width
    enc_height = height
    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False, dir=tmp_dir) as tmp:
        tmp_path = tmp.name

    try:
        extract_proc = _open_extract(input_path, start_time, end_time, video_codec)
        encode_cmd = build_ffmpeg_encode_lossless(enc_width, enc_height, fps, tmp_path)
        encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL)

        frame_num = 0
        total_plates = 0
        total_quads = 0
        _process_start_ts = time.perf_counter()
        overlay_gate = MotoSizeGate(
            min_size=min_moto_h_frac * float(height),
            enter_frac=moto_size_enter_frac,
            exit_frac=moto_size_exit_frac,
            min_iou=min_vehicle_iou,
        )

        with tqdm(total=total_frames, unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames "
                             "[{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            try:
                while True:
                    frame = _read_frame(extract_proc, frame_size, height, width)
                    if frame is None:
                        break

                    timings = {}
                    _t0 = time.perf_counter()
                    cache_fi = cache.replay_index(frame_num, fps)
                    cached = zone_cache.get(cache_fi)
                    if cached is None and cache.reading:
                        break
                    vehicles = vehicles_held.get(
                        cache_fi,
                        cached[1] if cached is not None else [],
                    )
                    observed = [
                        p for p in (cached[0] if cached is not None else [])
                        if (p[5] if len(p) > 5 else "crop") != "sahi"
                    ]
                    moto_large_boxes = overlay_gate.update(vehicles)
                    timings["detect"] = (time.perf_counter() - _t0) * 1000

                    _t1 = time.perf_counter()
                    plates = list(zones_map.get(cache_fi, []))
                    plates = add_moto_base_zones(
                        frame, vehicles, plates, moto_large_boxes,
                        enabled=moto_base_blur,
                        only_if_no_plate=moto_base_if_no_plate,
                        smoother=geom_smoother,
                        **moto_base_kw,
                    )
                    timings["track"] = (time.perf_counter() - _t1) * 1000
                    _t2 = time.perf_counter()

                    if own_plate_region:
                        plates = list(plates) + [(*own_plate_region, 1.0, "own")]

                    if plates:
                        total_plates += len(plates)
                        plates = [
                            p if _is_base_zone(p) else
                            _clamp_rect_max(p, width, height, max_frac=max_box_frac)
                            for p in plates
                        ]

                    if debug_overlay:
                        frame = draw_extended_overlay(
                            frame, plates, vehicles,
                            blur_padding=blur_padding,
                            blur_strength=blur_strength,
                            redact_mode=redact_mode,
                            redact_color=redact_color,
                            overlay_img=overlay_img,
                            min_moto_h_frac=min_moto_h_frac,
                            moto_large_boxes=moto_large_boxes,
                        )
                    elif debug:
                        frame = draw_debug_overlay(
                            frame, plates, vehicles,
                            own_plate_region=own_plate_region,
                            blur_padding=blur_padding,
                            blur_strength=blur_strength,
                            redact_mode=redact_mode,
                            redact_color=redact_color,
                            overlay_img=overlay_img,
                            observed_plates=observed,
                            min_moto_h_frac=min_moto_h_frac,
                            moto_large_boxes=moto_large_boxes,
                        )
                    elif plates:
                        other = [p for p in plates if not _is_base_zone(p)]
                        base = [p for p in plates if _is_base_zone(p)]
                        if other:
                            frame = apply_redaction(
                                frame, other,
                                mode=redact_mode,
                                blur_strength=blur_strength,
                                color=redact_color,
                                overlay_img=overlay_img,
                                padding=blur_padding,
                                max_box_frac=max_box_frac,
                            )
                        if base:
                            frame = apply_blur_circle(
                                frame, base,
                                blur_strength=blur_strength,
                                padding=blur_padding,
                            )

                    timings["render"] = (time.perf_counter() - _t2) * 1000

                    forced = forced_fixlist.get(cache_fi) if forced_fixlist else None
                    if forced:
                        fixed_rects = [(*r, 0.99) for r in forced]
                        frame = apply_blur_feathered(
                            frame, fixed_rects,
                            blur_strength=blur_strength,
                            padding=blur_padding + 4,
                        )

                    if debug_hud:
                        hud = draw_hud_panel(
                            frame_h=frame.shape[0],
                            telemetry={
                                "frame_num": frame_num,
                                "total_frames": total_frames,
                                "fps_target": fps,
                                "vehicles": vehicles,
                                "plates": plates,
                                "tracks": sorted({
                                    z[6] for z in plates
                                    if len(z) > 6 and z[6] != -1
                                }),
                                "timings": timings,
                            },
                        )
                        frame = np.concatenate([frame, hud], axis=1)

                    encode_proc.stdin.write(frame.tobytes())
                    frame_num += 1
                    pbar.update(1)
                    pbar.set_postfix(plates=total_plates, refresh=False)
            finally:
                _close_extract(extract_proc)
                encode_proc.stdin.close()
                encode_proc.wait()
                cache.close()

        _elapsed_process = time.perf_counter() - _process_start_ts
        action = "annotated" if debug else "redacted"
        print(f"\n  Processed : {frame_num} frames")
        print(f"  Detections: {total_plates} plate regions {action}")

        print("  Encoding final output (lossless HEVC + audio sync fix)...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        mux_audio(tmp_path, input_path, output_path, start_time, end_time, fps,
                  total_frames=frame_num, preset=preset, tmp_dir=tmp_dir,
                  quality=quality)

        _elapsed_total = time.perf_counter() - _run_start_ts
        _print_run_summary(
            elapsed_total=_elapsed_total,
            elapsed_process=_elapsed_process,
            device=device,
            info=info,
            input_path=input_path,
            output_path=output_path,
            frame_num=frame_num,
            total_plates=total_plates,
            total_quads=total_quads,
            redact_mode=redact_mode,
            redact_color=redact_color,
            redact_image_path=redact_image_path,
            vehicle_filter=vehicle_filter,
        )
        return frame_num, total_plates

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
