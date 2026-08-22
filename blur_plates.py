#!/usr/bin/env python3
"""
License Plate Blurring Tool
===========================
Detects license plates using YOLOv8 vehicle detection plus a plate model
on each vehicle crop, then blurs them. Output is visually
lossless (FFV1 intermediate → HEVC CRF/CQ 18) with the original audio
preserved.

Usage:
    python blur_plates.py <input> <output> [--start MM:SS] [--end MM:SS] [--blur N] [--conf F]
    python blur_plates.py <input> <output> --own-plate x1,y1,x2,y2   # fixed region for camera-mounted plates

Example:
    python blur_plates.py final.mov output_blurred.mov --start 2:48 --end 2:51
    python blur_plates.py final.mov output.mov --own-plate 1700,900,2200,1100
"""

"""
Module layout
─────────────
The implementation lives in the ``plates`` package; this module re-exports
the full public API so ``import blur_plates`` keeps working unchanged for
scripts, tests and debug tooling, and provides the CLI entry point.
"""
import argparse
import json
import os
import sys

from plates.common import (
    parse_color,
    parse_region,
    time_to_seconds,
    _clamp_rect_max,
    _format_duration,
    _overlaps,
    iou,
    merge_overlapping,
    suppress_duplicate_plates,
)
from plates.config import load_config
from plates.constants import (
    DETECT_WIDTH,
    PLATE_MODEL_PATH,
    TRT_BATCH,
    VEHICLE_CLASSES,
    VEHICLE_FILTER_MAP,
    _CUVID_DECODERS,
    _DBG_BLUR_COLOR,
    _DBG_FONT,
    _DBG_GHOST_COLOR,
    _DBG_OWN_COLOR,
    _DBG_PLATE_COLOR,
    _DBG_PREDICT_COLOR,
    _DBG_SUPPRESSED_COLOR,
    _DBG_VEHICLE_COLOR,
    _DD_BLUE,
    _DD_DIM,
    _DD_F_BODY,
    _DD_F_DISP,
    _DD_F_MONO,
    _DD_F_MONOB,
    _DD_FONT_DIR,
    _DD_GHOST,
    _DD_GREEN,
    _DD_HUD_BG,
    _DD_HUD_LINE,
    _DD_HUD_W,
    _DD_ORANGE,
    _DD_RED,
    _DD_SOURCE_COLOR,
    _DD_TRAIL,
    _DD_VOID_BG,
    _DD_WHITE,
    _DD_YELLOW,
)
from plates.detect import detect_plates, detect_plates_batched, auto_batch_size, _unsharp_mask
from plates.ffmpeg import (
    build_ffmpeg_extract,
    build_ffmpeg_encode_lossless,
    estimate_frame_count,
    get_video_info,
    mux_audio,
    _ffmpeg_with_progress,
    _source_color_args,
)
from plates.common import covered_fraction
from plates.models import load_models, _infer_batched, _pad_to_batch, _prefer_engine
from plates.overlay import (
    draw_debug_overlay,
    draw_extended_overlay,
    draw_hud_panel,
    _dbg_box,
    _dd_dashed_rect,
    _dd_font,
    _dd_have_pil_fonts,
    _dd_tag,
    _draw_hud_panel_fallback,
)
from plates.pipeline import blur_license_plates
from plates.redact import (
    apply_blur,
    apply_blur_feathered,
    apply_blur_rotated,
    apply_blur_circle,
    apply_image_overlay,
    apply_redaction,
    apply_solid_color,
    estimate_plate_quad,
    load_overlay_image,
    _align_quad,
    _match_quad_state,
    _quad_state_update,
)
from plates.report import _hardware_label, _print_run_summary
from plates.track import build_zones, zone_stats


def main():
    cfg = load_config()
    det  = cfg["detection"]
    blr  = cfg["blur"]
    out  = cfg["output"]
    trk  = cfg.get("zones", {})
    pre  = cfg.get("preprocessing", {})
    red  = cfg.get("redact", {})

    parser = argparse.ArgumentParser(
        description="Blur license plates in video with zero quality loss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full video
  python blur_plates.py final.mov output.mov

  # Clip from 2:48 to 2:51
  python blur_plates.py final.mov sample.mov --start 2:48 --end 2:51

  # Camera mounted behind own plate (Insta360 on motorbike etc.)
  python blur_plates.py final.mov output.mov --own-plate 1700,900,2200,1100

  # Motorbike-only with lower in-vehicle plate threshold
  python blur_plates.py final.mov output.mov --vehicles motorbike --plate-conf-in-vehicle 0.05

  # Debug mode — see what gets detected without blurring
  python blur_plates.py final.mov debug.mov --start 0:10 --end 0:20 --debug

  # Replace plates with a solid colour instead of blurring
  python blur_plates.py final.mov output.mov --mode color --color 255,0,0

  # Overlay a custom image (logo / sticker / portrait) onto every plate
  python blur_plates.py final.mov output.mov --mode image --image my_sticker.png
        """,
    )
    parser.add_argument("input",  help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--start", help="Start time MM:SS or HH:MM:SS", default=None)
    parser.add_argument("--end",   help="End time   MM:SS or HH:MM:SS", default=None)
    parser.add_argument("--blur",  type=int,   default=blr["strength"],
                        help=f"Blur kernel size, odd (default: {blr['strength']})")
    parser.add_argument("--conf",  type=float, default=det["vehicle_conf"],
                        help=f"Vehicle detection confidence (default: {det['vehicle_conf']})")
    parser.add_argument("--plate-conf", dest="plate_conf", type=float,
                        default=det["plate_conf"],
                        help=f"Plate confidence — full frame (default: {det['plate_conf']})")
    parser.add_argument("--plate-conf-in-vehicle", dest="plate_conf_in_vehicle", type=float,
                        default=det["plate_conf_in_vehicle"],
                        help=f"Plate confidence inside vehicle boxes (default: {det['plate_conf_in_vehicle']})")
    parser.add_argument("--own-plate", dest="own_plate", default=None,
                        metavar="x1,y1,x2,y2",
                        help="Fixed region to always blur (e.g. camera behind own plate)")
    parser.add_argument("--vehicles", dest="vehicles", default="all",
                        choices=["all", "motorbike", "car", "bus", "truck"],
                        help="Only blur plates on the specified vehicle type (default: all)")
    parser.add_argument("--debug", action="store_true",
                        help="Write detection overlay video instead of a clean blur "
                             "(green=vehicles, magenta=moto below size threshold, "
                             "blue=plate-model boxes, yellow=interpolated)")
    parser.add_argument("--debug-overlay", dest="debug_overlay",
                        action="store_true",
                        help="DEBUG DATA mode (A): rich in-frame overlay with source "
                             "tags (crop/bridge) on interpolated plate zones.")
    parser.add_argument("--debug-hud", dest="debug_hud", action="store_true",
                        help="DEBUG DATA mode (B): add a brand-styled side panel "
                             "with frame#, counts and per-stage timings. "
                             "Output video gets wider by ~320 px.")
    parser.add_argument("--mode", dest="mode",
                        default=red.get("mode", "blur"),
                        choices=["blur", "color", "image"],
                        help="Redaction style applied to detected plates "
                             "(default: blur). "
                             "color = solid fill, image = stretched overlay.")
    parser.add_argument("--color", dest="color",
                        default=red.get("color", "0,0,0"),
                        metavar="R,G,B",
                        help="Solid fill colour when --mode color (default: 0,0,0 = black). "
                             "Values 0-255.")
    parser.add_argument("--image", dest="image",
                        default=red.get("image", None),
                        metavar="PATH",
                        help="Overlay image when --mode image. PNG with alpha is supported. "
                             "The image is stretched to fill each plate rectangle.")
    parser.add_argument("--detect-cache", dest="detect_cache",
                        default=None,
                        metavar="JSONL",
                        help="Cache the raw detector output per frame. The file is "
                             "written when it does not exist and replayed when it "
                             "does, so blur experiments skip inference entirely. "
                             "The cache is refused if the detection settings differ "
                             "from the ones it was built with.")
    parser.add_argument("--fixlist", dest="fixlist",
                        default=None,
                        metavar="JSON",
                        help="JSON map of frame_idx -> [[x1,y1,x2,y2],...] from the audit "
                             "pass; those frames get a guaranteed rectangular blur on top "
                             "of the normal pipeline redaction. Only valid for full-video "
                             "runs (no --start/--end).")

    args = parser.parse_args()

    start_sec = time_to_seconds(args.start) if args.start else None
    end_sec   = time_to_seconds(args.end)   if args.end   else None

    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        print("Error: --end must be after --start")
        sys.exit(1)

    own_plate = parse_region(args.own_plate) if args.own_plate else None

    # Validate redaction-mode prerequisites
    redact_color = parse_color(args.color) if args.mode == "color" else (0, 0, 0)
    if args.mode == "image":
        if not args.image:
            print("Error: --mode image requires --image PATH")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"Error: overlay image not found: {args.image}")
            sys.exit(1)

    if args.fixlist:
        with open(args.fixlist, "r", encoding="utf-8-sig") as fh:
            fix_loaded = json.load(fh)
        # JSON keys are strings; the pipeline looks up by int frame index.
        forced_fix = {int(k): v for k, v in fix_loaded.items()}
        print(f"  Fixlist : {len(forced_fix)} forced frames loaded (audit pass)")
    else:
        forced_fix = None

    blur_license_plates(
        input_path=args.input,
        output_path=args.output,
        start_time=start_sec,
        end_time=end_sec,
        blur_strength=args.blur,
        blur_padding=blr["padding"],
        vehicle_conf=args.conf,
        plate_conf=args.plate_conf,
        plate_conf_in_vehicle=args.plate_conf_in_vehicle,
        sahi_slice_size=640,   # unused; kept in cache header for old jsonl
        sahi_overlap=0.2,
        own_plate_region=own_plate,
        vehicle_filter=args.vehicles,
        preset=out["preset"],
        quality=int(out.get("quality", 18)),
        tmp_dir=out["tmp_dir"],
        debug=args.debug,
        debug_overlay=args.debug_overlay,
        debug_hud=args.debug_hud,
        detect_scale=0.5,  # unused; kept in cache header for old jsonl
        sharpen=bool(pre.get("sharpen", False)),
        sharpen_amount=float(pre.get("sharpen_amount", 1.5)),
        sharpen_sigma=float(pre.get("sharpen_sigma", 1.0)),
        vehicle_crop_scale=float(pre.get("vehicle_crop_scale", 2.0)),
        moto_crop_scale=float(pre.get("moto_crop_scale", 2.0)),
        moto_crop_bottom_frac=float(pre.get("moto_crop_bottom_frac", 0.28)),
        moto_crop_side_pad_frac=float(pre.get("moto_crop_side_pad_frac", 0.05)),
        plate_crop_imgsz=int(pre.get("plate_crop_imgsz", 1280)),
        crop_clahe=bool(pre.get("crop_clahe", False)),
        crop_clahe_clip=float(pre.get("crop_clahe_clip", 2.0)),
        crop_clahe_grid=int(pre.get("crop_clahe_grid", 8)),
        moto_close_conf=float(det.get("moto_close_conf", 0.20)),
        moto_min_conf=float(det.get("moto_min_conf", 0.30)),
        redact_mode=args.mode,
        redact_color=redact_color,
        redact_image_path=args.image if args.mode == "image" else None,
        max_box_frac=float(blr.get("max_box_frac", 0.15)),
        forced_fixlist=forced_fix,
        detect_cache=args.detect_cache,
        zone_params={
            "max_gap_frames": int(trk.get("max_gap_frames", 30)),
            "max_disp_px": float(trk.get("max_disp_px", 80.0)),
            "max_disp_frac": float(trk.get("max_disp_frac", 0.35)),
            "min_vehicle_iou": float(trk.get("min_vehicle_iou", 0.15)),
            "conf_floor": float(trk.get("conf_floor", det.get("plate_conf_in_vehicle", 0.15))),
            "min_plate_side_px": float(trk.get("min_plate_side_px", 12.0)),
            "max_area_ratio": float(trk.get("max_area_ratio", 2.5)),
            "max_class_flip_frames": int(trk.get("max_class_flip_frames", 10)),
            "min_moto_h_frac": float(trk.get("moto_min_blur_box_h_frac", 0.08333)),
            "moto_size_enter_frac": float(trk.get("moto_size_enter_frac", 1.15)),
            "moto_size_exit_frac": float(trk.get("moto_size_exit_frac", 0.85)),
            "moto_min_conf": float(det.get("moto_min_conf", 0.30)),
            "moto_base_blur": bool(trk.get("moto_base_blur", True)),
            "moto_base_if_no_plate": bool(trk.get("moto_base_if_no_plate", True)),
            "moto_base_height_frac": float(trk.get("moto_base_height_frac", 1.0 / 3.0)),
            "moto_base_min_height": float(trk.get("moto_base_min_height", 22.0)),
            "moto_base_smooth_alpha": float(trk.get("moto_base_smooth_alpha", 0.22)),
            "moto_base_smooth_alpha_pos": float(trk.get("moto_base_smooth_alpha_pos", 0.35)),
        },
    )


if __name__ == "__main__":
    main()
