#!/usr/bin/env python3
"""
License Plate Blurring Tool
===========================
Detects license plates using YOLOv8 vehicle detection + a fine-tuned plate
model run with SAHI sliced inference, then blurs them. Output is visually
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
from plates.kalman import BoxFilter
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
    apply_blur_round,
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
from plates.track import PlateHistory, SceneTracker, VehicleTrack, _VehicleDetections


def main():
    cfg = load_config()
    det  = cfg["detection"]
    sahi = cfg["sahi"]
    blr  = cfg["blur"]
    out  = cfg["output"]
    trk  = cfg.get("tracking", {})
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
                        help="Write detection overlay video instead of blurring "
                             "(blue=vehicles, green=plate regions, orange=own plate)")
    parser.add_argument("--debug-overlay", dest="debug_overlay",
                        action="store_true",
                        help="DEBUG DATA mode (A): rich in-frame overlay with source "
                             "tags (SAHI/crop+/pred), trajectory trails and ghost "
                             "boxes for tracked vehicles whose detector missed.")
    parser.add_argument("--debug-hud", dest="debug_hud", action="store_true",
                        help="DEBUG DATA mode (B): add a brand-styled side panel "
                             "with frame#, counts, track list and per-stage timings. "
                             "Output video gets wider by ~320 px.")
    parser.add_argument("--detect-scale", dest="detect_scale", type=float,
                        default=float(det.get("detect_scale", 1.0)),
                        help="Fraction of resolution used for detection (default: "
                             f"{det.get('detect_scale', 1.0)}).  "
                             "0.5 = ~5× faster on 4K, minimal accuracy loss at "
                             "typical dashcam distances. Blur always at full res.")
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
    parser.add_argument("--zone-filter", dest="zone_filter",
                        default=None, choices=["ema", "kalman"],
                        help="Temporal filter for the blur zone. Overrides "
                             f"[tracking] zone_filter (config: "
                             f"{trk.get('zone_filter', 'ema')}). 'kalman' removes "
                             "the lag of the chained EMAs and the lag/overshoot "
                             "alternation on frames where the plate is missed.")
    parser.add_argument("--detect-cache", dest="detect_cache",
                        default=None,
                        metavar="JSONL",
                        help="Cache the raw detector output per frame. The file is "
                             "written when it does not exist and replayed when it "
                             "does, so tracking/blur experiments skip inference "
                             "entirely. The cache is refused if the detection "
                             "settings differ from the ones it was built with.")
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
        sahi_slice_size=sahi["slice_size"],
        sahi_overlap=sahi["overlap"],
        own_plate_region=own_plate,
        vehicle_filter=args.vehicles,
        preset=out["preset"],
        quality=int(out.get("quality", 18)),
        tmp_dir=out["tmp_dir"],
        debug=args.debug,
        debug_overlay=args.debug_overlay,
        debug_hud=args.debug_hud,
        tracking_enabled=bool(trk.get("enabled", True)),
        max_gap_frames=int(trk.get("max_gap_frames", 8)),
        history_frames=int(trk.get("history_frames", 15)),
        min_vehicle_conf=float(trk.get("min_vehicle_conf", 0.60)),
        predict_expand_max=int(trk.get("predict_expand_max", 20)),
        standalone_min_ar=float(det.get("standalone_min_ar", 1.2)),
        standalone_max_ar=float(det.get("standalone_max_ar", 6.0)),
        detect_scale=args.detect_scale,
        sharpen=bool(pre.get("sharpen", False)),
        sharpen_amount=float(pre.get("sharpen_amount", 1.5)),
        sharpen_sigma=float(pre.get("sharpen_sigma", 1.0)),
        vehicle_crop_scale=float(pre.get("vehicle_crop_scale", 1.0)),
        moto_crop_scale=float(pre.get("moto_crop_scale", 3.0)),
        moto_crop_bottom_frac=float(pre.get("moto_crop_bottom_frac", 0.28)),
        moto_crop_side_pad_frac=float(pre.get("moto_crop_side_pad_frac", 0.05)),
        redact_mode=args.mode,
        redact_color=redact_color,
        redact_image_path=args.image if args.mode == "image" else None,
        max_box_frac=float(blr.get("max_box_frac", 0.15)),
        predict_max_disp=float(trk.get("predict_max_disp", 40.0)),
        fallback_enabled=bool(trk.get("fallback_enabled", True)),
        fallback_frac=float(trk.get("fallback_frac", 0.40)),
        fallback_pad_frac=float(trk.get("fallback_pad_frac", 0.25)),
        fallback_min_frames=int(trk.get("fallback_min_frames", 3)),
        ema_alpha=float(trk.get("ema_alpha", 0.6)),
        moto_ar_min=float(trk.get("moto_ar_min", 0.9)),
        moto_ar_max=float(trk.get("moto_ar_max", 1.3)),
        moto_anchor=bool(trk.get("moto_anchor", True)),
        moto_anchor_frac=float(trk.get("moto_anchor_frac", 0.45)),
        moto_anchor_y=float(trk.get("moto_anchor_y", 0.70)),
        moto_anchor_pad=float(trk.get("moto_anchor_pad", 0.15)),
        moto_ghost_frames=int(trk.get("moto_ghost_frames", 6)),
        moto_close_frac=float(trk.get("moto_close_frac", 0.40)),
        moto_close_conf=float(trk.get("moto_close_conf", 0.20)),
        moto_close_zone_w=float(trk.get("moto_close_zone_w", 1.6)),
        moto_edge_px=int(trk.get("moto_edge_px", 4)),
        moto_near_frac=float(trk.get("moto_near_frac", 0.15)),
        moto_zone_min_side=float(trk.get("moto_zone_min_side", 40.0)),
        moto_anchor_y_max=float(trk.get("moto_anchor_y_max", 0.72)),
        moto_plate_conf=float(trk.get("moto_plate_conf", 0.30)),
        moto_plate_promote_frames=int(trk.get("moto_plate_promote_frames", 2)),
        moto_plate_hold_frames=int(trk.get("moto_plate_hold_frames", 15)),
        moto_plate_pad=int(blr.get("padding", 10)),
        moto_min_blur_box_h_frac=float(trk.get("moto_min_blur_box_h_frac", 0.10)),
        moto_max_zone_box_frac=float(trk.get("moto_max_zone_box_frac", 0.35)),
        moto_weak_fender_frac=float(trk.get("moto_weak_fender_frac", 0.92)),
        moto_quad_refine=bool(trk.get("moto_quad_refine", True)),
        blur_shape=str(blr.get("blur_shape", "rect")),
        forced_fixlist=forced_fix,
        emit_max_disp=float(trk.get("emit_max_disp", 80.0)),
        detect_cache=args.detect_cache,
        zone_filter=args.zone_filter or str(trk.get("zone_filter", "ema")),
        kf_params={k: float(trk[k]) for k in (
            "kf_process_pos", "kf_process_size", "kf_meas_pos", "kf_meas_size",
            "kf_gate_max", "kf_max_rejects", "kf_sigma_pad_k",
            "kf_sigma_pad_max", "kf_pad_decay", "kf_vel_decay",
            "kf_anchor_meas_scale",
        ) if k in trk},
    )


if __name__ == "__main__":
    main()
