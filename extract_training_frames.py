#!/usr/bin/env python3
"""
Extract letterboxed vehicle crops for plate-model fine-tuning.

Saves the same square canvas the detector sees at inference (CLAHE +
letterbox at plate_crop_imgsz, default 1280) — not the full dashcam frame.
YOLO labels are in that canvas, so train/infer match.

Sampling
────────
  Background: 1 fps, subsampled (uncertain always; confident 1-in-4;
    empty moto 1-in-3; empty car 1-in-8).
  Dense windows: higher fps, every motorcycle crop kept. Default windows
    cover known misses on the 224940 montage (lean, close GS, …).

Usage
─────
    python extract_training_frames.py /path/to/videos --outdir training_data_crops

    python review_annotations.py --dataset training_data_crops
    python finetune.py --data training_data_crops/dataset.yaml --imgsz 1280
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from blur_plates import (
    load_models,
    get_video_info,
    time_to_seconds,
    _dbg_box,
    _DBG_PLATE_COLOR,
    _DBG_PREDICT_COLOR,
)
from plates.config import load_config
from plates.detect import (
    _MOTO_CLS,
    build_vehicle_crop_canvases,
    detect_vehicles,
)
from plates.models import _infer_batched

VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".mts", ".m4v", ".mpg", ".mpeg"}

CONF_UNCERTAIN_LOW = 0.05
CONF_UNCERTAIN_HIGH = 0.25
CONFIDENT_KEEP_1IN = 4
MOTO_EMPTY_KEEP_1IN = 3
CAR_EMPTY_KEEP_1IN = 8
VAL_FRACTION = 0.15

# Known misses on montage_20260808_224940 (padded a little past the clip).
DEFAULT_DENSE = "0:27-0:29.5,0:37.5-0:40.5,1:22.5-1:25.5,2:23.5-2:26.5,2:33.5-2:37.5"


# ── Frame extraction via ffmpeg pipe ──────────────────────────────────────────

def _iter_frames(video_path: Path, sample_fps: float, start_sec, end_sec):
    """Yield (raw_frame_idx, timestamp, bgr_frame, w, h, fps)."""
    info = get_video_info(str(video_path))
    w, h, fps = info["width"], info["height"], info["fps"]
    frame_size = w * h * 3

    cmd = ["ffmpeg", "-y"]
    if start_sec is not None:
        cmd += ["-ss", f"{start_sec:.6f}"]
    cmd += ["-i", str(video_path)]
    if end_sec is not None:
        cmd += ["-t", f"{end_sec - (start_sec or 0.0):.6f}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24",
            "-vf", f"fps={sample_fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            ts = (start_sec or 0.0) + raw_idx / sample_fps
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
            yield raw_idx, ts, frame, w, h, fps
            raw_idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))
    return cx, cy, bw, bh


def _parse_windows(spec: str):
    windows = []
    if not spec or not spec.strip():
        return windows
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-", 1)
        windows.append((float(time_to_seconds(a)), float(time_to_seconds(b))))
    return windows


def _in_windows(ts, windows) -> bool:
    return any(lo <= ts <= hi for lo, hi in windows)


def _is_dead_frame(frame):
    return frame.std() < 5.0


def _box_centre_on_content(x1, y1, x2, y2, pad_x, pad_y, cw, ch, scale):
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return (pad_x <= cx <= pad_x + cw * scale
            and pad_y <= cy <= pad_y + ch * scale)


def _should_keep_crop(plate_rects, is_moto, counters, force=False):
    if force:
        return True
    uncertain = any(
        CONF_UNCERTAIN_LOW <= (r[4] if len(r) > 4 else 0) < CONF_UNCERTAIN_HIGH
        for r in plate_rects
    )
    if uncertain:
        return True
    has_plate = any((r[4] if len(r) > 4 else 0) >= CONF_UNCERTAIN_HIGH
                    for r in plate_rects)
    if is_moto:
        if has_plate:
            counters["moto_ok"] = counters.get("moto_ok", 0) + 1
            return counters["moto_ok"] % CONFIDENT_KEEP_1IN == 0
        counters["moto_empty"] = counters.get("moto_empty", 0) + 1
        return counters["moto_empty"] % MOTO_EMPTY_KEEP_1IN == 0
    if has_plate:
        counters["car_ok"] = counters.get("car_ok", 0) + 1
        return counters["car_ok"] % CONFIDENT_KEEP_1IN == 0
    counters["car_empty"] = counters.get("car_empty", 0) + 1
    return counters["car_empty"] % CAR_EMPTY_KEEP_1IN == 0


def _review_crop(canvas, plate_rects):
    vis = canvas.copy()
    for rect in plate_rects:
        x1, y1, x2, y2 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        conf = rect[4] if len(rect) > 4 else 0.0
        color = _DBG_PREDICT_COLOR if conf < CONF_UNCERTAIN_HIGH else _DBG_PLATE_COLOR
        label = f"plate {conf:.2f}" + ("  LOW" if conf < CONF_UNCERTAIN_HIGH else "")
        _dbg_box(vis, x1, y1, x2, y2, color, label, thickness=2)
    return vis


def _infer_boxes_on_crops(plate_model, crops, conf, imgsz):
    """Return a list (one per crop) of (x1, y1, x2, y2, conf) in canvas pixels."""
    if not crops:
        return []
    rgb = [cv2.cvtColor(c[0], cv2.COLOR_BGR2RGB) for c in crops]
    results = _infer_batched(plate_model.model, rgb, conf=conf, imgsz=imgsz)
    out = []
    for crop_r, (canvas, cw, ch, _cx1, _cy1, _src, scale, pad_x, pad_y, _cls) in zip(
            results, crops):
        boxes = []
        if crop_r.boxes is None:
            out.append(boxes)
            continue
        ch_canvas, cw_canvas = canvas.shape[:2]
        for box in crop_r.boxes:
            c = float(box.conf[0])
            x1, y1, x2, y2 = (float(box.xyxy[0][0]), float(box.xyxy[0][1]),
                              float(box.xyxy[0][2]), float(box.xyxy[0][3]))
            if not _box_centre_on_content(x1, y1, x2, y2, pad_x, pad_y, cw, ch, scale):
                continue
            x1i = int(max(0, min(cw_canvas - 1, round(x1))))
            y1i = int(max(0, min(ch_canvas - 1, round(y1))))
            x2i = int(max(0, min(cw_canvas, round(x2))))
            y2i = int(max(0, min(ch_canvas, round(y2))))
            if x2i > x1i and y2i > y1i:
                boxes.append((x1i, y1i, x2i, y2i, c))
        out.append(boxes)
    return out


def _crop_kwargs_from_config():
    cfg = load_config()
    det = cfg.get("detection", {})
    pre = cfg.get("preprocessing", {})
    return dict(
        vehicle_conf=float(det.get("vehicle_conf", 0.35)),
        vehicle_conf_floor=float(det.get("moto_close_conf", 0.20)),
        vehicle_crop_scale=float(pre.get("vehicle_crop_scale", 2.0)),
        moto_crop_scale=float(pre.get("moto_crop_scale", 2.0)),
        moto_crop_bottom_frac=float(pre.get("moto_crop_bottom_frac", 0.28)),
        moto_crop_side_pad_frac=float(pre.get("moto_crop_side_pad_frac", 0.05)),
        plate_crop_imgsz=int(pre.get("plate_crop_imgsz", 1280)),
        crop_clahe=bool(pre.get("crop_clahe", True)),
        crop_clahe_clip=float(pre.get("crop_clahe_clip", 2.0)),
        crop_clahe_grid=int(pre.get("crop_clahe_grid", 8)),
        sharpen=bool(pre.get("sharpen", False)),
        sharpen_amount=float(pre.get("sharpen_amount", 1.5)),
        sharpen_sigma=float(pre.get("sharpen_sigma", 1.0)),
    )


def _write_crop(outdir, stem, split, canvas, plate_rects, ext, write_params,
                write_review):
    img_h, img_w = canvas.shape[:2]
    img_path = outdir / "images" / split / f"{stem}{ext}"
    cv2.imwrite(str(img_path), canvas, write_params)
    lbl_path = outdir / "labels" / split / f"{stem}.txt"
    lines = []
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf = rect[4] if len(rect) > 4 else 0.0
        if conf >= CONF_UNCERTAIN_LOW:
            cx, cy, bw, bh = _to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    lbl_path.write_text("\n".join(lines))
    if write_review:
        rev = _review_crop(canvas, plate_rects)
        cv2.imwrite(str(outdir / "review" / f"{stem}.jpg"),
                    rev, [cv2.IMWRITE_JPEG_QUALITY, 82])


def _process_pass(video_path, outdir, vehicle_model, plate_model, crop_kw,
                  sample_fps, start_sec, end_sec, write_review, budget, rng,
                  counters, collected, ext, write_params, skip_windows,
                  force_moto, image_format="png"):
    n_dead = 0
    crop_imgsz = crop_kw["plate_crop_imgsz"]
    vid_stem = video_path.stem
    for _raw_idx, ts, frame, _w, _h, _fps in _iter_frames(
            video_path, sample_fps, start_sec, end_sec):
        if len(collected) >= budget:
            break
        if skip_windows and _in_windows(ts, skip_windows):
            continue
        if _is_dead_frame(frame):
            n_dead += 1
            continue
        vehicles = detect_vehicles(
            frame, vehicle_model,
            vehicle_conf=crop_kw["vehicle_conf"],
            vehicle_conf_floor=crop_kw["vehicle_conf_floor"],
        )
        if not vehicles:
            continue
        crops = build_vehicle_crop_canvases(
            frame, vehicles,
            vehicle_crop_scale=crop_kw["vehicle_crop_scale"],
            moto_crop_scale=crop_kw["moto_crop_scale"],
            moto_crop_bottom_frac=crop_kw["moto_crop_bottom_frac"],
            moto_crop_side_pad_frac=crop_kw["moto_crop_side_pad_frac"],
            plate_crop_imgsz=crop_imgsz,
            crop_clahe=crop_kw["crop_clahe"],
            crop_clahe_clip=crop_kw["crop_clahe_clip"],
            crop_clahe_grid=crop_kw["crop_clahe_grid"],
            sharpen=crop_kw["sharpen"],
            sharpen_amount=crop_kw["sharpen_amount"],
            sharpen_sigma=crop_kw["sharpen_sigma"],
        )
        if not crops:
            continue
        all_boxes = _infer_boxes_on_crops(
            plate_model, crops, CONF_UNCERTAIN_LOW, crop_imgsz)
        for i, (crop, boxes) in enumerate(zip(crops, all_boxes)):
            if len(collected) >= budget:
                break
            canvas, _cw, _ch, _cx1, _cy1, source, _sc, _px, _py, cls_id = crop
            is_moto = cls_id == _MOTO_CLS
            force = bool(force_moto and is_moto)
            if not _should_keep_crop(boxes, is_moto, counters, force=force):
                continue
            split = "val" if rng.random() < VAL_FRACTION else "train"
            stem = f"{vid_stem}_t{ts:07.2f}_{source}_{i}"
            _write_crop(outdir, stem, split, canvas, boxes, ext, write_params,
                        write_review)
            collected.append((stem, split, source, is_moto))
    return n_dead


def _process_video(video_path, outdir, vehicle_model, plate_model, crop_kw,
                   sample_fps, start_sec, end_sec, dense_windows, dense_fps,
                   write_review, budget, rng, image_format="png",
                   do_dense=True, do_background=True):
    counters = {}
    collected = []
    if image_format == "png":
        ext, write_params = ".png", []
    else:
        ext, write_params = ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]

    n_dead = 0
    info = get_video_info(str(video_path))
    duration = float(info.get("duration") or 0.0)
    windows = []
    for lo, hi in dense_windows:
        wlo = max(lo, start_sec or 0.0)
        whi = hi if end_sec is None else min(hi, end_sec)
        if duration > 0:
            whi = min(whi, duration)
        if whi > wlo:
            windows.append((wlo, whi))

    if do_dense and windows and dense_fps > 0:
        for lo, hi in windows:
            if len(collected) >= budget:
                break
            n_dead += _process_pass(
                video_path, outdir, vehicle_model, plate_model, crop_kw,
                dense_fps, lo, hi, write_review, budget, rng, counters,
                collected, ext, write_params, skip_windows=None,
                force_moto=True, image_format=image_format,
            )

    if do_background and len(collected) < budget:
        n_dead += _process_pass(
            video_path, outdir, vehicle_model, plate_model, crop_kw,
            sample_fps, start_sec, end_sec, write_review, budget, rng,
            counters, collected, ext, write_params,
            skip_windows=windows, force_moto=False, image_format=image_format,
        )

    if n_dead:
        tqdm.write(f"    (skipped {n_dead} dead frame(s))")
    return collected


def _write_yaml(outdir: Path, imgsz: int):
    yaml = f"""\
# Auto-generated by extract_training_frames.py
# Images are letterboxed vehicle crops at {imgsz}px (same as inference).
# Review/correct labels with:   python review_annotations.py --dataset {outdir}
# Fine-tune with:               python finetune.py --data {outdir.resolve()}/dataset.yaml --imgsz {imgsz}

path:  {outdir.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  - license_plate
"""
    (outdir / "dataset.yaml").write_text(yaml)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",
                    help="Video file or folder of videos")
    ap.add_argument("--outdir", default="training_data_crops",
                    help="Output dataset directory (default: training_data_crops/)")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="Background sample rate (default: 1.0)")
    ap.add_argument("--dense", default=DEFAULT_DENSE,
                    help="Comma-separated MM:SS-MM:SS windows at --dense-fps")
    ap.add_argument("--dense-fps", type=float, default=10.0,
                    help="Sample rate inside dense windows (default: 10)")
    ap.add_argument("--start", default=None, help="Start time MM:SS")
    ap.add_argument("--end", default=None, help="End time MM:SS")
    ap.add_argument("--max-frames", type=int, default=1200,
                    help="Max crops written across all videos (default: 1200)")
    ap.add_argument("--no-review", action="store_true",
                    help="Skip writing review overlay images")
    ap.add_argument("--format", choices=["png", "jpg"], default="png",
                    help="Image format (default: png)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    rng = random.Random(args.seed)
    crop_kw = _crop_kwargs_from_config()
    dense_windows = _parse_windows(args.dense)

    if inp.is_dir():
        videos = sorted(p for p in inp.iterdir()
                        if p.suffix.lower() in VIDEO_EXTENSIONS
                        and "_blurred" not in p.stem)
        if not videos:
            sys.exit(f"No video files found in {inp}")
    elif inp.is_file():
        videos = [inp]
    else:
        sys.exit(f"Input not found: {inp}")
    # Dense windows were chosen from 224940 misses — extract that video first
    # so a crop budget cannot skip them.
    videos.sort(key=lambda p: (0 if "224940" in p.stem else 1, p.name))

    for split in ("train", "val"):
        (outdir / "images" / split).mkdir(parents=True, exist_ok=True)
        (outdir / "labels" / split).mkdir(parents=True, exist_ok=True)
    if not args.no_review:
        (outdir / "review").mkdir(parents=True, exist_ok=True)

    start_sec = time_to_seconds(args.start) if args.start else None
    end_sec = time_to_seconds(args.end) if args.end else None

    print("\nLoading models...")
    vehicle_model, plate_model, device = load_models(plate_conf=CONF_UNCERTAIN_LOW)
    print(f"Models ready on {device}. Processing {len(videos)} video(s).")
    print(f"  Crop canvas : {crop_kw['plate_crop_imgsz']}  CLAHE={crop_kw['crop_clahe']}")
    if dense_windows:
        print(f"  Dense       : {args.dense_fps:.0f} fps in {len(dense_windows)} window(s)")
    print()

    all_frames: list = []
    budget = args.max_frames

    def _run_videos(phase, do_dense, do_background):
        nonlocal budget
        for video in tqdm(videos, unit="video", desc=phase):
            if budget <= 0:
                tqdm.write("  Crop budget reached — stopping.")
                break
            tqdm.write(f"\n  {video.name}")
            collected = _process_video(
                video, outdir, vehicle_model, plate_model, crop_kw,
                sample_fps=args.fps,
                start_sec=start_sec, end_sec=end_sec,
                dense_windows=dense_windows, dense_fps=args.dense_fps,
                write_review=not args.no_review,
                budget=budget, rng=rng,
                image_format=args.format,
                do_dense=do_dense, do_background=do_background,
            )
            budget -= len(collected)
            all_frames.extend(collected)
            n_t = sum(1 for _, s, *_ in collected if s == "train")
            n_v = sum(1 for _, s, *_ in collected if s == "val")
            n_m = sum(1 for *_, m in collected if m)
            tqdm.write(f"    → {len(collected)} crops  (train={n_t}, val={n_v}, moto={n_m})")

    _run_videos("Dense windows", do_dense=True, do_background=False)
    _run_videos("Background 1fps", do_dense=False, do_background=True)

    _write_yaml(outdir, crop_kw["plate_crop_imgsz"])

    n_train = sum(1 for _, s, *_ in all_frames if s == "train")
    n_val = sum(1 for _, s, *_ in all_frames if s == "val")
    n_moto = sum(1 for *_, m in all_frames if m)

    print(f"\n{'═'*58}")
    print(f"  Dataset ready   {len(all_frames)} crops  "
          f"(train={n_train}, val={n_val}, moto={n_moto})")
    print(f"  Output: {outdir.resolve()}/")
    print(f"{'═'*58}")
    print(f"""
Step 2 — Correct auto-annotations (draw the real plate on empty moto crops)
    python review_annotations.py --dataset {outdir}

Step 3 — Fine-tune at the inference canvas size
    python finetune.py --data {outdir}/dataset.yaml --imgsz {crop_kw['plate_crop_imgsz']} --name dashcam_crops --device 0 --batch 4 --workers 2
""")


if __name__ == "__main__":
    main()
