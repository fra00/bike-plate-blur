"""Audit pass: detect still-readable license plates on a (supposedly blurred)
video and emit a fixlist JSON for blur_plates --fixlist.

For every frame the plate detector runs at production thresholds; each detected
plate crop is scored with Laplacian variance.  A truly blurred plate is flat
(low variance); a plate that survived the blur keeps high-frequency content
(high variance) and is flagged.

Usage:
  python audit_blur.py blurred.mp4 fixlist.json [--var-thresh 150]
                                               [--min-w 40] [--min-h 12]
                                               [--out-crops crops/]
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from tqdm import tqdm

import blur_plates as bp


def _overlaps(a, b, margin=0):
    """True if rect a (4-tuple) overlaps rect b within *margin* pixels."""
    return (a[0] < b[2] + margin and a[2] > b[0] - margin
            and a[1] < b[3] + margin and a[3] > b[1] - margin)


def laplacian_var(rect, frame):
    """Laplacian variance of the plate crop, normalised to a fixed size."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(rect[0])), max(0, int(rect[1]))
    x2, y2 = min(w, int(rect[2])), min(h, int(rect[3]))
    if x2 - x1 < 8 or y2 - y1 < 4:
        return 0.0, None
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (96, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), crop


def merge_rects(rects):
    """Merge overlapping rects into bounding boxes."""
    merged = []
    for r in sorted(rects, key=lambda r: (r[0], r[1])):
        if not merged:
            merged.append(r)
            continue
        mx1, my1, mx2, my2 = merged[-1]
        x1, y1, x2, y2 = r
        if (min(mx2, x2) > max(mx1, x1) and min(my2, y2) > max(my1, y1)):
            merged[-1] = (min(mx1, x1), min(my1, y1),
                          max(mx2, x2), max(my2, y2))
        else:
            merged.append(r)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Audit a blurred video for "
                                                 "still-readable plates.")
    parser.add_argument("input", help="(Blurred) input video path")
    parser.add_argument("fixlist", help="Output fixlist JSON path")
    parser.add_argument("--var-thresh", dest="var_thresh", type=float,
                        default=150.0,
                        help="Laplacian variance above which a plate counts "
                             "as readable (default: 150)")
    parser.add_argument("--min-w", dest="min_w", type=float, default=40.0,
                        help="Minimum plate width to consider (default: 40)")
    parser.add_argument("--min-h", dest="min_h", type=float, default=12.0,
                        help="Minimum plate height to consider (default: 12)")
    parser.add_argument("--conf", dest="audit_conf", type=float, default=0.15,
                        help="Minimum plate confidence in audit (default: 0.15)")
    parser.add_argument("--covered", dest="covered", default=None,
                        metavar="JSON",
                        help="Fixlist JSON (frame -> patches) from the previous "
                             "iteration. Detections overlapping a known patch are "
                             "assumed covered and skipped, so forced-blur patch "
                             "edges are not re-flagged.")
    parser.add_argument("--out-crops", dest="out_crops", default=None,
                        help="Optional folder to dump flagged plate crops for "
                             "human review")
    args = parser.parse_args()

    covered = None
    if args.covered:
        covered = {int(k): v for k, v in json.load(
            open(args.covered, "r", encoding="utf-8-sig")).items()}
        print(f"  Covered map: {len(covered)} frames with known patches")

    cfg = bp.load_config()
    det  = cfg["detection"]
    pre  = cfg.get("preprocessing", {})
    trk  = cfg.get("tracking", {}) or cfg.get("zones", {})

    print(f"  Audit input : {args.input}")
    info = bp.get_video_info(args.input)
    width, height, fps = info["width"], info["height"], info["fps"]
    frame_size = width * height * 3

    print("  Loading models...")
    vehicle_model, plate_model, device = bp.load_models(
        plate_conf=min(det["plate_conf"], det["plate_conf_in_vehicle"])
    )
    print("  Models ready\n")

    proc = subprocess.Popen(bp.build_ffmpeg_extract(args.input, None, None),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    fixlist = {}        # frame_idx -> merged rects
    all_vars = []
    det_total = 0
    frame_num = 0
    os.makedirs(args.out_crops, exist_ok=True) if args.out_crops else None

    try:
        with tqdm(desc="audit", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (height, width, 3)).copy()

                plates, _, _ = bp.detect_plates(
                    frame, vehicle_model, plate_model,
                    device=device,
                    vehicle_conf=float(det["vehicle_conf"]),
                    vehicle_conf_floor=float(trk.get("moto_close_conf",
                                                     det["vehicle_conf"])),
                    vehicle_filter="all",
                    plate_conf=float(det["plate_conf"]),
                    plate_conf_in_vehicle=float(det["plate_conf_in_vehicle"]),
                    sharpen=bool(pre.get("sharpen", False)),
                    sharpen_amount=float(pre.get("sharpen_amount", 1.5)),
                    sharpen_sigma=float(pre.get("sharpen_sigma", 1.0)),
                    vehicle_crop_scale=float(pre.get("vehicle_crop_scale", 1.0)),
                    collect_rejected=False,
                )

                flagged = []
                known = covered.get(frame_num, []) if covered else []
                for p in plates:
                    conf = float(p[4])
                    if conf < args.audit_conf:
                        continue
                    pw, ph = p[2] - p[0], p[3] - p[1]
                    if pw < args.min_w or ph < args.min_h:
                        continue
                    if known and any(
                            _overlaps(p, kp, margin=8) for kp in known):
                        continue
                    det_total += 1
                    var, crop = laplacian_var(p, frame)
                    all_vars.append(var)
                    if var > args.var_thresh:
                        flagged.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
                        if args.out_crops and crop is not None:
                            name = os.path.join(
                                args.out_crops, f"f{frame_num:06d}_v{var:.0f}.jpg")
                            cv2.imwrite(name, crop)

                if flagged:
                    fixlist[str(frame_num)] = merge_rects(flagged)

                frame_num += 1
                pbar.update(1)
                pbar.set_postfix(dets=det_total, flagged=len(fixlist),
                                 refresh=False)
    finally:
        proc.stdout.close()
        proc.wait()

    with open(args.fixlist, "w", encoding="utf-8") as fh:
        json.dump(fixlist, fh, indent=0)

    arr = np.array(all_vars, dtype=float)
    print(f"\n  Frames     : {frame_num}")
    print(f"  Plates seen: {det_total}")
    print(f"  Flagged    : {len(fixlist)} frames  -> {args.fixlist}")
    if arr.size:
        for q in (50, 75, 90, 95, 99):
            print(f"    var p{q:>3}: {np.percentile(arr, q):9.1f}")
        print(f"    var max : {arr.max():9.1f}")

    if len(fixlist) <= 100:
        runs = []
        idxs = sorted(int(k) for k in fixlist)
        for k in idxs:
            if runs and k == runs[-1][1] + 1:
                runs[-1][1] = k
            else:
                runs.append([k, k])
        for a, b in runs:
            t0, t1 = a / fps, b / fps
            print(f"    run {a:6d}-{b:6d}  ({t0:7.2f}s-{t1:7.2f}s)"
                  f"  {b - a + 1:4d} frames")


if __name__ == "__main__":
    sys.exit(main())
