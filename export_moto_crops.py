"""Export motorcycle rear-ROI crops for fine-tune dataset building (phase 3).

Saves positive crops (frame + moto with a geo-OK plate) and hard-negatives
(taillight / top-box region when no geo plate). Labels are YOLO-format txt
relative to the exported crop.

    python export_moto_crops.py testvideo/montage....mp4 cache/ref_200_230_det_v2.jsonl \
        --out dataset/v2m --start 2:00 --end 2:30 --max-pos 400 --max-neg 200

Does not train a model — only prepares crops + labels for Ultralytics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

from plates.common import time_to_seconds
from plates.detcache import DetectionCache
from plates.detect import moto_rear_roi, plate_in_moto_geometry_ok

_MOTO_CLS = 3


def _tightest_moto(plate, vehicles):
    pcx = (plate[0] + plate[2]) * 0.5
    pcy = (plate[1] + plate[3]) * 0.5
    best = None
    for v in vehicles:
        if v[0] != _MOTO_CLS:
            continue
        vx1, vy1, vx2, vy2 = v[1], v[2], v[3], v[4]
        if not (vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2):
            continue
        area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
        if best is None or area < best[0]:
            best = (area, v)
    return None if best is None else best[1]


def _write_yolo_label(path, box_xyxy, crop_w, crop_h):
    x1, y1, x2, y2 = box_xyxy
    cx = ((x1 + x2) * 0.5) / crop_w
    cy = ((y1 + y2) * 0.5) / crop_h
    bw = (x2 - x1) / crop_w
    bh = (y2 - y1) / crop_h
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def export(video_path, cache_path, out_dir, start=None, end=None,
           bottom_frac=0.28, side_pad=0.05, max_pos=400, max_neg=200,
           every_n=3):
    with open(cache_path, "r", encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
    cache = DetectionCache(cache_path, meta)
    if not cache.reading:
        sys.exit(f"ERROR: empty cache {cache_path}")

    pos_dir = os.path.join(out_dir, "images", "pos")
    neg_dir = os.path.join(out_dir, "images", "neg")
    lbl_dir = os.path.join(out_dir, "labels", "pos")
    for d in (pos_dir, neg_dir, lbl_dir):
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    # Cache frame indices are relative to the clip start stored in meta.
    clip_start_s = float(meta.get("start") or 0.0)
    clip_start_f = int(round(clip_start_s * fps))
    abs_start_f = int(round(start * fps)) if start is not None else clip_start_f
    abs_end_f = int(round(end * fps)) if end is not None else int(1e9)

    n_pos = n_neg = 0
    for frame_num in cache.frame_indices():
        abs_frame = clip_start_f + frame_num
        if abs_frame < abs_start_f or abs_frame >= abs_end_f:
            continue
        if frame_num % every_n != 0:
            continue
        plates, vehicles = cache.get(frame_num)
        motos = [v for v in vehicles if v[0] == _MOTO_CLS]
        if not motos:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]

        for v in motos:
            if n_pos >= max_pos and n_neg >= max_neg:
                break
            vx1, vy1, vx2, vy2 = v[1], v[2], v[3], v[4]
            rx1, ry1, rx2, ry2 = moto_rear_roi(
                vx1, vy1, vx2, vy2, bottom_frac=bottom_frac,
                side_pad_frac=side_pad, frame_w=w, frame_h=h)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]

            geo_plates = []
            for p in plates:
                owner = _tightest_moto(p, vehicles)
                if owner is None:
                    continue
                if (owner[1], owner[2], owner[3], owner[4]) != (vx1, vy1, vx2, vy2):
                    continue
                if plate_in_moto_geometry_ok(p, (vx1, vy1, vx2, vy2)):
                    lx1 = max(0, p[0] - rx1)
                    ly1 = max(0, p[1] - ry1)
                    lx2 = min(cw, p[2] - rx1)
                    ly2 = min(ch, p[3] - ry1)
                    if lx2 > lx1 and ly2 > ly1:
                        geo_plates.append((lx1, ly1, lx2, ly2, p[4]))

            stem = f"f{abs_frame:06d}_m{vx1}_{vy1}"
            if geo_plates and n_pos < max_pos:
                best = max(geo_plates, key=lambda t: t[4])
                img_path = os.path.join(pos_dir, stem + ".jpg")
                cv2.imwrite(img_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                _write_yolo_label(os.path.join(lbl_dir, stem + ".txt"),
                                  best[:4], cw, ch)
                n_pos += 1
            elif not geo_plates and n_neg < max_neg:
                img_path = os.path.join(neg_dir, stem + ".jpg")
                cv2.imwrite(img_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                n_neg += 1

    cap.release()
    manifest = {
        "video": os.path.basename(video_path),
        "cache": cache_path,
        "positives": n_pos,
        "hard_negatives": n_neg,
        "note": (
            "Train with Ultralytics YOLO11n (or reuse v1m family) at 640, "
            "augment with blur/JPEG/angle. Hold out a val split with no frame "
            "overlap vs train. Output: license-plate-finetune-v2m.pt"
        ),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Exported {n_pos} positives + {n_neg} hard-negatives → {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("cache")
    ap.add_argument("--out", default="dataset/v2m")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-pos", type=int, default=400)
    ap.add_argument("--max-neg", type=int, default=200)
    ap.add_argument("--every", type=int, default=3)
    args = ap.parse_args()
    start = time_to_seconds(args.start) if args.start else None
    end = time_to_seconds(args.end) if args.end else None
    export(args.video, args.cache, args.out, start=start, end=end,
           max_pos=args.max_pos, max_neg=args.max_neg, every_n=args.every)


if __name__ == "__main__":
    main()
