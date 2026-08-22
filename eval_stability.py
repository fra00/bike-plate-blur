"""Measure blur-zone stability from a detection cache.

Builds interpolated zones from cached detections (same path as blur_plates.py)
and reports:

  coverage  — a plate the detector found but no emitted zone covers
  offset    — distance between plate centre and covering zone, in plate widths
  jitter    — second difference of the zone centre along a track (wobble)
  blinks    — a track that emitted, stopped for 1–3 frames, then resumed

No inference and no encoding.

Usage:
  python eval_stability.py cache/ref_200_230.jsonl
  python eval_stability.py cache/ref_200_230.jsonl --json baseline.json
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np

from plates.common import covered_fraction
from plates.config import load_config
from plates.constants import VEHICLE_CLASSES, VEHICLE_FILTER_MAP
from plates.detcache import DetectionCache
from plates.track import build_zones


_COVER_FULL = 0.90
_BLINK_MAX = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="Detection cache written by --detect-cache")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Also write the metrics as JSON (for A/B comparisons)")
    ap.add_argument("--min-plate-px", dest="min_plate_px", type=int, default=12)
    ap.add_argument("--min-plate-h", dest="min_plate_h", type=int, default=6)
    ap.add_argument("--conf-floor", dest="conf_floor", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    det = cfg["detection"]
    zn = cfg.get("zones", {})
    floor = (args.conf_floor if args.conf_floor is not None
             else float(zn.get("conf_floor", det["plate_conf_in_vehicle"])))
    min_plate_px = args.min_plate_px
    min_plate_h = args.min_plate_h

    with open(args.cache, "r", encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
    width, height = int(meta["width"]), int(meta["height"])

    cache = DetectionCache(args.cache, meta)
    if not cache.reading:
        sys.exit(f"ERROR: {args.cache} is empty")

    vehicle_filter = meta.get("vehicle_filter", "all")
    filter_classes = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
    zones_map = build_zones(
        cache,
        max_gap_frames=int(zn.get("max_gap_frames", 15)),
        max_disp_px=float(zn.get("max_disp_px", 80.0)),
        max_disp_frac=float(zn.get("max_disp_frac", 0.35)),
        conf_floor=floor,
        filter_classes=filter_classes,
        min_moto_h_frac=float(zn.get("moto_min_blur_box_h_frac", 0.08333)),
        frame_height=height,
        min_vehicle_iou=float(zn.get("min_vehicle_iou", 0.15)),
        moto_size_enter_frac=float(zn.get("moto_size_enter_frac", 1.15)),
        moto_size_exit_frac=float(zn.get("moto_size_exit_frac", 0.85)),
        min_plate_side_px=float(zn.get("min_plate_side_px", 12.0)),
        max_class_flip_frames=int(zn.get("max_class_flip_frames", 10)),
        max_area_ratio=float(zn.get("max_area_ratio", 2.5)),
    )

    n_frames = 0
    n_zones = 0
    uncovered = []
    partial = []
    micro = []
    offsets = []
    by_source = {}
    emitted_at = {}
    centres = {}

    for frame_num in cache.frame_indices():
        plates_raw, _vehicles = cache.get(frame_num)
        zones = zones_map.get(frame_num, [])
        n_frames += 1
        n_zones += len(zones)

        for z in zones:
            source = z[5] if len(z) > 5 else "sahi"
            tid = z[6] if len(z) > 6 else None
            by_source[source] = by_source.get(source, 0) + 1
            if tid is None:
                continue
            emitted_at.setdefault(tid, []).append(frame_num)
            centres.setdefault(tid, {})[frame_num] = (
                (z[0] + z[2]) * 0.5, (z[1] + z[3]) * 0.5,
                z[2] - z[0], z[3] - z[1],
            )

        for p in plates_raw:
            conf = p[4] if len(p) > 4 else 1.0
            if conf < floor:
                continue
            best_frac, best_zone = 0.0, None
            for z in zones:
                frac = covered_fraction(p[:4], z[:4])
                if frac > best_frac:
                    best_frac, best_zone = frac, z
            if p[2] - p[0] < min_plate_px or p[3] - p[1] < min_plate_h:
                if best_frac < _COVER_FULL:
                    micro.append((frame_num, round(conf, 3),
                                  int(p[2] - p[0]), int(p[3] - p[1])))
                continue
            if best_frac <= 0.0:
                uncovered.append((frame_num, round(conf, 3), tuple(p[:4])))
            elif best_frac < _COVER_FULL:
                partial.append((frame_num, round(conf, 3), round(best_frac, 3)))
            if best_zone is not None:
                pw = max(1.0, p[2] - p[0])
                dx = ((p[0] + p[2]) - (best_zone[0] + best_zone[2])) * 0.5
                dy = ((p[1] + p[3]) - (best_zone[1] + best_zone[3])) * 0.5
                offsets.append(float(np.hypot(dx, dy)) / pw)

    jitter_pos, jitter_size, blinks = [], [], []
    for tid, seq in centres.items():
        fs = sorted(seq)
        for a, b, c in zip(fs, fs[1:], fs[2:]):
            if b - a != 1 or c - b != 1:
                continue
            (x0, y0, w0, _), (x1, y1, w1, _), (x2, y2, w2, _) = seq[a], seq[b], seq[c]
            jitter_pos.append(float(np.hypot(x2 - 2 * x1 + x0, y2 - 2 * y1 + y0)))
            jitter_size.append(abs(w2 - 2 * w1 + w0) / max(1.0, w1))
        for prev, nxt in zip(fs, fs[1:]):
            gap = nxt - prev - 1
            if 1 <= gap <= _BLINK_MAX:
                blinks.append((tid, prev, gap))

    def pct(values, q):
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    report = {
        "cache": os.path.basename(args.cache),
        "frames": n_frames,
        "conf_floor": floor,
        "zones_total": n_zones,
        "zones_per_frame": round(n_zones / n_frames, 3) if n_frames else 0.0,
        "zones_by_source": by_source,
        "min_plate_px": min_plate_px,
        "min_plate_h": min_plate_h,
        "coverage": {
            "uncovered": len(uncovered),
            "partial": len(partial),
            "uncovered_frames": sorted({f for f, _, _ in uncovered})[:40],
            "micro_missed": len(micro),
            "micro_sizes": [f"{w}x{h}" for _, _, w, h in micro][:12],
        },
        "offset_plate_widths": {
            "median": round(statistics.median(offsets), 3) if offsets else 0.0,
            "p95": round(pct(offsets, 95), 3),
            "max": round(max(offsets), 3) if offsets else 0.0,
        },
        "jitter_px": {
            "median": round(statistics.median(jitter_pos), 2) if jitter_pos else 0.0,
            "p95": round(pct(jitter_pos, 95), 2),
            "max": round(max(jitter_pos), 2) if jitter_pos else 0.0,
            "samples": len(jitter_pos),
        },
        "jitter_size_rel": {
            "median": round(statistics.median(jitter_size), 3) if jitter_size else 0.0,
            "p95": round(pct(jitter_size, 95), 3),
        },
        "blinks": {
            "count": len(blinks),
            "frames": [f for _, f, _ in blinks][:40],
        },
        "tracks": len(centres),
    }

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  Blur-zone stability — {report['cache']}")
    print(f"{bar}")
    print(f"  Frames              : {report['frames']}")
    print(f"  Zones               : {report['zones_total']} "
          f"({report['zones_per_frame']} per frame)  |  tracks: {report['tracks']}")
    print(f"  Zones by source     : "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    print(f"  Coverage misses     : {report['coverage']['uncovered']} uncovered, "
          f"{report['coverage']['partial']} partial (<{int(_COVER_FULL*100)}% of plate"
          f", plates >= {min_plate_px}x{min_plate_h}px)")
    print(f"  Micro detections    : {report['coverage']['micro_missed']} missed "
          f"below {min_plate_px}x{min_plate_h}px"
          + (f"  e.g. {', '.join(report['coverage']['micro_sizes'][:6])}"
             if report['coverage']['micro_sizes'] else ""))
    print(f"  Centre offset       : median {report['offset_plate_widths']['median']}, "
          f"p95 {report['offset_plate_widths']['p95']}, "
          f"max {report['offset_plate_widths']['max']}  (plate widths)")
    print(f"  Jitter (wobble)     : median {report['jitter_px']['median']} px, "
          f"p95 {report['jitter_px']['p95']} px, "
          f"max {report['jitter_px']['max']} px  "
          f"({report['jitter_px']['samples']} samples)")
    print(f"  Size jitter         : median {report['jitter_size_rel']['median']}, "
          f"p95 {report['jitter_size_rel']['p95']}  (relative to zone width)")
    print(f"  Blinks (1-{_BLINK_MAX} frames) : {report['blinks']['count']}")
    if report["coverage"]["uncovered_frames"]:
        print(f"  Uncovered on frames : {report['coverage']['uncovered_frames']}")
    if report["blinks"]["frames"]:
        print(f"  Blink after frames  : {report['blinks']['frames']}")
    print(f"{bar}\n")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  Metrics written to {args.json_out}\n")


if __name__ == "__main__":
    main()
