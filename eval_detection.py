"""Detection-quality metrics over a detection cache (no tracking).

Measures how often a motorcycle box contains a geometrically plausible plate
detection. Use this to A/B detection changes against byte-identical or freshly
built caches — tracking / blur settings do not affect these numbers.

    python eval_detection.py cache/ref_200_230.jsonl --json cache/det_baseline_ref.json

Note: changing detect_scale, SAHI, crop upscale, or moto ROI invalidates a cache
(see plates/detcache.py _META_KEYS). Write a new path such as
cache/ref_200_230_det_v2.jsonl instead of overwriting the baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from plates.common import _overlaps
from plates.detcache import DetectionCache
from plates.detect import plate_in_moto_geometry_ok as _plate_geometry_ok

_MOTO_CLS = 3
_KEY_SECONDS = (9.0, 18.0, 24.0, 28.0)


def plate_geometry_ok(plate, box, ry_lo=0.25, ry_hi=0.78,
                      min_rw=0.08, min_rh=0.05) -> bool:
    return _plate_geometry_ok(plate, box, ry_lo=ry_lo, ry_hi=ry_hi,
                              min_rw=min_rw, min_rh=min_rh)


def _tightest_moto(plate, vehicles):
    """Smallest motorcycle box containing the plate centre, or None."""
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
            best = (area, (vx1, vy1, vx2, vy2))
    return None if best is None else best[1]


def evaluate(cache_path: str, fps: float = 29.97, conf_floor: float = 0.15,
             min_plate_px: int = 8) -> dict:
    with open(cache_path, "r", encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
    cache = DetectionCache(cache_path, meta)
    if not cache.reading:
        sys.exit(f"ERROR: {cache_path} is empty")

    n_frames = 0
    moto_frames = 0
    moto_with_any = 0
    moto_with_geo = 0
    by_source = {}
    plate_sizes = []
    geo_sizes = []
    miss_examples = []   # frames with moto but no geo plate
    key_hits = {s: {"moto": 0, "geo": 0, "plates": []} for s in _KEY_SECONDS}

    for frame_num in cache.frame_indices():
        plates, vehicles = cache.get(frame_num)
        n_frames += 1
        motos = [v for v in vehicles if v[0] == _MOTO_CLS]
        if not motos:
            continue
        moto_frames += 1
        t = frame_num / fps

        in_box = []
        geo_ok = []
        for p in plates:
            if len(p) < 5 or p[4] < conf_floor:
                continue
            w = p[2] - p[0]
            h = p[3] - p[1]
            if w < min_plate_px or h < 4:
                continue
            box = _tightest_moto(p, vehicles)
            if box is None:
                # also count plates overlapping any moto by IoU-ish overlap
                for v in motos:
                    if _overlaps(p[:4], v[1:5]):
                        box = v[1:5]
                        break
            if box is None:
                continue
            src = p[5] if len(p) > 5 else "sahi"
            by_source[src] = by_source.get(src, 0) + 1
            plate_sizes.append((w, h))
            in_box.append(p)
            if plate_geometry_ok(p, box):
                geo_ok.append(p)
                geo_sizes.append((w, h))

        if in_box:
            moto_with_any += 1
        if geo_ok:
            moto_with_geo += 1
        else:
            if len(miss_examples) < 24:
                miss_examples.append({
                    "frame": frame_num,
                    "t": round(t, 2),
                    "n_moto": len(motos),
                    "n_plate_any": len(in_box),
                })

        for s in _KEY_SECONDS:
            if abs(t - s) <= 0.5:
                key_hits[s]["moto"] += 1
                if geo_ok:
                    key_hits[s]["geo"] += 1
                    key_hits[s]["plates"].extend(
                        [round(p[4], 3) for p in geo_ok[:3]])

    def _med(vals):
        return round(statistics.median(vals), 1) if vals else None

    report = {
        "cache": cache_path,
        "meta": {k: meta.get(k) for k in (
            "detect_scale", "vehicle_crop_scale", "sahi_slice_size",
            "plate_conf_in_vehicle", "moto_crop_scale", "moto_crop_bottom_frac",
            "moto_crop_side_pad_frac",
        )},
        "frames": n_frames,
        "moto_frames": moto_frames,
        "moto_with_any_plate": moto_with_any,
        "moto_with_geo_plate": moto_with_geo,
        "recall_any": round(moto_with_any / moto_frames, 4) if moto_frames else 0.0,
        "recall_geo": round(moto_with_geo / moto_frames, 4) if moto_frames else 0.0,
        "plates_by_source": by_source,
        "plate_size_median_wh": (
            [_med([w for w, _ in plate_sizes]), _med([h for _, h in plate_sizes])]
            if plate_sizes else None),
        "geo_plate_size_median_wh": (
            [_med([w for w, _ in geo_sizes]), _med([h for _, h in geo_sizes])]
            if geo_sizes else None),
        "key_seconds": {
            str(s): {
                "moto_frames": key_hits[s]["moto"],
                "geo_frames": key_hits[s]["geo"],
                "sample_confs": key_hits[s]["plates"][:8],
            } for s in _KEY_SECONDS
        },
        "miss_examples": miss_examples[:16],
        "note": (
            "Changing detect_scale / SAHI / crop / moto ROI invalidates a cache "
            "(_META_KEYS in plates/detcache.py). Write a new path instead of "
            "overwriting the baseline."
        ),
    }
    # Drop empty meta keys
    report["meta"] = {k: v for k, v in report["meta"].items() if v is not None}
    return report


def _print(report: dict) -> None:
    print("=" * 62)
    print(f"  Detection quality — {report['cache']}")
    print("=" * 62)
    print(f"  Frames              : {report['frames']}")
    print(f"  Moto frames         : {report['moto_frames']}")
    print(f"  With any in-box pl. : {report['moto_with_any_plate']}  "
          f"(recall_any={report['recall_any']})")
    print(f"  With geo-OK plate   : {report['moto_with_geo_plate']}  "
          f"(recall_geo={report['recall_geo']})")
    src = report["plates_by_source"]
    print(f"  Plates by source    : "
          + (", ".join(f"{k}={v}" for k, v in sorted(src.items())) or "-"))
    print(f"  Plate size median   : {report['plate_size_median_wh']}")
    print(f"  Geo plate size med. : {report['geo_plate_size_median_wh']}")
    print("  Key seconds (±0.5s) :")
    for s, info in report["key_seconds"].items():
        print(f"    t={s:>4}s  moto_frames={info['moto_frames']}  "
              f"geo_frames={info['geo_frames']}  "
              f"confs={info['sample_confs']}")
    print("=" * 62)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="Detection cache JSONL")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Also write the report as JSON")
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--conf-floor", type=float, default=0.15)
    args = ap.parse_args()

    report = evaluate(args.cache, fps=args.fps, conf_floor=args.conf_floor)
    _print(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  Metrics written to {args.json_out}")


if __name__ == "__main__":
    main()
