"""Measure blur-zone stability from a detection cache.

Replays the tracker over cached detections (see --detect-cache) and reports the
three failure modes that are visible in the output video:

  coverage  — a plate the detector found but no emitted zone covers: privacy
              miss. Reported both as "uncovered" and "partial" (< 90 % of the
              plate area inside a zone).
  offset    — distance between the plate centre and the centre of the zone that
              covers it, in units of plate width. 0.5 means the blur sits half a
              plate off-centre, which is what "the blur drifts off the plate"
              looks like frame by frame.
  jitter    — magnitude of the second difference of the zone centre along a
              track, in px. A steady pan has a near-zero second difference, so
              this isolates wobble/oscillation from legitimate motion.
  blinks    — a track that emitted a zone, stopped for 1..3 frames, then
              resumed: the flicker the eye notices most.

No inference and no encoding: a 30 s clip is measured in seconds, so temporal
filters can be tuned against numbers instead of impressions.

Usage:
  python eval_stability.py cache/ref_200_230.jsonl
  python eval_stability.py cache/ref_200_230.jsonl --json baseline.json
  python eval_stability.py cache/ref_200_230.jsonl --with-frames   # exact parity
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

import numpy as np

from plates.common import covered_fraction
from plates.config import load_config
from plates.detcache import DetectionCache
from plates.ffmpeg import build_ffmpeg_extract, get_video_info
from plates.track import SceneTracker


# A plate is considered covered when at least this fraction of its area falls
# inside a single emitted zone. Below it, part of the plate stays readable.
_COVER_FULL = 0.90
# Blink = emission gap of at most this many frames between two emissions of the
# same track. Longer gaps read as "the blur ended", not as flicker.
_BLINK_MAX = 3


def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


_covered_frac = covered_fraction   # same definition the tracker measures with


_KF_KEYS = (
    "kf_process_pos", "kf_process_size", "kf_meas_pos", "kf_meas_size",
    "kf_gate_max", "kf_max_rejects", "kf_sigma_pad_k", "kf_sigma_pad_max",
    "kf_pad_decay", "kf_vel_decay", "kf_anchor_meas_scale",
)


def _tracker_from_config(cfg, zone_filter=None, overrides=None):
    """Build a SceneTracker with the same settings blur_plates.py would use."""
    det = cfg["detection"]
    trk = cfg.get("tracking", {})
    blr = cfg["blur"]
    kf_params = {k: float(trk[k]) for k in _KF_KEYS if k in trk}
    kwargs = dict(
        zone_filter=zone_filter or str(trk.get("zone_filter", "ema")),
        **kf_params,
        max_gap_frames=int(trk.get("max_gap_frames", 8)),
        history_frames=int(trk.get("history_frames", 15)),
        min_vehicle_conf=float(trk.get("min_vehicle_conf", 0.60)),
        standalone_min_ar=float(det.get("standalone_min_ar", 1.2)),
        standalone_max_ar=float(det.get("standalone_max_ar", 6.0)),
        predict_expand_max=int(trk.get("predict_expand_max", 20)),
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
        emit_max_disp=float(trk.get("emit_max_disp", 80.0)),
    )
    # Tracking parameters live outside the detection cache, so any of them can be
    # swept against byte-identical detections without re-running inference.
    unknown = sorted(set(overrides or ()) - set(kwargs))
    if unknown:
        sys.exit(f"ERROR: unknown tracker parameter(s): {', '.join(unknown)}")
    kwargs.update(overrides or {})
    return SceneTracker(**kwargs)


def _frame_reader(meta, width, height):
    """Yield decoded BGR frames for the cached clip (exact LK parity)."""
    path = meta.get("input")
    candidates = [path, os.path.join("testvideo", path or "")]
    src = next((c for c in candidates if c and os.path.exists(c)), None)
    if src is None:
        sys.exit(f"ERROR: --with-frames needs the source video; {path!r} not found")
    cmd = build_ffmpeg_extract(src, meta.get("start"), meta.get("end"))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(nbytes)
            if len(raw) < nbytes:
                return
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
    finally:
        proc.stdout.close()
        proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="Detection cache written by --detect-cache")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Also write the metrics as JSON (for A/B comparisons)")
    ap.add_argument("--with-frames", dest="with_frames", action="store_true",
                    help="Decode the source video so the tracker's optical-flow "
                         "gap-fill runs exactly as in a real run (slower)")
    ap.add_argument("--zone-filter", dest="zone_filter", default=None,
                    choices=["ema", "kalman"],
                    help="Override [tracking] zone_filter, to A/B two temporal "
                         "filters against byte-identical detections")
    ap.add_argument("--kf", dest="kf", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Override any SceneTracker parameter for this run, "
                         "repeatable (e.g. kf_process_pos=8, min_vehicle_conf=0.3). "
                         "Tracking settings are not part of the detection cache, "
                         "so a sweep replays the same detections in seconds.")
    ap.add_argument("--min-plate-px", dest="min_plate_px", type=int, default=12,
                    help="Detections narrower than this (px) are reported as "
                         "micro detections instead of coverage misses: they carry "
                         "no readable characters and their width-normalised "
                         "offset would swamp the worst-case figures (default 12)")
    ap.add_argument("--min-plate-h", dest="min_plate_h", type=int, default=6,
                    help="Same treatment for detections shorter than this (px): "
                         "below roughly 6 px of character height nothing is "
                         "legible, whatever the width (default 6)")
    ap.add_argument("--conf-floor", dest="conf_floor", type=float, default=None,
                    help="Coverage is required for detections at or above this "
                         "confidence (default: plate_conf_in_vehicle from config)")
    args = ap.parse_args()

    cfg = load_config()
    floor = (args.conf_floor if args.conf_floor is not None
             else float(cfg["detection"]["plate_conf_in_vehicle"]))
    min_plate_px = args.min_plate_px
    min_plate_h  = args.min_plate_h

    # The cache header is the ground truth for geometry: read it directly so the
    # metrics never depend on re-probing the video.
    with open(args.cache, "r", encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
    width, height = int(meta["width"]), int(meta["height"])

    cache = DetectionCache(args.cache, meta)
    if not cache.reading:
        sys.exit(f"ERROR: {args.cache} is empty")

    overrides = {}
    for item in args.kf:
        key, _, value = item.partition("=")
        overrides[key.strip()] = float(value)

    tracker = _tracker_from_config(cfg, args.zone_filter, overrides)
    frames = _frame_reader(meta, width, height) if args.with_frames else None

    n_frames      = 0
    n_zones       = 0
    uncovered     = []      # (frame, conf, plate)
    partial       = []      # (frame, conf, covered_frac)
    micro         = []      # (frame, conf, w, h) — sub-min_plate_px detections
    offsets       = []      # |Δcentre| / plate width
    by_source     = {}
    standalone    = 0
    emitted_at    = {}      # tid -> sorted list of frame indices
    centres       = {}      # tid -> {frame: (cx, cy, w, h)}

    for frame_num in cache.frame_indices():
        plates_raw, vehicles = cache.get(frame_num)
        frame = next(frames, None) if frames is not None else None
        zones, _ = tracker.update(vehicles, list(plates_raw), frame,
                                  frame_size=(width, height))
        n_frames += 1
        n_zones  += len(zones)

        for z in zones:
            source = z[5] if len(z) > 5 else "sahi"
            tid    = z[6] if len(z) > 6 else None
            by_source[source] = by_source.get(source, 0) + 1
            if tid is None:
                standalone += 1
                continue
            if source == "patch":
                # Guaranteed-coverage rectangle over a detection the smoothed
                # zone missed: not part of the track's trajectory, so including
                # it would report the filter/detector disagreement as jitter.
                continue
            emitted_at.setdefault(tid, []).append(frame_num)
            centres.setdefault(tid, {})[frame_num] = (
                (z[0] + z[2]) * 0.5, (z[1] + z[3]) * 0.5,
                z[2] - z[0], z[3] - z[1],
            )

        # Coverage is judged against what the detector actually saw this frame:
        # anything it can read, the blur must cover.
        for p in plates_raw:
            conf = p[4] if len(p) > 4 else 1.0
            if conf < floor:
                continue
            best_frac, best_zone = 0.0, None
            for z in zones:
                frac = _covered_frac(p[:4], z[:4])
                if frac > best_frac:
                    best_frac, best_zone = frac, z
            # Detections a few px wide are detector noise: they hold no readable
            # characters, and normalising their offset by their own width turns a
            # 150 px distance into "150 plate widths", which would otherwise
            # dominate every worst-case figure. Counted separately, never hidden.
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

    # ── Jitter: second difference of the zone centre on consecutive frames ────
    jitter_pos, jitter_size, blinks = [], [], []
    for tid, seq in centres.items():
        fs = sorted(seq)
        for a, b, c in zip(fs, fs[1:], fs[2:]):
            if b - a != 1 or c - b != 1:
                continue        # only consecutive frames: gaps are not jitter
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
        "cache":            os.path.basename(args.cache),
        "zone_filter":      tracker.zone_filter,
        "kf_overrides":     overrides,
        "frames":           n_frames,
        "conf_floor":       floor,
        "with_frames":      bool(args.with_frames),
        "zones_total":      n_zones,
        "zones_per_frame":  round(n_zones / n_frames, 3) if n_frames else 0.0,
        "zones_by_source":  by_source,
        "zones_untracked":  standalone,
        "min_plate_px":     min_plate_px,
        "min_plate_h":      min_plate_h,
        "coverage": {
            "uncovered":        len(uncovered),
            "partial":          len(partial),
            "uncovered_frames": sorted({f for f, _, _ in uncovered})[:40],
            "micro_missed":     len(micro),
            "micro_sizes":      [f"{w}x{h}" for _, _, w, h in micro][:12],
        },
        "offset_plate_widths": {
            "median": round(statistics.median(offsets), 3) if offsets else 0.0,
            "p95":    round(pct(offsets, 95), 3),
            "max":    round(max(offsets), 3) if offsets else 0.0,
        },
        "jitter_px": {
            "median": round(statistics.median(jitter_pos), 2) if jitter_pos else 0.0,
            "p95":    round(pct(jitter_pos, 95), 2),
            "max":    round(max(jitter_pos), 2) if jitter_pos else 0.0,
            "samples": len(jitter_pos),
        },
        "jitter_size_rel": {
            "median": round(statistics.median(jitter_size), 3) if jitter_size else 0.0,
            "p95":    round(pct(jitter_size, 95), 3),
        },
        "blinks": {
            "count":  len(blinks),
            "frames": [f for _, f, _ in blinks][:40],
        },
        "tracks": len(centres),
    }

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  Blur-zone stability — {report['cache']}")
    print(f"{bar}")
    print(f"  Zone filter         : {report['zone_filter']}")
    print(f"  Frames              : {report['frames']}")
    print(f"  Zones               : {report['zones_total']} "
          f"({report['zones_per_frame']} per frame)  |  tracks: {report['tracks']}")
    print(f"  Zones by source     : "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    print(f"  Untracked zones     : {report['zones_untracked']}  "
          f"(no track id — cannot be smoothed or gap-filled)")
    print(f"  Coverage misses     : {report['coverage']['uncovered']} uncovered, "
          f"{report['coverage']['partial']} partial (<{int(_COVER_FULL*100)}% of plate"
          f", plates >= {min_plate_px}x{min_plate_h}px)")
    print(f"  Micro detections    : {report['coverage']['micro_missed']} missed "
          f"below {min_plate_px}x{min_plate_h}px  (no readable characters)"
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
