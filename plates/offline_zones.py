"""Offline gap fills: past + future bridges to patch tracker blinks.

Primary blur quality still comes from the online SceneTracker (Kalman /
smart moto zones). This module only proposes short interpolations between
cached plate detections when motion is plausible for the *same* vehicle.

Pipeline usage (hybrid):
  1. Run the tracker as usual on cached detections.
  2. If the tracker emitted nothing on a frame, attach bridge fills for
     that frame (``merge_tracker_with_offline_fills``).
  3. Never replace a tracker zone with a raw/bridged box — that was what
     made pure ``--offline-zones`` look worse than Kalman.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from plates.common import _overlaps, iou
from plates.detect import plate_in_moto_geometry_ok

_MOTO_CLS = 3


@dataclass
class _Det:
    frame: int
    box: tuple          # (x1, y1, x2, y2)
    conf: float
    source: str
    vehicle: tuple | None  # (x1, y1, x2, y2) or None


@dataclass
class _Tracklet:
    dets: list = field(default_factory=list)  # sorted by frame
    tid: int = 0


def _centre(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _diag(box):
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _dist(a, b):
    ca, cb = _centre(a), _centre(b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def _lerp_box(a, b, t: float) -> tuple:
    """Linear interpolate two xyxy boxes; *t* in [0, 1]."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))


def _plate_rel(plate, vehicle):
    """Plate box as fractions of the vehicle box (cx, cy, w, h)."""
    vx1, vy1, vx2, vy2 = vehicle
    vw = max(1.0, vx2 - vx1)
    vh = max(1.0, vy2 - vy1)
    pcx = (plate[0] + plate[2]) * 0.5
    pcy = (plate[1] + plate[3]) * 0.5
    pw = plate[2] - plate[0]
    ph = plate[3] - plate[1]
    return (
        (pcx - vx1) / vw,
        (pcy - vy1) / vh,
        pw / vw,
        ph / vh,
    )


def _plate_from_rel(rel, vehicle) -> tuple:
    rx, ry, rw, rh = rel
    vx1, vy1, vx2, vy2 = vehicle
    vw = max(1.0, vx2 - vx1)
    vh = max(1.0, vy2 - vy1)
    pcx = vx1 + rx * vw
    pcy = vy1 + ry * vh
    pw = max(4.0, rw * vw)
    ph = max(4.0, rh * vh)
    return (
        int(round(pcx - pw * 0.5)),
        int(round(pcy - ph * 0.5)),
        int(round(pcx + pw * 0.5)),
        int(round(pcy + ph * 0.5)),
    )


def _lerp_rel(ra, rb, t: float):
    return tuple(ra[i] + (rb[i] - ra[i]) * t for i in range(4))


def _same_vehicle(va, vb, min_iou: float = 0.15) -> bool:
    if va is None or vb is None:
        return False
    return iou(va, vb) >= min_iou or _overlaps(va, vb)


def _tightest_vehicle(plate, vehicles, filter_classes=None):
    """Smallest vehicle box containing the plate centre (optional class filter)."""
    pcx = (plate[0] + plate[2]) * 0.5
    pcy = (plate[1] + plate[3]) * 0.5
    best = None
    for v in vehicles:
        cls = v[0]
        if filter_classes is not None and cls not in filter_classes:
            continue
        vx1, vy1, vx2, vy2 = v[1], v[2], v[3], v[4]
        if not (vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2):
            continue
        area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
        if best is None or area < best[0]:
            best = (area, (vx1, vy1, vx2, vy2), cls)
    if best is None:
        return None, None
    return best[1], best[2]


def motion_gate_ok(box_a, box_b, gap_frames: int, vehicle=None,
                   max_disp_px: float = 80.0, max_disp_frac: float = 0.35,
                   margin: float = 1.15) -> bool:
    """True if A→B displacement is plausible over *gap_frames* missing frames.

    Allowed travel scales with gap length and vehicle size (when known):
      allowed = max(max_disp_px, vehicle_diag * max_disp_frac) * gap_frames * margin
    """
    if gap_frames <= 0:
        return True
    dist = _dist(box_a, box_b)
    base = float(max_disp_px)
    if vehicle is not None:
        base = max(base, _diag(vehicle) * max_disp_frac)
    allowed = base * gap_frames * margin
    return dist <= allowed


def _collect_dets(cache, conf_floor: float, moto_only: bool,
                  min_blur_box_h_frac: float, frame_height: int):
    """Flatten cache into per-frame detection lists."""
    by_frame: dict[int, list[_Det]] = {}
    filter_classes = {_MOTO_CLS} if moto_only else None
    for fi in cache.frame_indices():
        plates, vehicles = cache.get(fi)
        dets = []
        for p in plates:
            if len(p) < 5 or p[4] < conf_floor:
                continue
            box = tuple(int(p[i]) for i in range(4))
            conf = float(p[4])
            src = p[5] if len(p) > 5 else "sahi"
            veh, vcls = _tightest_vehicle(p, vehicles, filter_classes)
            if moto_only:
                if veh is None:
                    for v in vehicles:
                        if v[0] != _MOTO_CLS:
                            continue
                        if _overlaps(box, v[1:5]) and plate_in_moto_geometry_ok(box, v[1:5]):
                            veh, vcls = v[1:5], _MOTO_CLS
                            break
                if veh is None:
                    continue
                if (min_blur_box_h_frac > 0 and frame_height > 0
                        and (veh[3] - veh[1]) < min_blur_box_h_frac * frame_height):
                    continue
                if not plate_in_moto_geometry_ok(box, veh):
                    continue
            dets.append(_Det(fi, box, conf, str(src), veh))
        by_frame[fi] = dets
    return by_frame


def _build_tracklets(by_frame: dict, max_gap: int, max_disp_px: float,
                     max_disp_frac: float,
                     min_vehicle_iou: float = 0.15) -> list[_Tracklet]:
    """Greedy temporal association into tracklets."""
    frames = sorted(by_frame)
    tracklets: list[_Tracklet] = []
    next_tid = 1

    for fi in frames:
        for det in by_frame[fi]:
            best = None  # (cost, tracklet)
            for tr in tracklets:
                last = tr.dets[-1]
                gap = fi - last.frame - 1
                if gap < 0 or gap > max_gap:
                    continue
                if last.vehicle is not None and det.vehicle is not None:
                    if not _same_vehicle(last.vehicle, det.vehicle, min_vehicle_iou):
                        continue
                elif last.vehicle is not None or det.vehicle is not None:
                    # One side missing vehicle → demand tighter plate motion
                    if not motion_gate_ok(
                            last.box, det.box, max(gap, 1),
                            vehicle=last.vehicle or det.vehicle,
                            max_disp_px=max_disp_px * 0.5,
                            max_disp_frac=max_disp_frac * 0.5):
                        continue
                veh = det.vehicle or last.vehicle
                if not motion_gate_ok(last.box, det.box, max(gap, 1),
                                      vehicle=veh,
                                      max_disp_px=max_disp_px,
                                      max_disp_frac=max_disp_frac):
                    continue
                cost = _dist(last.box, det.box)
                if best is None or cost < best[0]:
                    best = (cost, tr)
            if best is not None:
                best[1].dets.append(det)
            else:
                tracklets.append(_Tracklet(dets=[det], tid=next_tid))
                next_tid += 1
    return tracklets


def _bridge_tracklet(tr: _Tracklet, max_gap: int, max_disp_px: float,
                     max_disp_frac: float,
                     min_vehicle_iou: float = 0.15,
                     vehicles_by_frame: dict | None = None) -> dict[int, list]:
    """Return frame → list of zone tuples for this tracklet (raw + bridged)."""
    out: dict[int, list] = {}
    dets = sorted(tr.dets, key=lambda d: d.frame)
    for d in dets:
        out.setdefault(d.frame, []).append(
            (d.box[0], d.box[1], d.box[2], d.box[3], d.conf, "offline", tr.tid)
        )
    for i in range(len(dets) - 1):
        a, b = dets[i], dets[i + 1]
        gap = b.frame - a.frame - 1
        if gap <= 0 or gap > max_gap:
            continue
        if a.vehicle is not None and b.vehicle is not None:
            if not _same_vehicle(a.vehicle, b.vehicle, min_vehicle_iou):
                continue
        veh = a.vehicle or b.vehicle
        if not motion_gate_ok(a.box, b.box, gap, vehicle=veh,
                              max_disp_px=max_disp_px,
                              max_disp_frac=max_disp_frac):
            continue

        use_rel = a.vehicle is not None and b.vehicle is not None
        rel_a = _plate_rel(a.box, a.vehicle) if use_rel else None
        rel_b = _plate_rel(b.box, b.vehicle) if use_rel else None

        for f in range(a.frame + 1, b.frame):
            t = (f - a.frame) / (b.frame - a.frame)
            conf = a.conf * (1 - t) + b.conf * t
            if use_rel:
                veh_f = _lerp_box(a.vehicle, b.vehicle, t)
                # Prefer a live moto box on this frame when it matches the path
                if vehicles_by_frame is not None:
                    cands = vehicles_by_frame.get(f, [])
                    best_v = None
                    best_iou = 0.0
                    for v in cands:
                        if v[0] != _MOTO_CLS:
                            continue
                        vv = v[1:5]
                        ii = iou(vv, veh_f)
                        if ii > best_iou:
                            best_iou, best_v = ii, vv
                    if best_v is not None and best_iou >= 0.15:
                        veh_f = best_v
                box = _plate_from_rel(_lerp_rel(rel_a, rel_b, t), veh_f)
            else:
                box = _lerp_box(a.box, b.box, t)
            out.setdefault(f, []).append(
                (*box, conf, "bridge", tr.tid)
            )
    return out


def build_offline_zones(
    cache,
    *,
    max_gap_frames: int = 15,
    max_disp_px: float = 80.0,
    max_disp_frac: float = 0.35,
    conf_floor: float = 0.15,
    moto_only: bool = True,
    min_blur_box_h_frac: float = 0.10,
    frame_height: int = 1080,
    min_vehicle_iou: float = 0.15,
) -> dict[int, list]:
    """Build per-frame blur zones from a detection cache.

    Returns ``{frame_idx: [(x1,y1,x2,y2,conf,source,track_id), ...]}``.
    Prefer feeding these through ``merge_tracker_with_offline_fills`` rather
    than using them as the sole blur source.
    """
    by_frame = _collect_dets(
        cache, conf_floor=conf_floor, moto_only=moto_only,
        min_blur_box_h_frac=min_blur_box_h_frac, frame_height=frame_height,
    )
    tracklets = _build_tracklets(
        by_frame, max_gap=max_gap_frames,
        max_disp_px=max_disp_px, max_disp_frac=max_disp_frac,
        min_vehicle_iou=min_vehicle_iou,
    )
    vehicles_by_frame = {}
    for fi in cache.frame_indices():
        _, vehicles = cache.get(fi)
        vehicles_by_frame[fi] = vehicles

    zones: dict[int, list] = {}
    for tr in tracklets:
        part = _bridge_tracklet(
            tr, max_gap=max_gap_frames,
            max_disp_px=max_disp_px, max_disp_frac=max_disp_frac,
            min_vehicle_iou=min_vehicle_iou,
            vehicles_by_frame=vehicles_by_frame,
        )
        for f, zs in part.items():
            zones.setdefault(f, []).extend(zs)
    return zones


def merge_tracker_with_offline_fills(tracker_plates: list,
                                     offline_zones: list | None) -> list:
    """Hybrid merge: tracker zones win; bridges only fill total blinks.

    If the tracker emitted anything this frame, return it unchanged.
    Otherwise attach offline *bridge* zones (not raw detections — those
    already went through the tracker and were rejected for a reason).
    """
    if tracker_plates:
        return list(tracker_plates)
    if not offline_zones:
        return []
    fills = []
    for z in offline_zones:
        if len(z) > 5 and z[5] == "bridge":
            fills.append(z)
    return fills


def offline_zone_stats(zones: dict[int, list]) -> dict:
    n_frames = len(zones)
    n_raw = n_bridge = 0
    for zs in zones.values():
        for z in zs:
            src = z[5] if len(z) > 5 else ""
            if src == "bridge":
                n_bridge += 1
            else:
                n_raw += 1
    return {
        "frames_with_zones": n_frames,
        "raw_zones": n_raw,
        "bridged_zones": n_bridge,
    }
