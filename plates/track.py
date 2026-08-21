"""Offline plate zones: associate detections to vehicles, then interpolate gaps.

A plate exists only inside a vehicle box. After the full detection cache is
available, detections are chained into tracklets and short plausible gaps are
filled by interpolating the plate pose relative to the vehicle. Cuts and
teleports are not bridged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from plates.common import _overlaps, iou
from plates.constants import VEHICLE_CLASSES
from plates.detect import plate_in_moto_geometry_ok

_MOTO_CLS = 3
_VEHICLE_CLASSES = set(VEHICLE_CLASSES)


@dataclass
class _Det:
    frame: int
    box: tuple
    conf: float
    source: str
    vehicle: tuple
    vcls: int


@dataclass
class _Tracklet:
    dets: list = field(default_factory=list)
    tid: int = 0


def _centre(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _diag(box):
    return math.hypot(box[2] - box[0], box[3] - box[1])


def moto_box_size(box) -> float:
    """Lean-stable size: max of width and height (pixels)."""
    return max(float(box[2] - box[0]), float(box[3] - box[1]))


def moto_size_is_large(size: float, min_size: float, prev_large: bool | None,
                       enter_frac: float = 1.15, exit_frac: float = 0.85) -> bool:
    """Schmitt trigger around *min_size*.

    First observation uses the nominal threshold. After that, stay large until
    size drops below ``exit_frac * min_size``; become large only at or above
    ``enter_frac * min_size``.
    """
    if min_size <= 0:
        return True
    if prev_large is None:
        return size >= min_size
    if prev_large:
        return size >= min_size * exit_frac
    return size >= min_size * enter_frac


class MotoSizeGate:
    """Per-moto size hysteresis, matched across consecutive frames by IoU."""

    def __init__(self, min_size: float, enter_frac: float = 1.15,
                 exit_frac: float = 0.85, min_iou: float = 0.15):
        self.min_size = float(min_size)
        self.enter_frac = float(enter_frac)
        self.exit_frac = float(exit_frac)
        self.min_iou = float(min_iou)
        self._prev: list[tuple[tuple, bool]] = []

    def update(self, vehicles) -> set[tuple]:
        """Advance one frame. Returns boxes of motos currently large enough."""
        motos = []
        for v in vehicles:
            if v[0] != _MOTO_CLS:
                continue
            motos.append((int(v[1]), int(v[2]), int(v[3]), int(v[4])))

        pairs = []
        for i, (pbox, _) in enumerate(self._prev):
            for j, cbox in enumerate(motos):
                ii = iou(pbox, cbox)
                if ii >= self.min_iou:
                    pairs.append((ii, i, j))
        pairs.sort(key=lambda t: t[0], reverse=True)
        mapping: dict[int, int] = {}
        used_p, used_c = set(), set()
        for _, i, j in pairs:
            if i in used_p or j in used_c:
                continue
            used_p.add(i)
            used_c.add(j)
            mapping[j] = i

        new_state = []
        large: set[tuple] = set()
        for j, box in enumerate(motos):
            prev_large = self._prev[mapping[j]][1] if j in mapping else None
            is_large = moto_size_is_large(
                moto_box_size(box), self.min_size, prev_large,
                self.enter_frac, self.exit_frac,
            )
            new_state.append((box, is_large))
            if is_large:
                large.add(box)
        self._prev = new_state
        return large


def _dist(a, b):
    ca, cb = _centre(a), _centre(b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def _lerp_box(a, b, t: float) -> tuple:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))


def hold_vehicles(cache, *, max_gap_frames: int = 15, min_iou: float = 0.15,
                  max_disp_px: float = 80.0, max_disp_frac: float = 0.35) -> dict:
    """Fill short YOLO misses by interpolating the same vehicle's box.

    Returns ``{frame_idx: [vehicle, ...]}`` (detected boxes plus holds).
    Cuts and teleports are not bridged. Does not invent vehicles after the
    last observation.
    """
    frames = cache.frame_indices()
    out: dict[int, list] = {}
    by_cls: dict[int, list] = {}
    for fi in frames:
        rec = cache.get(fi)
        vs = list(rec[1]) if rec is not None else []
        out[fi] = vs
        for v in vs:
            by_cls.setdefault(int(v[0]), []).append((fi, v))

    for cls, obs in by_cls.items():
        obs.sort(key=lambda t: t[0])
        chains: list[list] = []
        for fi, v in obs:
            best = None
            box = v[1:5]
            for ci, ch in enumerate(chains):
                last_fi, last_v = ch[-1]
                gap = fi - last_fi - 1
                if gap < 0 or gap > max_gap_frames:
                    continue
                if not _same_vehicle(last_v[1:5], box, min_iou):
                    continue
                if not motion_gate_ok(
                    last_v[1:5], box, max(gap, 1), vehicle=box,
                    max_disp_px=max_disp_px, max_disp_frac=max_disp_frac,
                ):
                    continue
                cost = _dist(last_v[1:5], box)
                if best is None or cost < best[0]:
                    best = (cost, ci)
            if best is not None:
                chains[best[1]].append((fi, v))
            else:
                chains.append([(fi, v)])
        for ch in chains:
            for i in range(len(ch) - 1):
                (fa, va), (fb, vb) = ch[i], ch[i + 1]
                span = fb - fa
                if span <= 1:
                    continue
                for f in range(fa + 1, fb):
                    if f not in out:
                        continue
                    t = (f - fa) / span
                    box = _lerp_box(va[1:5], vb[1:5], t)
                    conf = float(va[5]) * (1.0 - t) + float(vb[5]) * t
                    if any(x[0] == cls and iou(x[1:5], box) >= min_iou
                           for x in out[f]):
                        continue
                    out[f].append((cls, box[0], box[1], box[2], box[3], conf))
    return out


def _plate_rel(plate, vehicle):
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


def motion_gate_ok(box_a, box_b, gap_frames: int, vehicle=None,
                   max_disp_px: float = 80.0, max_disp_frac: float = 0.35,
                   margin: float = 1.15) -> bool:
    """True if A→B travel is plausible over *gap_frames* missing frames."""
    if gap_frames <= 0:
        return True
    dist = _dist(box_a, box_b)
    base = float(max_disp_px)
    if vehicle is not None:
        base = max(base, _diag(vehicle) * max_disp_frac)
    return dist <= base * gap_frames * margin


def _tightest_vehicle(plate, vehicles, filter_classes):
    """Smallest vehicle box containing the plate centre."""
    pcx = (plate[0] + plate[2]) * 0.5
    pcy = (plate[1] + plate[3]) * 0.5
    best = None
    for v in vehicles:
        cls = v[0]
        if cls not in filter_classes:
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


def _associate_vehicle(plate, vehicles, filter_classes):
    """Vehicle that owns this plate, or None (standalone plates are dropped)."""
    box = plate[:4]
    veh, vcls = _tightest_vehicle(plate, vehicles, filter_classes)
    if veh is not None:
        return veh, vcls
    # Crop remap: centre can sit just outside the YOLO box.
    for v in vehicles:
        cls = v[0]
        if cls not in filter_classes:
            continue
        vv = v[1:5]
        if not _overlaps(box, vv):
            continue
        if cls == _MOTO_CLS and not plate_in_moto_geometry_ok(box, vv):
            continue
        return vv, cls
    return None, None


def _plate_ok(box, veh, vcls) -> bool:
    if vcls == _MOTO_CLS:
        return plate_in_moto_geometry_ok(box, veh)
    return True


def _score(box, conf, veh) -> float:
    """Prefer a compact in-box plate over a large fender/wheel blob."""
    vh = max(1.0, veh[3] - veh[1])
    rh = (box[3] - box[1]) / vh
    compactness = 1.0 / (1.0 + max(0.0, rh - 0.12) * 8.0)
    return float(conf) * compactness


def _collect_dets(cache, conf_floor: float, filter_classes: set,
                  min_moto_h_frac: float, frame_height: int,
                  moto_size_enter_frac: float = 1.15,
                  moto_size_exit_frac: float = 0.85,
                  min_vehicle_iou: float = 0.15):
    """One detection per vehicle per frame; plates outside vehicles are dropped."""
    by_frame: dict[int, list[_Det]] = {}
    gate = MotoSizeGate(
        min_size=min_moto_h_frac * float(frame_height),
        enter_frac=moto_size_enter_frac,
        exit_frac=moto_size_exit_frac,
        min_iou=min_vehicle_iou,
    )
    for fi in cache.frame_indices():
        plates, vehicles = cache.get(fi)
        large_motos = gate.update(vehicles)
        best_by_veh: dict[tuple, _Det] = {}
        for p in plates:
            if len(p) < 5 or p[4] < conf_floor:
                continue
            box = tuple(int(p[i]) for i in range(4))
            conf = float(p[4])
            src = str(p[5]) if len(p) > 5 else "crop"
            if src == "sahi":
                continue
            veh, vcls = _associate_vehicle(p, vehicles, filter_classes)
            if veh is None or not _plate_ok(box, veh, vcls):
                continue
            if vcls == _MOTO_CLS:
                vb = (int(veh[0]), int(veh[1]), int(veh[2]), int(veh[3]))
                if vb not in large_motos:
                    continue
            det = _Det(fi, box, conf, src, veh, vcls)
            key = tuple(veh)
            prev = best_by_veh.get(key)
            if prev is None or _score(box, conf, veh) > _score(prev.box, prev.conf, prev.vehicle):
                best_by_veh[key] = det
        by_frame[fi] = list(best_by_veh.values())
    return by_frame


def _build_tracklets(by_frame: dict, max_gap: int, max_disp_px: float,
                     max_disp_frac: float,
                     min_vehicle_iou: float = 0.15) -> list[_Tracklet]:
    frames = sorted(by_frame)
    tracklets: list[_Tracklet] = []
    next_tid = 1

    for fi in frames:
        used: set[int] = set()
        for det in by_frame[fi]:
            best = None
            for ti, tr in enumerate(tracklets):
                if ti in used:
                    continue
                last = tr.dets[-1]
                gap = fi - last.frame - 1
                if gap < 0 or gap > max_gap:
                    continue
                if last.vcls != det.vcls:
                    continue
                if not _same_vehicle(last.vehicle, det.vehicle, min_vehicle_iou):
                    continue
                if not motion_gate_ok(last.box, det.box, max(gap, 1),
                                      vehicle=det.vehicle,
                                      max_disp_px=max_disp_px,
                                      max_disp_frac=max_disp_frac):
                    continue
                cost = _dist(last.box, det.box)
                if best is None or cost < best[0]:
                    best = (cost, ti, tr)
            if best is not None:
                _, ti, tr = best
                tr.dets.append(det)
                used.add(ti)
            else:
                tracklets.append(_Tracklet(dets=[det], tid=next_tid))
                next_tid += 1
    return tracklets


def _live_vehicle(vehicles, veh_guess, vcls, min_iou: float = 0.15):
    best_v, best_iou = None, 0.0
    for v in vehicles:
        if v[0] != vcls:
            continue
        vv = v[1:5]
        ii = iou(vv, veh_guess)
        if ii > best_iou:
            best_iou, best_v = ii, vv
    if best_v is not None and best_iou >= min_iou:
        return best_v
    return veh_guess


def _bridge_tracklet(tr: _Tracklet, max_gap: int, max_disp_px: float,
                     max_disp_frac: float,
                     min_vehicle_iou: float,
                     vehicles_by_frame: dict) -> dict[int, list]:
    out: dict[int, list] = {}
    dets = sorted(tr.dets, key=lambda d: d.frame)
    for d in dets:
        out.setdefault(d.frame, []).append(
            (d.box[0], d.box[1], d.box[2], d.box[3], d.conf, d.source, tr.tid)
        )
    for i in range(len(dets) - 1):
        a, b = dets[i], dets[i + 1]
        gap = b.frame - a.frame - 1
        if gap <= 0 or gap > max_gap:
            continue
        if not _same_vehicle(a.vehicle, b.vehicle, min_vehicle_iou):
            continue
        if not motion_gate_ok(a.box, b.box, gap, vehicle=a.vehicle or b.vehicle,
                              max_disp_px=max_disp_px,
                              max_disp_frac=max_disp_frac):
            continue
        rel_a = _plate_rel(a.box, a.vehicle)
        rel_b = _plate_rel(b.box, b.vehicle)
        for f in range(a.frame + 1, b.frame):
            t = (f - a.frame) / (b.frame - a.frame)
            veh_f = _live_vehicle(
                vehicles_by_frame.get(f, []),
                _lerp_box(a.vehicle, b.vehicle, t),
                a.vcls,
                min_vehicle_iou,
            )
            box = _plate_from_rel(_lerp_rel(rel_a, rel_b, t), veh_f)
            conf = a.conf * (1 - t) + b.conf * t
            out.setdefault(f, []).append(
                (*box, conf, "bridge", tr.tid)
            )
    return out


def build_zones(
    cache,
    *,
    max_gap_frames: int = 15,
    max_disp_px: float = 80.0,
    max_disp_frac: float = 0.35,
    conf_floor: float = 0.15,
    filter_classes: set | None = None,
    min_moto_h_frac: float = 0.1065,
    frame_height: int = 1080,
    min_vehicle_iou: float = 0.15,
    moto_size_enter_frac: float = 1.15,
    moto_size_exit_frac: float = 0.85,
) -> dict[int, list]:
    """Build per-frame blur zones from a full detection cache.

    Returns ``{frame_idx: [(x1,y1,x2,y2,conf,source,track_id), ...]}``.
    """
    classes = set(filter_classes) if filter_classes is not None else set(_VEHICLE_CLASSES)
    by_frame = _collect_dets(
        cache, conf_floor=conf_floor, filter_classes=classes,
        min_moto_h_frac=min_moto_h_frac, frame_height=frame_height,
        moto_size_enter_frac=moto_size_enter_frac,
        moto_size_exit_frac=moto_size_exit_frac,
        min_vehicle_iou=min_vehicle_iou,
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


def zone_stats(zones: dict[int, list]) -> dict:
    n_obs = n_bridge = 0
    for zs in zones.values():
        for z in zs:
            src = z[5] if len(z) > 5 else ""
            if src == "bridge":
                n_bridge += 1
            else:
                n_obs += 1
    return {
        "frames_with_zones": len(zones),
        "observed_zones": n_obs,
        "bridged_zones": n_bridge,
    }
