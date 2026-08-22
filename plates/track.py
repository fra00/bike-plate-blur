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
from plates.detect import drop_low_conf_motos, plate_in_moto_geometry_ok

_MOTO_CLS = 3
_VEHICLE_CLASSES = set(VEHICLE_CLASSES)
# YOLO often flips moto ↔ car/truck for a few frames on the same bike.
_CONFUSED_CLS = frozenset({2, 3, 7})
_DEFAULT_AREA_RATIO = 2.5
_DEFAULT_CLASS_FLIP_FRAMES = 10
_DEFAULT_MIN_PLATE_SIDE = 12.0


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


def _box_area(box) -> float:
    return max(1.0, float(box[2] - box[0]) * float(box[3] - box[1]))


def _min_side(box) -> float:
    return min(float(box[2] - box[0]), float(box[3] - box[1]))


def _plate_veh_rel(plate, veh) -> float:
    """Plate area as a fraction of the vehicle box (stable if the bike recedes)."""
    return _box_area(plate) / _box_area(veh)


def _area_ratio_ok(prev, new, max_ratio: float = _DEFAULT_AREA_RATIO) -> bool:
    """False when *new* is a blob much smaller than the tracked plate."""
    if max_ratio <= 0:
        return True
    return _box_area(prev) / _box_area(new) <= max_ratio


def _vcls_link_ok(ca: int, cb: int, gap: int, max_class_flip_frames: int) -> bool:
    if ca == cb:
        return True
    return (ca in _CONFUSED_CLS and cb in _CONFUSED_CLS
            and 0 <= gap <= max_class_flip_frames)


def _identity_ok(va, vb, gap: int, min_iou: float, max_class_flip_frames: int) -> bool:
    """Same vehicle, or a short class-flip / box jump the motion gate will check."""
    if _same_vehicle(va, vb, min_iou):
        return True
    return 0 <= gap <= max_class_flip_frames


def hold_vehicles(cache, *, max_gap_frames: int = 15, min_iou: float = 0.15,
                  max_disp_px: float = 80.0, max_disp_frac: float = 0.35,
                  max_class_flip_frames: int = _DEFAULT_CLASS_FLIP_FRAMES,
                  moto_min_conf: float = 0.30) -> dict:
    """Fill short YOLO misses by interpolating the same vehicle's box.

    Returns ``{frame_idx: [vehicle, ...]}`` (detected boxes plus holds).
    Cuts and teleports are not bridged. Does not invent vehicles after the
    last observation. Moto/car/truck may share a chain across a short class
    flip; the held box is emitted as motorcycle if either endpoint is one.
    Motorcycle boxes below *moto_min_conf* are dropped (including class-flip
    relabels of a low-conf car/truck).
    """
    frames = cache.frame_indices()
    out: dict[int, list] = {}
    by_group: dict[object, list] = {}
    for fi in frames:
        rec = cache.get(fi)
        vs = list(rec[1]) if rec is not None else []
        out[fi] = vs
        for v in vs:
            cls = int(v[0])
            key = _CONFUSED_CLS if cls in _CONFUSED_CLS else cls
            by_group.setdefault(key, []).append((fi, v))

    for obs in by_group.values():
        obs.sort(key=lambda t: t[0])
        chains: list[list] = []
        for fi, v in obs:
            best = None
            box = v[1:5]
            cb = int(v[0])
            for ci, ch in enumerate(chains):
                last_fi, last_v = ch[-1]
                gap = fi - last_fi - 1
                if gap < 0 or gap > max_gap_frames:
                    continue
                ca = int(last_v[0])
                if not _vcls_link_ok(ca, cb, gap, max_class_flip_frames):
                    continue
                if not _identity_ok(last_v[1:5], box, gap, min_iou,
                                    max_class_flip_frames):
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
            chain_has_moto = any(int(v[0]) == _MOTO_CLS for _, v in ch)
            if chain_has_moto:
                for fi, v in ch:
                    if int(v[0]) == _MOTO_CLS:
                        continue
                    box = v[1:5]
                    if any(x[0] == _MOTO_CLS and iou(x[1:5], box) >= min_iou
                           for x in out[fi]):
                        continue
                    conf = float(v[5])
                    if conf < moto_min_conf:
                        continue
                    out[fi].append(
                        (_MOTO_CLS, box[0], box[1], box[2], box[3], conf))
            for i in range(len(ch) - 1):
                (fa, va), (fb, vb) = ch[i], ch[i + 1]
                span = fb - fa
                if span <= 1:
                    continue
                hold_cls = _MOTO_CLS if chain_has_moto else int(va[0])
                for f in range(fa + 1, fb):
                    if f not in out:
                        continue
                    t = (f - fa) / span
                    box = _lerp_box(va[1:5], vb[1:5], t)
                    conf = float(va[5]) * (1.0 - t) + float(vb[5]) * t
                    if hold_cls == _MOTO_CLS and conf < moto_min_conf:
                        continue
                    if any(x[0] == hold_cls and iou(x[1:5], box) >= min_iou
                           for x in out[f]):
                        continue
                    out[f].append(
                        (hold_cls, box[0], box[1], box[2], box[3], conf))
    _fork_split_holds(
        cache, out,
        max_gap_frames=max_gap_frames,
        min_iou=min_iou,
        max_disp_px=max_disp_px,
        max_disp_frac=max_disp_frac,
    )
    _nms_held_motos(out)
    if moto_min_conf > 0:
        for fi in out:
            out[fi] = drop_low_conf_motos(out[fi], moto_min_conf)
    return out


def _centre_inside(inner, outer) -> bool:
    cx, cy = _centre(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _area(box) -> float:
    return max(0.0, float(box[2] - box[0]) * float(box[3] - box[1]))


def _emit_held(out, f, cls, box, conf, min_skip_iou: float) -> bool:
    if f not in out:
        return False
    if any(int(x[0]) == cls and iou(x[1:5], box) >= min_skip_iou for x in out[f]):
        return False
    out[f].append((cls, box[0], box[1], box[2], box[3], conf))
    return True


def _remainder_box(wide, kept):
    """Side of *wide* that *kept* does not cover (second bike after a merge)."""
    wx1, wy1, wx2, wy2 = (float(wide[0]), float(wide[1]),
                          float(wide[2]), float(wide[3]))
    kx1, ky1, kx2, ky2 = (float(kept[0]), float(kept[1]),
                          float(kept[2]), float(kept[3]))
    left_lost = max(0.0, kx1 - wx1)
    right_lost = max(0.0, wx2 - kx2)
    if right_lost >= left_lost and right_lost >= 16:
        return (int(round(kx2)), int(round(wy1)), int(round(wx2)), int(round(wy2)))
    if left_lost >= 16:
        return (int(round(wx1)), int(round(wy1)), int(round(kx1)), int(round(wy2)))
    return None


def _fork_split_holds(cache, out, *, max_gap_frames: int, min_iou: float,
                      max_disp_px: float, max_disp_frac: float) -> None:
    """Restore a moto YOLO dropped when two bikes shared one box, then split.

    If a wide box at *fa* later continues as bike B, and a new bike C appears
    whose centre sat inside that wide box, interpolate the lost remainder → C.
    """
    frames = cache.frame_indices()
    raw: dict[int, list] = {}
    for fi in frames:
        rec = cache.get(fi)
        vs = rec[1] if rec is not None else []
        raw[fi] = [v for v in vs if int(v[0]) in _CONFUSED_CLS]

    for fc in frames:
        for c in raw.get(fc, []):
            if int(c[0]) != _MOTO_CLS:
                continue
            cb = c[1:5]
            best = None
            for fa in range(fc - 1, fc - max_gap_frames - 2, -1):
                if fa not in raw:
                    continue
                gap = fc - fa - 1
                if gap < 1:
                    continue
                for a in raw[fa]:
                    ab = a[1:5]
                    if not _centre_inside(cb, ab):
                        continue
                    if _area(ab) < 1.25 * _area(cb):
                        continue
                    if not motion_gate_ok(
                        ab, cb, gap, vehicle=cb,
                        max_disp_px=max_disp_px, max_disp_frac=max_disp_frac,
                    ):
                        continue
                    sibling_box = None
                    for fb in range(fa + 1, fc + 1):
                        for b in raw.get(fb, []):
                            bb = b[1:5]
                            if iou(bb, cb) >= 0.45:
                                continue
                            if _centre_inside(bb, ab) or iou(bb, ab) >= min_iou:
                                sibling_box = bb
                                break
                        if sibling_box is not None:
                            break
                    if sibling_box is None:
                        continue
                    start = _remainder_box(ab, sibling_box)
                    # No leftover strip → sibling is the same bike, not a split.
                    # Interpolating the full wide box toward a later moto C
                    # paints ghosts on empty road before C exists.
                    if start is None:
                        continue
                    if _area(start) < 16 * 16:
                        continue
                    if not motion_gate_ok(
                        start, cb, gap, vehicle=cb,
                        max_disp_px=max_disp_px, max_disp_frac=max_disp_frac,
                    ):
                        continue
                    if best is None or gap < best[0]:
                        best = (gap, fa, a, start)
            if best is None:
                continue
            _, fa, a, start = best
            span = fc - fa
            for f in range(fa + 1, fc):
                t = (f - fa) / span
                box = _lerp_box(start, cb, t)
                conf = float(a[5]) * (1.0 - t) + float(c[5]) * t
                _emit_held(out, f, _MOTO_CLS, box, conf, min_skip_iou=0.45)


def _nms_held_motos(out, min_iou: float = 0.28) -> None:
    """Drop duplicate moto boxes (split-hold overlap). Keep the larger AABB."""
    for fi, vs in out.items():
        motos = [v for v in vs if int(v[0]) == _MOTO_CLS]
        others = [v for v in vs if int(v[0]) != _MOTO_CLS]
        motos.sort(key=lambda v: _area(v[1:5]), reverse=True)
        kept = []
        for v in motos:
            box = v[1:5]
            if any(
                iou(box, u[1:5]) >= min_iou or _centre_inside(box, u[1:5])
                for u in kept
            ):
                continue
            kept.append(v)
        out[fi] = others + kept


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


def _prefer_larger(new: _Det, old: _Det) -> bool:
    """True if *new* should replace *old* on the same vehicle (larger area)."""
    an, ao = _box_area(new.box), _box_area(old.box)
    if an != ao:
        return an > ao
    return new.conf > old.conf


def _prefer_plate(new: _Det, old: _Det, last_plate) -> bool:
    """Prefer the hit closer to this vehicle's last plate; else the larger box.

    Nested / overlapping hits (same plate, one box inside the other) always
    keep the larger box — nearest-to-last would otherwise lock onto a corner
    blob once the track has already shrunk.
    """
    if iou(new.box, old.box) >= 0.3:
        return _prefer_larger(new, old)
    if last_plate is not None:
        dn, do = _dist(new.box, last_plate), _dist(old.box, last_plate)
        if dn != do:
            return dn < do
    return _prefer_larger(new, old)


def _assign_last_plates(cands: dict, last: list, fi: int,
                        min_iou: float, mem_frames: int):
    """Map current vehicle keys to (pos, peak_rel, last-record index)."""
    pairs = []
    for key, dets in cands.items():
        veh = dets[0].vehicle
        for i, (lv, _pos, _peak_rel, lf) in enumerate(last):
            if fi - lf > mem_frames:
                continue
            ii = iou(veh, lv)
            if ii >= min_iou:
                pairs.append((ii, key, i))
    pairs.sort(key=lambda x: x[0], reverse=True)
    assigned: dict[tuple, tuple] = {}
    used_last: set[int] = set()
    for _ii, key, i in pairs:
        if key in assigned or i in used_last:
            continue
        assigned[key] = (last[i][1], last[i][2], i)
        used_last.add(i)
    return assigned


def _collect_dets(cache, conf_floor: float, filter_classes: set,
                  min_moto_h_frac: float, frame_height: int,
                  moto_size_enter_frac: float = 1.15,
                  moto_size_exit_frac: float = 0.85,
                  min_vehicle_iou: float = 0.15,
                  min_plate_side_px: float = _DEFAULT_MIN_PLATE_SIDE,
                  plate_mem_frames: int = 30,
                  max_area_ratio: float = _DEFAULT_AREA_RATIO):
    """One detection per vehicle per frame; plates outside vehicles are dropped.

    When a vehicle has several plate hits, keep the one closest to that
    vehicle's last chosen plate (top-case / fog-light blobs are usually
    farther). With no history, keep the larger box. Hits much smaller
    than this vehicle's peak plate/vehicle area ratio are skipped (the
    last plate is kept so interpolation can run). The ratio does not
    ratchet down, so a plate that shrinks a little each frame is still
    dropped once it is a blob relative to the bike. A bike that recedes
    keeps its detections because both boxes shrink together.
    """
    by_frame: dict[int, list[_Det]] = {}
    gate = MotoSizeGate(
        min_size=min_moto_h_frac * float(frame_height),
        enter_frac=moto_size_enter_frac,
        exit_frac=moto_size_exit_frac,
        min_iou=min_vehicle_iou,
    )
    last: list[tuple[tuple, tuple, float, int]] = []  # veh, pos, peak_rel, frame
    mem = max(1, int(plate_mem_frames))
    for fi in cache.frame_indices():
        plates, vehicles = cache.get(fi)
        large_motos = gate.update(vehicles)
        cands: dict[tuple, list[_Det]] = {}
        for p in plates:
            if len(p) < 5 or p[4] < conf_floor:
                continue
            box = tuple(int(p[i]) for i in range(4))
            if min_plate_side_px > 0 and _min_side(box) < min_plate_side_px:
                continue
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
            cands.setdefault(tuple(veh), []).append(det)

        last_for = _assign_last_plates(
            cands, last, fi, min_vehicle_iou, mem)

        chosen: list[_Det] = []
        consumed: set[int] = set()
        new_last: list[tuple] = []
        for key, dets in cands.items():
            rec = last_for.get(key)
            lp = rec[0] if rec is not None else None
            peak_rel = rec[1] if rec is not None else None
            if peak_rel is not None and peak_rel > 0:
                dets = [d for d in dets
                        if _plate_veh_rel(d.box, d.vehicle) * max_area_ratio
                        >= peak_rel]
                if not dets:
                    continue
            best = dets[0]
            for d in dets[1:]:
                if _prefer_plate(d, best, lp):
                    best = d
            chosen.append(best)
            new_rel = _plate_veh_rel(best.box, best.vehicle)
            if rec is not None:
                consumed.add(rec[2])
                new_rel = max(float(rec[1]), new_rel)
            new_last.append((best.vehicle, best.box, new_rel, fi))
        by_frame[fi] = chosen

        for i, rec in enumerate(last):
            if i not in consumed and fi - rec[3] <= mem:
                new_last.append(rec)
        last = new_last
    return by_frame


def _tracklet_match(last: _Det, det: _Det, gap: int, max_gap: int,
                    max_disp_px: float, max_disp_frac: float,
                    min_vehicle_iou: float, max_class_flip_frames: int,
                    max_area_ratio: float):
    """Return None if *det* may join *last*, else a reject reason."""
    if gap < 0 or gap > max_gap:
        return "gap"
    if not _vcls_link_ok(last.vcls, det.vcls, gap, max_class_flip_frames):
        return "cls"
    if not _identity_ok(last.vehicle, det.vehicle, gap, min_vehicle_iou,
                        max_class_flip_frames):
        return "veh"
    if not motion_gate_ok(last.box, det.box, max(gap, 1),
                          vehicle=det.vehicle,
                          max_disp_px=max_disp_px,
                          max_disp_frac=max_disp_frac):
        return "motion"
    if not _area_ratio_ok(last.box, det.box, max_area_ratio):
        return "area"
    return None


def _build_tracklets(by_frame: dict, max_gap: int, max_disp_px: float,
                     max_disp_frac: float,
                     min_vehicle_iou: float = 0.15,
                     max_class_flip_frames: int = _DEFAULT_CLASS_FLIP_FRAMES,
                     max_area_ratio: float = _DEFAULT_AREA_RATIO) -> list[_Tracklet]:
    frames = sorted(by_frame)
    tracklets: list[_Tracklet] = []
    next_tid = 1

    for fi in frames:
        used: set[int] = set()
        for det in by_frame[fi]:
            best = None
            area_only = False
            for ti, tr in enumerate(tracklets):
                if ti in used:
                    continue
                last = tr.dets[-1]
                gap = fi - last.frame - 1
                reason = _tracklet_match(
                    last, det, gap, max_gap, max_disp_px, max_disp_frac,
                    min_vehicle_iou, max_class_flip_frames, max_area_ratio,
                )
                if reason is None:
                    cost = _dist(last.box, det.box)
                    if best is None or cost < best[0]:
                        best = (cost, ti, tr)
                elif reason == "area":
                    area_only = True
            if best is not None:
                _, ti, tr = best
                tr.dets.append(det)
                used.add(ti)
            elif area_only:
                continue
            else:
                tracklets.append(_Tracklet(dets=[det], tid=next_tid))
                next_tid += 1
    return tracklets


def _live_vehicle(vehicles, veh_guess, vcls, min_iou: float = 0.15):
    allow = _CONFUSED_CLS if vcls in _CONFUSED_CLS else {vcls}
    best_v, best_iou = None, 0.0
    for v in vehicles:
        if v[0] not in allow:
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
                     vehicles_by_frame: dict,
                     max_class_flip_frames: int = _DEFAULT_CLASS_FLIP_FRAMES,
                     max_area_ratio: float = _DEFAULT_AREA_RATIO) -> dict[int, list]:
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
        if _tracklet_match(
            a, b, gap, max_gap, max_disp_px, max_disp_frac,
            min_vehicle_iou, max_class_flip_frames, max_area_ratio,
        ) is not None:
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
    min_moto_h_frac: float = 0.08333,
    frame_height: int = 1080,
    min_vehicle_iou: float = 0.15,
    moto_size_enter_frac: float = 1.15,
    moto_size_exit_frac: float = 0.85,
    min_plate_side_px: float = _DEFAULT_MIN_PLATE_SIDE,
    max_class_flip_frames: int = _DEFAULT_CLASS_FLIP_FRAMES,
    max_area_ratio: float = _DEFAULT_AREA_RATIO,
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
        min_plate_side_px=min_plate_side_px,
        plate_mem_frames=max_gap_frames,
        max_area_ratio=max_area_ratio,
    )
    tracklets = _build_tracklets(
        by_frame, max_gap=max_gap_frames,
        max_disp_px=max_disp_px, max_disp_frac=max_disp_frac,
        min_vehicle_iou=min_vehicle_iou,
        max_class_flip_frames=max_class_flip_frames,
        max_area_ratio=max_area_ratio,
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
            max_class_flip_frames=max_class_flip_frames,
            max_area_ratio=max_area_ratio,
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
