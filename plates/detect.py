# ─── Detection: vehicles + in-box plate crops ──────────────────────────────
import cv2
import numpy as np

from plates.common import _overlaps, iou, merge_overlapping
from plates.constants import DETECT_WIDTH, VEHICLE_CLASSES, VEHICLE_FILTER_MAP
from plates.models import _infer_batched


_MOTO_CLS = 3
_CROP_INFER_SIZE = 640
# YOLO often emits two boxes on one motorcycle (rider vs full bike). Same
# class + high overlap / nested box → one vehicle.
_VEHICLE_DEDUP_IOU = 0.4
_VEHICLE_DEDUP_CONTAIN = 0.6


def _xyxy(v):
    return v[1], v[2], v[3], v[4]


def _same_vehicle_detection(a, b, iou_thresh=_VEHICLE_DEDUP_IOU,
                            contain_thresh=_VEHICLE_DEDUP_CONTAIN) -> bool:
    """True when two same-class boxes are the same physical vehicle."""
    if a[0] != b[0]:
        return False
    ba, bb = _xyxy(a), _xyxy(b)
    overlap = iou(ba, bb)
    if overlap >= iou_thresh:
        return True
    ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
    ix2, iy2 = min(ba[2], bb[2]), min(ba[3], bb[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return False
    aa = max(1, (ba[2] - ba[0]) * (ba[3] - ba[1]))
    ab = max(1, (bb[2] - bb[0]) * (bb[3] - bb[1]))
    return inter / min(aa, ab) >= contain_thresh


def _union_vehicle(a, b):
    """One box covering both detections; keep the higher confidence."""
    return (
        a[0],
        min(a[1], b[1]), min(a[2], b[2]),
        max(a[3], b[3]), max(a[4], b[4]),
        max(a[5], b[5]),
    )


def dedupe_vehicles(vehicles, iou_thresh=_VEHICLE_DEDUP_IOU,
                    contain_thresh=_VEHICLE_DEDUP_CONTAIN):
    """Collapse duplicate YOLO boxes on the same vehicle (per class)."""
    if len(vehicles) < 2:
        return list(vehicles)
    ordered = sorted(vehicles, key=lambda v: -v[5])
    kept = []
    for v in ordered:
        merged = False
        for i, k in enumerate(kept):
            if _same_vehicle_detection(v, k, iou_thresh, contain_thresh):
                kept[i] = _union_vehicle(v, k)
                merged = True
                break
        if not merged:
            kept.append(v)
    return kept


def _unsharp_mask(img: np.ndarray, amount: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """
    Sharpen *img* via unsharp masking.

    A Gaussian-blurred copy is subtracted from the original and the
    difference is added back at *amount* × strength.  Works entirely in
    uint8 space through OpenCV's addWeighted so there is no float cast.

    amount : 0.5 = subtle,  1.5 = strong,  3.0 = very aggressive
    sigma  : Gaussian radius in pixels (1.0–2.0 is typical)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    # img*(1+amount) - blurred*amount  ≡ img + amount*(img - blurred)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def enhance_crop_contrast(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel (LAB). Crop-only; does not stretch geometry."""
    if img.size == 0 or img.ndim != 3:
        return img
    h, w = img.shape[:2]
    g = int(grid)
    g = max(1, min(g, max(1, h // 2), max(1, w // 2)))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(g, g))
    l2 = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l2, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def letterbox_to_square(img: np.ndarray, size: int = _CROP_INFER_SIZE,
                        pad_value: int = 114, max_scale: float | None = None):
    """Place *img* on a *size*×*size* canvas, aspect ratio unchanged (pad, never stretch).

    Scale is ``min(size/w, size/h)`` so the image fits. If *max_scale* is set,
    upscale stops there and the rest of the canvas is padding (no extra zoom
    just to fill the square). Images larger than the canvas still downscale to fit.

    Returns (canvas, scale, pad_x, pad_y) mapping original pixels → canvas.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
        return canvas, 1.0, 0.0, 0.0
    fit = min(size / w, size / h)
    scale = fit if max_scale is None else min(fit, float(max_scale))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, img.shape[2] if img.ndim == 3 else 1),
                     pad_value, dtype=img.dtype)
    pad_x = (size - nw) / 2.0
    pad_y = (size - nh) / 2.0
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, scale, pad_x, pad_y


def unletterbox_xyxy(x1, y1, x2, y2, scale, pad_x, pad_y, crop_w, crop_h):
    """Map a box from letterboxed *size*² space back into the pre-letterbox crop."""
    inv = 1.0 / scale if scale > 0 else 1.0
    ox1 = (x1 - pad_x) * inv
    oy1 = (y1 - pad_y) * inv
    ox2 = (x2 - pad_x) * inv
    oy2 = (y2 - pad_y) * inv
    ox1 = int(max(0, min(crop_w, round(ox1))))
    oy1 = int(max(0, min(crop_h, round(oy1))))
    ox2 = int(max(0, min(crop_w, round(ox2))))
    oy2 = int(max(0, min(crop_h, round(oy2))))
    return ox1, oy1, ox2, oy2


def moto_rear_roi(vx1, vy1, vx2, vy2, bottom_frac=0.40, side_pad_frac=0.05,
                  frame_w=None, frame_h=None):
    """Lower-box ROI for motorcycle plate search (frame coordinates)."""
    bw = max(1, vx2 - vx1)
    bh = max(1, vy2 - vy1)
    pad = int(round(bw * side_pad_frac))
    rx1 = vx1 - pad
    rx2 = vx2 + pad
    ry1 = vy1 + int(round(bh * bottom_frac))
    ry2 = vy2
    if frame_w is not None:
        rx1 = max(0, rx1)
        rx2 = min(frame_w, rx2)
    if frame_h is not None:
        ry1 = max(0, ry1)
        ry2 = min(frame_h, ry2)
    return rx1, ry1, rx2, ry2


def plate_in_moto_geometry_ok(plate, box, **_unused) -> bool:
    """True when the plate centre sits inside the vehicle box.

    Where inside the box does not matter (low-mounted plates, lean, close
    crop). Unused kwargs are kept so older callers still bind.
    """
    cx = (plate[0] + plate[2]) * 0.5
    cy = (plate[1] + plate[3]) * 0.5
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _tightest_moto_box(plate, vehicles):
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


def _box_contains_centre(plate, box, expand: float = 0.15) -> bool:
    """Plate centre inside *box* expanded by *expand* on every side."""
    pcx = (plate[0] + plate[2]) * 0.5
    pcy = (plate[1] + plate[3]) * 0.5
    bw, bh = box[2] - box[0], box[3] - box[1]
    padx, pady = bw * expand, bh * expand
    return (box[0] - padx <= pcx <= box[2] + padx
            and box[1] - pady <= pcy <= box[3] + pady)


def filter_moto_plate_geometry(plates, vehicles, collect_rejected=False):
    """Keep plates whose centre is inside a vehicle box. Drop crop remaps that sit on none."""
    kept, rejected = [], []
    for p in plates:
        moto = _tightest_moto_box(p, vehicles)
        if moto is None:
            # Crop-remap artefacts: an upscaled crop detection whose centre
            # sits far from EVERY vehicle box is a remap of the moto crop onto
            # the rider's head or the road beside the bike (194359:2), not a
            # plate. Real car "crop" plates keep passing — their centre is
            # near the car box — so the rejection only bites true artefacts.
            if len(p) > 5 and p[5] in ("crop", "crop_moto"):
                near_any = any(_box_contains_centre(p, v[1:5], 0.15)
                               for v in vehicles)
                if not near_any:
                    if collect_rejected:
                        rejected.append((p[0], p[1], p[2], p[3], p[4], "rejected"))
                    continue
            kept.append(p)
            continue
        # If the plate also overlaps a car/bus/truck, keep it (car crop path).
        overlaps_other = False
        for v in vehicles:
            if v[0] == _MOTO_CLS:
                continue
            if _overlaps(p[:4], v[1:5]):
                overlaps_other = True
                break
        if overlaps_other or plate_in_moto_geometry_ok(p, moto):
            kept.append(p)
        elif collect_rejected:
            rejected.append((p[0], p[1], p[2], p[3], p[4], "rejected"))
    return kept, rejected


def detect_plates(frame, vehicle_model, plate_model, device="cpu", vehicle_conf=0.3,
                  vehicle_conf_floor=None,
                  vehicle_filter="all", plate_conf=0.15, plate_conf_in_vehicle=0.07,
                  sahi_slice_size=640, sahi_overlap=0.2, detect_scale=1.0,
                  sharpen=False, sharpen_amount=1.5, sharpen_sigma=1.0,
                  vehicle_crop_scale=2.0, moto_crop_scale=2.0,
                  moto_crop_bottom_frac=0.28, moto_crop_side_pad_frac=0.05,
                  plate_crop_imgsz=1280,
                  crop_clahe=False, crop_clahe_clip=2.0, crop_clahe_grid=8,
                  collect_rejected=False):
    """
    Returns (plate_rects, all_vehicles, rejected) where:
      plate_rects  — list of (x1, y1, x2, y2, conf) regions to blur
      all_vehicles — list of (cls_id, x1, y1, x2, y2, conf) for every detected vehicle

    Plate search runs only inside detected vehicle boxes. Each crop is placed
    on a *plate_crop_imgsz* square (letterbox, aspect preserved). Upscale is
    capped at vehicle_crop_scale / moto_crop_scale (default 2.0); the rest is
    padding. There is no full-frame sliced pass.

    sahi_slice_size / sahi_overlap / detect_scale:
      Ignored. Kept so callers and cache headers stay compatible.

    Dual-confidence: crop hits use plate_conf_in_vehicle. plate_conf is unused
    for new detections (no standalone full-frame plates).

    sharpen / sharpen_amount / sharpen_sigma:
      Unsharp masking on each vehicle crop, not the full frame.

    vehicle_crop_scale:
      Max upscale of the full vehicle box onto the crop canvas (aspect kept).

    moto_crop_scale / moto_crop_bottom_frac / moto_crop_side_pad_frac:
      Extra motorcycle rear-box ROI. 1.0 = skip the rear ROI (full-box crop
      still runs). Source tag is "crop_moto".

    plate_crop_imgsz:
      Square canvas (and YOLO imgsz) for plate crops. 1280 = larger tela.

    crop_clahe / crop_clahe_clip / crop_clahe_grid:
      CLAHE on the vehicle crop (LAB L channel) before letterbox. Off by default.

    collect_rejected:
      When True, below-threshold detections are collected into `rejected`
      (tagged source 'rejected') instead of being silently dropped.
    """
    h, w = frame.shape[:2]
    scale = DETECT_WIDTH / w
    small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # ── Step 1: vehicle detection — collect all vehicles ──────────────────────
    # When vehicle_conf_floor < vehicle_conf the model runs at the floor and the
    # tracker applies a size-adaptive acceptance (a huge near-certain box is
    # trusted at lower confidence so a close moto re-acquires instantly).
    inv = 1.0 / scale
    all_vehicles = []   # (cls_id, x1, y1, x2, y2, conf)
    v_conf = min(vehicle_conf, vehicle_conf_floor) if vehicle_conf_floor else vehicle_conf
    v_results = vehicle_model(small, conf=v_conf, verbose=False)
    for r in v_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            all_vehicles.append((cls, int(vx1*inv), int(vy1*inv), int(vx2*inv), int(vy2*inv), conf))

    all_vehicles = dedupe_vehicles(all_vehicles)

    filter_classes = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
    del sahi_slice_size, sahi_overlap, detect_scale

    plate_model.confidence_threshold = min(plate_conf, plate_conf_in_vehicle)
    plate_rects = []
    rejected = []

    # ── Step 2: per-vehicle crop pass (the only plate search) ─────────────────
    # Full vehicle box (and moto rear ROI) on a square canvas. Aspect ratio
    # is preserved; upscale is capped so a large box is not zoomed then shrunk.
    _CROP_PAD = 12
    crop_imgsz = max(32, int(plate_crop_imgsz))
    crops = []   # (rgb canvas, cw, ch, inv_scale, cx1, cy1, source, lb_scale, pad_x, pad_y)

    def _queue_crop(cx1, cy1, cx2, cy2, max_scale, source_tag):
        if max_scale < 1.0:
            return
        cx1c = max(0, cx1)
        cy1c = max(0, cy1)
        cx2c = min(w, cx2)
        cy2c = min(h, cy2)
        if cx2c <= cx1c or cy2c <= cy1c:
            return
        crop = frame[cy1c:cy2c, cx1c:cx2c]
        if crop.size == 0:
            return
        if crop_clahe:
            crop = enhance_crop_contrast(crop, crop_clahe_clip, crop_clahe_grid)
        if sharpen:
            crop = _unsharp_mask(crop, sharpen_amount, sharpen_sigma)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = crop_rgb.shape[:2]
        canvas, lb_scale, pad_x, pad_y = letterbox_to_square(
            crop_rgb, size=crop_imgsz, max_scale=max_scale)
        crops.append((canvas, cw, ch, 1.0, cx1c, cy1c,
                      source_tag, lb_scale, pad_x, pad_y))

    full_scale = max(1.0, float(vehicle_crop_scale))
    moto_scale = max(1.0, float(moto_crop_scale))
    for (cls, vx1, vy1, vx2, vy2, _vc) in all_vehicles:
        if cls not in filter_classes:
            continue
        if cls == _MOTO_CLS:
            # Rear ROI at moto_crop_scale (source crop_moto). Also letterbox the
            # full bike box so mid-box plates above the ROI floor are not dropped.
            if moto_crop_scale > 1.0:
                rx1, ry1, rx2, ry2 = moto_rear_roi(
                    vx1, vy1, vx2, vy2,
                    bottom_frac=moto_crop_bottom_frac,
                    side_pad_frac=moto_crop_side_pad_frac,
                    frame_w=w, frame_h=h,
                )
                _queue_crop(rx1, ry1, rx2, ry2, moto_scale, "crop_moto")
            _queue_crop(vx1 - _CROP_PAD, vy1 - _CROP_PAD,
                        vx2 + _CROP_PAD, vy2 + _CROP_PAD,
                        full_scale, "crop")
        else:
            _queue_crop(vx1 - _CROP_PAD, vy1 - _CROP_PAD,
                        vx2 + _CROP_PAD, vy2 + _CROP_PAD,
                        full_scale, "crop")

    if crops:
        min_conf_thr = min(plate_conf, plate_conf_in_vehicle)
        crop_results = _infer_batched(
            plate_model.model, [c[0] for c in crops], conf=min_conf_thr,
            imgsz=crop_imgsz,
        )
        for crop_r, (_img, up_w, up_h, inv_crop, cx1, cy1,
                     source_tag, lb_scale, pad_x, pad_y) in zip(crop_results, crops):
            if crop_r.boxes is None:
                continue
            for box in crop_r.boxes:
                c = float(box.conf[0])
                bx1, by1, bx2, by2 = (float(box.xyxy[0][0]), float(box.xyxy[0][1]),
                                      float(box.xyxy[0][2]), float(box.xyxy[0][3]))
                px1, py1, px2, py2 = unletterbox_xyxy(
                    bx1, by1, bx2, by2, lb_scale, pad_x, pad_y, up_w, up_h)
                ox1 = cx1 + int(px1 * inv_crop)
                oy1 = cy1 + int(py1 * inv_crop)
                ox2 = cx1 + int(px2 * inv_crop)
                oy2 = cy1 + int(py2 * inv_crop)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                if c < plate_conf_in_vehicle:
                    if collect_rejected:
                        rejected.append((ox1, oy1, ox2, oy2, c, "rejected"))
                    continue
                plate_rects.append((ox1, oy1, ox2, oy2, c, source_tag))

    # ── Step 4: moto geometry filter + dedup ──────────────────────────────────
    plate_rects, geo_rej = filter_moto_plate_geometry(
        plate_rects, all_vehicles, collect_rejected=collect_rejected)
    if collect_rejected:
        rejected.extend(geo_rej)
    return merge_overlapping(plate_rects), all_vehicles, rejected

def auto_batch_size(width: int, height: int, sahi_slice_size: int = 640,
                    sahi_overlap: float = 0.2) -> int:
    """How many frames to batch given free GPU memory. Returns 1 on CPU.

    sahi_* arguments are ignored (kept for callers). Estimate assumes a
    handful of 640² vehicle crops per frame.
    """
    _ = (width, height, sahi_slice_size, sahi_overlap)
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        free_bytes, _free_total = torch.cuda.mem_get_info()
        usable = max(0, free_bytes - 3 * 1024 ** 3)
        crops_per_frame = 4
        bytes_per_frame = crops_per_frame * 640 * 640 * 3 * 2 * 4
        batch = max(1, int(usable / max(1, bytes_per_frame)))
        return min(batch, 64)
    except Exception:
        return 1


def detect_plates_batched(frames, vehicle_model, plate_model, device,
                          vehicle_conf=0.3, vehicle_filter="all",
                          plate_conf=0.45, plate_conf_in_vehicle=0.10,
                          sahi_slice_size=640, sahi_overlap=0.2, **kwargs):
    """Run detect_plates() on each frame (plate search is crop-only)."""
    out = []
    for frame in frames:
        r, v, _ = detect_plates(
            frame, vehicle_model, plate_model, device,
            vehicle_conf=vehicle_conf, vehicle_filter=vehicle_filter,
            plate_conf=plate_conf, plate_conf_in_vehicle=plate_conf_in_vehicle,
            sahi_slice_size=sahi_slice_size, sahi_overlap=sahi_overlap,
            **kwargs,
        )
        out.append((r, v))
    return out
