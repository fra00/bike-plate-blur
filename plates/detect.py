# ─── Detection: vehicles + SAHI-sliced plates ──────────────────────────────
import math

import cv2
import numpy as np

from plates.common import _overlaps, merge_overlapping
from plates.constants import DETECT_WIDTH, VEHICLE_CLASSES, VEHICLE_FILTER_MAP
from plates.models import _infer_batched


_MOTO_CLS = 3
_CROP_INFER_SIZE = 640


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


def letterbox_to_square(img: np.ndarray, size: int = _CROP_INFER_SIZE,
                        pad_value: int = 114):
    """Resize *img* into a *size*×*size* canvas preserving aspect ratio.

    Returns (canvas, scale, pad_x, pad_y) where *scale* maps original crop
    pixels → letterboxed pixels, and (*pad_x*, *pad_y*) are the left/top
    padding inserted before the resized content.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
        return canvas, 1.0, 0.0, 0.0
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
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


def plate_in_moto_geometry_ok(plate, box, ry_lo=0.25, ry_hi=0.78,
                              min_rw=0.08, min_rh=0.05) -> bool:
    """Reject taillight / wheel hits that sit outside a plausible rear-plate band."""
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    cx = (plate[0] + plate[2]) * 0.5
    cy = (plate[1] + plate[3]) * 0.5
    rx = (cx - box[0]) / bw
    ry = (cy - box[1]) / bh
    rw = (plate[2] - plate[0]) / bw
    rh = (plate[3] - plate[1]) / bh
    if not (0.05 <= rx <= 0.95):
        return False
    if not (ry_lo <= ry <= ry_hi):
        return False
    if rw < min_rw or rh < min_rh:
        return False
    if rw > 0.70 or rh > 0.55:
        return False
    return True


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


def filter_moto_plate_geometry(plates, vehicles, collect_rejected=False):
    """Drop (or mark rejected) plates that only sit on a moto but fail geometry."""
    kept, rejected = [], []
    for p in plates:
        moto = _tightest_moto_box(p, vehicles)
        if moto is None:
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
                  vehicle_crop_scale=1.0, moto_crop_scale=3.0,
                  moto_crop_bottom_frac=0.28, moto_crop_side_pad_frac=0.05,
                  collect_rejected=False):
    """
    Returns (plate_rects, all_vehicles, rejected) where:
      plate_rects  — list of (x1, y1, x2, y2, conf) regions to blur
      all_vehicles — list of (cls_id, x1, y1, x2, y2, conf) for every detected vehicle

    Dual-confidence strategy:
      - Plates inside a vehicle bounding box use plate_conf_in_vehicle (lower).
      - Plates outside any vehicle box use plate_conf (stricter).
      This lets you catch blurry / angled plates on motorbikes without flooding
      the full frame with false positives.

    detect_scale:
      Fraction of the original resolution sent to the plate model (default 1.0).
      0.5 processes 4K footage at 2K for ~5× faster detection on a compute-bound
      GPU; blur is always applied at the original full resolution.

    sharpen / sharpen_amount / sharpen_sigma:
      Apply unsharp masking to the detection frame before SAHI tiling.
      Helps with lens blur, mild motion blur, or heavily compressed footage.

    vehicle_crop_scale:
      When > 1.0, each car/bus/truck bounding box is extracted from the
      original full-resolution frame, upscaled by this factor with Lanczos
      resampling, letterboxed to 640² (aspect preserved), and fed to the
      plate model.  1.0 = disabled.  2.0 is recommended when enabling.

    moto_crop_scale / moto_crop_bottom_frac / moto_crop_side_pad_frac:
      Motorcycle crop pass uses a rear-box ROI (y from bottom_frac→1.0) and
      its own upscale factor (default 3.0). Source tag is "crop_moto".

    collect_rejected:
      When True, below-threshold detections are collected into `rejected`
      (tagged source 'rejected') instead of being silently dropped, so the
      debug overlay can draw them.  Adds no inference cost — the model already
      runs at the lowest threshold; this only changes a `continue` into an
      append.  Defaults False so production output is byte-for-byte unchanged.
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

    # Boxes for the active filter (used for plate overlap check)
    filter_classes = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
    filter_boxes = [(x1, y1, x2, y2) for (cls, x1, y1, x2, y2, _) in all_vehicles
                    if cls in filter_classes]

    # ── Step 2: SAHI sliced plate detection ───────────────────────────────────
    # Optionally downsample the frame for faster detection (detect_scale < 1.0).
    # Coordinates are scaled back to full resolution after detection so the blur
    # is always applied at the original quality.
    if detect_scale < 1.0:
        det_w = max(1, int(w * detect_scale))
        det_h = max(1, int(h * detect_scale))
        frame_for_det = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        coord_scale   = 1.0 / detect_scale   # multiply detected coords by this
        # Vehicle filter boxes also need to be in detection-space
        filter_boxes_det = [(int(x1 * detect_scale), int(y1 * detect_scale),
                             int(x2 * detect_scale), int(y2 * detect_scale))
                            for (x1, y1, x2, y2) in filter_boxes]
    else:
        frame_for_det    = frame
        coord_scale      = 1.0
        filter_boxes_det = filter_boxes

    # Optional: sharpen the detection frame to recover edge contrast on blurry footage
    if sharpen:
        frame_for_det = _unsharp_mask(frame_for_det, sharpen_amount, sharpen_sigma)

    # Model threshold is pre-set to min(plate_conf, plate_conf_in_vehicle) in
    # load_models() so no valid detection is thrown away before we can filter.
    plate_model.confidence_threshold = min(plate_conf, plate_conf_in_vehicle)
    frame_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)

    # Sliced detection without the SAHI loop overhead: tile the frame into
    # sahi_slice_size squares (with sahi_overlap), run ALL tiles through the
    # TensorRT engine in one fixed-batch call, then map detections back.
    stride   = int(sahi_slice_size * (1 - sahi_overlap))
    fh, fw   = frame_rgb.shape[:2]
    tiles    = []   # (tile RGB ndarray 640², origin_x, origin_y, actual_w, actual_h)
    y = 0
    while True:
        y2 = min(y + sahi_slice_size, fh)
        x = 0
        while True:
            x2 = min(x + sahi_slice_size, fw)
            tw, th = x2 - x, y2 - y
            tile = frame_rgb[y:y2, x:x2]
            if tw < sahi_slice_size or th < sahi_slice_size:
                pad = np.zeros((sahi_slice_size, sahi_slice_size, 3), dtype=np.uint8)
                pad[:th, :tw] = tile
                tile = pad
            tiles.append((tile, x, y, tw, th))
            if x2 >= fw:
                break
            x += stride
        if y2 >= fh:
            break
        y += stride

    tile_imgs = [t[0] for t in tiles]
    tile_results = _infer_batched(
        plate_model.model, tile_imgs,
        conf=min(plate_conf, plate_conf_in_vehicle), imgsz=sahi_slice_size,
    )

    # ── Step 3: context-aware confidence filtering ────────────────────────────
    plate_rects = []
    rejected    = []   # below-threshold detections (only populated if collect_rejected)
    for tile_r, (tile, ox, oy, tw, th) in zip(tile_results, tiles):
        if tile_r.boxes is None:
            continue
        for box in tile_r.boxes:
            x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
            x2, y2 = int(box.xyxy[0][2]), int(box.xyxy[0][3])
            # clamp to the unpadded tile area
            x1, x2 = min(x1, tw), min(x2, tw)
            y1, y2 = min(y1, th), min(y2, th)
            if x2 <= x1 or y2 <= y1:
                continue
            conf       = float(box.conf[0])
            plate      = (ox + x1, oy + y1, ox + x2, oy + y2)
            in_vehicle = any(_overlaps(plate, vb) for vb in filter_boxes_det)

            if vehicle_filter != "all" and not in_vehicle:
                continue  # vehicle filter requires vehicle overlap

            required = plate_conf_in_vehicle if in_vehicle else plate_conf
            if conf < required:
                if collect_rejected:
                    rejected.append((
                        int(plate[0] * coord_scale), int(plate[1] * coord_scale),
                        int(plate[2] * coord_scale), int(plate[3] * coord_scale),
                        conf, "rejected",
                    ))
                continue

            # Scale coords back to full-resolution space.
            # Tuple format: (x1, y1, x2, y2, conf, source) where source is one of
            # 'sahi' | 'crop' | 'pred' | 'own'.  Older code reading rect[:5] is
            # unaffected by the extra trailing field.
            plate_rects.append((
                int(plate[0] * coord_scale), int(plate[1] * coord_scale),
                int(plate[2] * coord_scale), int(plate[3] * coord_scale),
                conf, "sahi",
            ))

    # ── Step 3b: per-vehicle crop upscale pass ────────────────────────────────
    # Cars/trucks: full box + vehicle_crop_scale. Motorcycles: rear ROI +
    # moto_crop_scale. Both letterbox to 640² (aspect preserved) before a
    # batched engine call, then remap boxes back to frame coordinates.
    _CROP_PAD = 12
    crops = []   # (rgb 640², up_w, up_h, inv_scale, cx1, cy1, source, lb_scale, pad_x, pad_y)

    def _queue_crop(cx1, cy1, cx2, cy2, scale_factor, source_tag):
        if scale_factor <= 1.0:
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
        up_w = max(1, int((cx2c - cx1c) * scale_factor))
        up_h = max(1, int((cy2c - cy1c) * scale_factor))
        crop_up = cv2.resize(crop, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        if sharpen:
            crop_up = _unsharp_mask(crop_up, sharpen_amount, sharpen_sigma)
        crop_rgb = cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)
        canvas, lb_scale, pad_x, pad_y = letterbox_to_square(
            crop_rgb, size=_CROP_INFER_SIZE)
        crops.append((canvas, up_w, up_h, 1.0 / scale_factor, cx1c, cy1c,
                      source_tag, lb_scale, pad_x, pad_y))

    for (cls, vx1, vy1, vx2, vy2, _vc) in all_vehicles:
        if cls not in filter_classes:
            continue
        if cls == _MOTO_CLS:
            # Rear ROI at moto_crop_scale (source crop_moto). Also letterbox the
            # full bike box at vehicle_crop_scale so mid-box plates that sit
            # above the ROI floor are not dropped (was the main v2 regression).
            if moto_crop_scale > 1.0:
                rx1, ry1, rx2, ry2 = moto_rear_roi(
                    vx1, vy1, vx2, vy2,
                    bottom_frac=moto_crop_bottom_frac,
                    side_pad_frac=moto_crop_side_pad_frac,
                    frame_w=w, frame_h=h,
                )
                _queue_crop(rx1, ry1, rx2, ry2, moto_crop_scale, "crop_moto")
            if vehicle_crop_scale > 1.0:
                _queue_crop(vx1 - _CROP_PAD, vy1 - _CROP_PAD,
                            vx2 + _CROP_PAD, vy2 + _CROP_PAD,
                            vehicle_crop_scale, "crop")
        elif vehicle_crop_scale > 1.0:
            _queue_crop(vx1 - _CROP_PAD, vy1 - _CROP_PAD,
                        vx2 + _CROP_PAD, vy2 + _CROP_PAD,
                        vehicle_crop_scale, "crop")

    if crops:
        min_conf_thr = min(plate_conf, plate_conf_in_vehicle)
        crop_results = _infer_batched(
            plate_model.model, [c[0] for c in crops], conf=min_conf_thr,
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
    """
    Calculate how many frames to batch based on free GPU memory after models
    have been loaded.  Returns 1 when CUDA is not available (CPU mode).

    Memory estimate per frame:
        tiles_per_frame × tile_bytes × 4  (activation headroom, float16)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        free_bytes, _ = torch.cuda.mem_get_info()
        usable = max(0, free_bytes - 3 * 1024 ** 3)   # keep 3 GB headroom

        stride = int(sahi_slice_size * (1 - sahi_overlap))
        tiles_per_frame = (
            math.ceil(width  / stride) *
            math.ceil(height / stride)
        )
        # float16 tile tensor × 4 for intermediate activations
        bytes_per_frame = tiles_per_frame * sahi_slice_size * sahi_slice_size * 3 * 2 * 4

        batch = max(1, int(usable / bytes_per_frame))
        return min(batch, 64)          # practical cap: 64 frames at once
    except Exception:
        return 1

def detect_plates_batched(frames, vehicle_model, plate_model, device,
                          vehicle_conf=0.3, vehicle_filter="all",
                          plate_conf=0.45, plate_conf_in_vehicle=0.10,
                          sahi_slice_size=640, sahi_overlap=0.2):
    """
    GPU-efficient alternative to calling detect_plates() per frame.

    Strategy:
      1. One batched vehicle-detection call for all N frames.
      2. All SAHI tiles from all N frames are pooled and sent to the plate
         model in a single GPU inference call — far fewer kernel launches.
      3. Tile coordinates are mapped back to full-frame space per-frame.
      4. The same context-aware confidence filtering and NMS used in
         detect_plates() is applied, so results are equivalent.

    Falls back to single-frame detect_plates() when len(frames) == 1.
    """
    if len(frames) == 1:
        r, v, _ = detect_plates(
            frames[0], vehicle_model, plate_model, device,
            vehicle_conf=vehicle_conf, vehicle_filter=vehicle_filter,
            plate_conf=plate_conf, plate_conf_in_vehicle=plate_conf_in_vehicle,
            sahi_slice_size=sahi_slice_size, sahi_overlap=sahi_overlap,
        )
        return [(r, v)]

    h, w = frames[0].shape[:2]
    scale  = DETECT_WIDTH / w
    inv    = 1.0 / scale
    filter_classes = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
    min_conf = min(plate_conf, plate_conf_in_vehicle)

    # ── 1. Batch vehicle detection (one GPU call for all N frames) ─────────────
    smalls = [cv2.resize(f, (DETECT_WIDTH, int(h * scale)),
                         interpolation=cv2.INTER_LINEAR) for f in frames]
    batch_v = vehicle_model(smalls, conf=vehicle_conf, verbose=False)

    per_frame_vehicles = []
    for r in batch_v:
        vehicles = []
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in VEHICLE_CLASSES:
                    continue
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                conf_v = float(box.conf[0])
                vehicles.append((cls, int(vx1 * inv), int(vy1 * inv),
                                  int(vx2 * inv), int(vy2 * inv), conf_v))
        per_frame_vehicles.append(vehicles)

    # ── 2. Pool SAHI tiles from all frames into one list ──────────────────────
    stride     = int(sahi_slice_size * (1 - sahi_overlap))
    all_tiles  = []   # flat list of tile ndarrays (RGB, padded to sahi_slice_size²)
    tile_meta  = []   # (frame_idx, origin_x, origin_y, actual_w, actual_h)

    for fi, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        y = 0
        while True:
            y2 = min(y + sahi_slice_size, fh)
            x = 0
            while True:
                x2 = min(x + sahi_slice_size, fw)
                actual_w, actual_h = x2 - x, y2 - y
                tile = rgb[y:y2, x:x2]
                if actual_h < sahi_slice_size or actual_w < sahi_slice_size:
                    pad = np.zeros((sahi_slice_size, sahi_slice_size, 3), dtype=np.uint8)
                    pad[:actual_h, :actual_w] = tile
                    tile = pad
                all_tiles.append(tile)
                tile_meta.append((fi, x, y, actual_w, actual_h))
                if x2 >= fw:
                    break
                x += stride
            if y2 >= fh:
                break
            y += stride

    # ── 3. Single batched plate inference on all tiles ────────────────────────
    tile_results = plate_model.model(
        all_tiles, conf=min_conf, verbose=False, imgsz=sahi_slice_size,
    )

    # ── 4. Map tile detections back to full-frame coordinates ─────────────────
    per_frame_raw = [[] for _ in frames]
    for tile_r, (fi, ox, oy, tw, th) in zip(tile_results, tile_meta):
        if tile_r.boxes is None:
            continue
        for box in tile_r.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            # clamp to the unpadded tile area
            bx1, bx2 = min(bx1, tw), min(bx2, tw)
            by1, by2 = min(by1, th), min(by2, th)
            if bx2 <= bx1 or by2 <= by1:
                continue
            per_frame_raw[fi].append(
                (ox + bx1, oy + by1, ox + bx2, oy + by2, float(box.conf[0]))
            )

    # ── 5. Context-aware filtering + NMS, per frame ───────────────────────────
    results = []
    for fi, all_vehicles in enumerate(per_frame_vehicles):
        filter_boxes = [
            (x1, y1, x2, y2)
            for (cls, x1, y1, x2, y2, _) in all_vehicles
            if cls in filter_classes
        ]
        plate_rects = []
        for (x1, y1, x2, y2, conf_val) in per_frame_raw[fi]:
            plate     = (x1, y1, x2, y2)
            in_vehicle = any(_overlaps(plate, vb) for vb in filter_boxes)
            if vehicle_filter != "all" and not in_vehicle:
                continue
            required = plate_conf_in_vehicle if in_vehicle else plate_conf
            if conf_val < required:
                continue
            plate_rects.append((x1, y1, x2, y2, conf_val, "sahi"))
        results.append((merge_overlapping(plate_rects), all_vehicles))

    return results
