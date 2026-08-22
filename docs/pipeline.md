# Pipeline

Two passes over the video. Pass 1 can be skipped on later runs if you pass
`--detect-cache`.

```text
frame
  → YOLO vehicles (yolov8s.pt, COCO: car / motorcycle / bus / truck)
  → letterbox crops of each vehicle (motos: full box + rear ROI)
  → plate model (license-plate-finetune-v1m.pt)
  → JSONL cache (raw boxes only)
  → tracklets: keep plates whose centre sits in a vehicle box
  → interpolate short gaps (plate pose relative to that vehicle)
  → Gaussian blur (or colour / image) + HEVC + original audio
```

There is no full-frame plate scan, no SAHI tiling, no online Kalman/EMA
tracker, and no geometric “moto circle” fallback. Redaction is plate boxes
only (observed or interpolated).

## Pass 1 — detection

Vehicle boxes are de-duplicated (same class, high IoU or nested). Motorcycle
boxes below `moto_min_conf` are dropped after YOLO (`moto_close_conf` is the
detector floor so a close bike can still fire).

The plate model never sees the whole frame. Each kept vehicle is letterboxed
onto `plate_crop_imgsz` (upscale capped by `vehicle_crop_scale` /
`moto_crop_scale`). Motorcycles get two canvases:

- `crop` — padded full vehicle box
- `crop_moto` — lower `moto_crop_bottom_frac` of the box (rear plate)

Hits are mapped back to frame coordinates. A plate is kept only if its centre
lies inside a vehicle box (anywhere in the box is fine). Crop remaps that land
on empty road or the rider’s head are dropped. Cached `sahi` boxes from old
JSONL files are ignored when building zones.

Detection runs at `DETECT_WIDTH` (1280 px wide); blur is always full
resolution.

## Pass 2 — zones then redact

`build_zones` reads the **full** cache (not a rolling window):

1. Associate each plate to the tightest vehicle that contains it.
2. Drop standalone plates, tiny sides (`min_plate_side_px`), plates on motos
   below the size gate, nested blobs, and size jumps vs the vehicle’s peak
   plate/vehicle area ratio (`max_area_ratio`).
3. Chain hits into tracklets (same vehicle IoU / class-flip window).
4. Fill gaps of at most `max_gap_frames` by interpolating plate position and
   size **relative to the vehicle**. Travel must pass the motion gate.
   Cuts and teleports are not bridged.

The encoder then blurs those rectangles (`[blur]` padding / strength /
`max_box_frac`). `--own-plate` is a fixed extra rectangle. `--fixlist` adds
forced rectangles on a full-video run only.

Debug overlay vehicle boxes come from the **cache** (YOLO), not from a
hold/fork tracker. Magenta vs green is the moto size gate with hysteresis.

## Cache

`--detect-cache path.jsonl` writes on first run and replays later. The file is
refused if detection settings in the header no longer match `config.toml`
(crop size, ROI, CLAHE, confidences, models, …). Changing `[zones]` or
`[blur]` does **not** require a new cache.

Local caches, videos, `dataset/`, `runs/`, and extra `.pt` files are gitignored.
The two shipped weights (`license-plate-finetune-v1m.pt`, `yolov8s.pt`) are
tracked.
