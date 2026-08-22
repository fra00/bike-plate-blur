# Config and CLI

Edit [`config.toml`](../config.toml) for normal use. CLI flags override a
subset of the same values for one run.

Defaults below match the file in the repo.

## `[detection]`

| Key | Default | Role |
|-----|---------|------|
| `vehicle_conf` | `0.35` | YOLO vehicle confidence (`--conf`) |
| `plate_conf` | `0.45` | Stored in the cache header; new inference uses the in-vehicle threshold |
| `plate_conf_in_vehicle` | `0.15` | Plate model threshold on vehicle crops (`--plate-conf-in-vehicle`) |
| `moto_close_conf` | `0.20` | Lower YOLO floor so a close motorcycle can still fire |
| `moto_min_conf` | `0.30` | After YOLO: drop moto boxes below this (replay-safe, not stored in cache) |

Full-frame `detect_scale` and SAHI tiling were removed. Those CLI/cache header
fields still exist only so old JSONL files can be replayed.

## `[blur]`

| Key | Default | Role |
|-----|---------|------|
| `strength` | `35` | Gaussian kernel (odd); `--blur`. Close plates get a larger kernel |
| `padding` | `10` | Extra pixels around each zone |
| `max_box_frac` | `0.25` | Cap on a single box vs frame size |

## `[zones]`

| Key | Default | Role |
|-----|---------|------|
| `max_gap_frames` | `30` | Max missed frames to interpolate (~1 s at 30 fps) |
| `max_disp_px` / `max_disp_frac` | `80` / `0.35` | Motion gate: travel ≈ `max(px, vehicle_diag × frac) × gap` |
| `min_vehicle_iou` | `0.15` | Same-vehicle IoU for tracklets |
| `conf_floor` | `0.15` | Ignore weaker cached plates when building tracklets |
| `min_plate_side_px` | `12` | Drop specks |
| `max_area_ratio` | `2.5` | Drop a nested/shrunk blob vs the vehicle’s peak plate/vehicle ratio |
| `max_class_flip_frames` | `10` | Bridge moto ↔ car/truck YOLO flips |
| `moto_min_blur_box_h_frac` | `0.08333` | Skip motos whose max side is below this × frame height (~90 px at 1080p) |
| `moto_size_enter_frac` / `moto_size_exit_frac` | `1.15` / `0.85` | Hysteresis so rider+bike vs rear-only does not flicker the size gate |

There is no `moto_base_blur` (geometric circle). Coverage is plate tracklets
only.

## `[preprocessing]`

| Key | Default | Role |
|-----|---------|------|
| `sharpen` | `false` | Unsharp on the **crop**. Keep off on typical dashcam |
| `crop_clahe` | `true` | CLAHE on crop lightness (shadow / low contrast) |
| `plate_crop_imgsz` | `1280` | Letterbox canvas the plate model sees |
| `vehicle_crop_scale` / `moto_crop_scale` | `2.0` | Max upscale onto that canvas |
| `moto_crop_bottom_frac` | `0.28` | Rear ROI start (fraction of box height from the top) |
| `moto_crop_side_pad_frac` | `0.05` | Horizontal pad on the rear ROI |

Changing these invalidates `--detect-cache`.

## `[redact]` / `[output]`

`mode` is `blur`, `color`, or `image` (`--mode`, `--color`, `--image`).
Output is FFV1 intermediate → HEVC (`preset`, `quality` as CRF/CQ 18).
`tmp_dir = "auto"` uses the OS temp directory.

## CLI (on top of `input` / `output`)

| Flag | Meaning |
|------|---------|
| `--start` / `--end` | Time range (`MM:SS` or `HH:MM:SS`) |
| `--vehicles motorbike` | Only that COCO class (`all`, `motorbike`, `car`, `bus`, `truck`) |
| `--own-plate x1,y1,x2,y2` | Always-redact rectangle (camera behind your own plate) |
| `--detect-cache PATH` | Write/replay JSONL detections |
| `--debug` / `--debug-overlay` / `--debug-hud` | Overlay instead of (or on top of) a clean encode; see [debug.md](debug.md) |
| `--fixlist PATH` | Extra rectangles from an audit JSON; full-video runs only |
| `--blur` / `--conf` / `--plate-conf` / `--plate-conf-in-vehicle` | Override config for one run |

```bash
python blur_plates.py input.mp4 output.mp4
python blur_plates.py input.mp4 output.mp4 --start 2:00 --end 2:30 --detect-cache cache/run.jsonl
python batch_blur.py /path/to/videos --outdir /path/to/output --vehicles motorbike
```

`PLATE_DEVICE=cpu` or `cuda` forces the torch device.
