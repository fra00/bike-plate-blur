# plates-blur

**Detects and blurs license plates in video** (dashcam, action cam, motorcycle
footage). Optimized for **motorbikes**: lean angles, motion blur, small distant
plates, and tracking so the blur does not flicker or drift.

| | |
|---|---|
| **Repo** | https://github.com/fra00/plates-blur |
| **Entry point** | `python blur_plates.py <input> <output>` |
| **Config** | `config.toml` (no code edits needed for normal use) |
| **Output** | Visually lossless HEVC + **original audio** |

---

## What it does (plain language)

1. Finds **vehicles** (cars, motorcycles, buses, trucks).
2. Finds **license plates** (full frame + zoomed crops, especially moto rear).
3. **Tracks** plates over time so blur stays on the plate between detections.
4. Draws a **blur** (or solid color / sticker image) on each plate region.
5. Writes a new video file; your original file is never modified.

Optional: skip blur on motorcycles that are **too small** to read a plate
(`moto_min_blur_box_h_frac` in `config.toml`).

---

## Requirements (checklist)

| Need | Notes |
|------|--------|
| **Python 3.11 or 3.12** | 3.14 may work; 3.12 is safest |
| **ffmpeg** + **ffprobe** | Must be on your `PATH` |
| **Disk space** | ~2 GB for venv + models |
| **GPU (optional)** | NVIDIA + CUDA PyTorch = much faster; **CPU works** but is slow |
| **Model files** | Included in the repo: `license-plate-finetune-v1m.pt` (~39 MB) and `yolov8n.pt` (~6 MB) |

---

## Install — step by step (for anyone)

Do these in order. Copy-paste one block at a time.

### 1. Install Python

- **Windows:** https://www.python.org/downloads/ → install **3.12**, tick
  **“Add python.exe to PATH”**.
- **macOS:** `brew install python@3.12`
- **Linux:** `sudo apt install python3.12 python3.12-venv` (or your distro equivalent)

Check:

```bash
python --version
```

You should see `Python 3.11` or `3.12` (on some systems the command is `python3`).

### 2. Install ffmpeg

- **Windows:** `winget install ffmpeg` then **open a new terminal**
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Check:

```bash
ffmpeg -version
ffprobe -version
```

### 3. Download this project

```bash
git clone https://github.com/fra00/plates-blur.git
cd plates-blur
```

Or download the ZIP from GitHub → Extract → open a terminal **inside** that folder.

### 4. Create a virtual environment

**Windows (PowerShell / cmd):**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should show `(venv)`.

### 5. Install Python packages

**If you have an NVIDIA GPU** (recommended):

```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If `cu124` fails, try `cu121` or see https://pytorch.org/get-started/locally/

**CPU only** (no NVIDIA, or unsupported GPU):

```bash
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

Force CPU later anytime:

```bash
# Windows PowerShell
$env:PLATE_DEVICE="cpu"

# macOS / Linux
export PLATE_DEVICE=cpu
```

### 6. Model weights (already in the repo)

After `git clone`, you should already have these next to `blur_plates.py`:

```text
license-plate-finetune-v1m.pt   ← plate detector (~39 MB)
yolov8n.pt                      ← vehicle detector (~6 MB)
```

No extra download step. (ONNX/TensorRT `.engine` files stay local/optional and are
**not** portable between GPUs — the `.pt` files are enough.)

If a file is missing (shallow copy / incomplete download), re-clone or restore it
into the project root with that exact name.

### 7. First test (short clip)

Put any short video in the folder (or use a full path):

```bash
python blur_plates.py your_video.mp4 out_test.mp4 --start 0:00 --end 0:05 --debug
```

- `--debug` draws boxes (no privacy blur) so you can verify detection works.
- Then run **without** `--debug` for a real blur:

```bash
python blur_plates.py your_video.mp4 out_blurred.mp4 --start 0:00 --end 0:05 --zone-filter kalman
```

Open `out_blurred.mp4` in a player. If it fails, see [Troubleshooting](#troubleshooting).

---

## Everyday usage

### Blur a whole video

```bash
python blur_plates.py input.mp4 output.mp4 --zone-filter kalman
```

### Only a time range (faster while testing)

```bash
python blur_plates.py input.mp4 output.mp4 --start 2:00 --end 2:30 --zone-filter kalman
```

### Only motorbikes

```bash
python blur_plates.py input.mp4 output.mp4 --vehicles motorbike --zone-filter kalman
```

### Always cover your own plate (fixed rectangle)

```bash
python blur_plates.py input.mp4 output.mp4 --own-plate 1700,900,2200,1100
```

Coordinates are `x1,y1,x2,y2` in pixels of the video frame.

### Speed tip: detection cache

Detection is the slow part. Cache it once, then re-run tracking/blur for free:

```bash
python blur_plates.py input.mp4 output.mp4 --start 2:00 --end 2:30 ^
  --detect-cache cache/my_clip.jsonl --zone-filter kalman
```

(On macOS/Linux use `\` instead of `^` for line breaks.)

First run **writes** the cache; later runs **read** it (as long as detection
settings in `config.toml` did not change).

### Batch a folder

```bash
python batch_blur.py /path/to/videos --outdir /path/to/output --vehicles motorbike
```

### Call from another app (shell / subprocess)

Any software that can run a command:

```bash
python C:\path\to\plates-blur\blur_plates.py "C:\in\video.mp4" "C:\out\video.mp4" --zone-filter kalman
```

Exit code `0` = success. Activate the same `venv` (or use
`C:\path\to\plates-blur\venv\Scripts\python.exe` on Windows).

---

## Useful options

| Flag | Meaning |
|------|---------|
| `--zone-filter kalman` | Smoother blur tracking (recommended) |
| `--start` / `--end` | Process only a time range (`MM:SS`) |
| `--vehicles motorbike` | Ignore cars/trucks/buses |
| `--detect-cache PATH` | Save/replay detections (JSONL) |
| `--detect-scale 0.5` | Faster detection (default often `0.5` in config) |
| `--debug` | Show boxes instead of blur |
| `--debug-overlay` | Rich debug overlay |
| `--mode color --color 0,0,0` | Black boxes instead of blur |
| `--mode image --image logo.png` | Sticker/logo on plates |
| `--blur N` | Blur strength (odd number) |

Most defaults live in **`config.toml`**. Common knobs:

```toml
[detection]
detect_scale = 0.5                 # 0.5 = faster; 1.0 = max quality, slower
plate_conf = 0.45
plate_conf_in_vehicle = 0.15

[blur]
strength = 25
blur_shape = "round"

[tracking]
zone_filter = "ema"                # override with --zone-filter kalman
moto_min_blur_box_h_frac = 0.10    # skip tiny distant motos (~10% of frame height)
```

---

## Project layout (what matters)

```text
plates-blur/
  blur_plates.py          ← main CLI
  batch_blur.py           ← folder batch
  config.toml             ← tunables
  requirements.txt        ← pip packages
  license-plate-finetune-v1m.pt   ← YOU add this (not in git)
  plates/                 ← library code
  eval_detection.py       ← measure detection quality on a cache
  export_moto_crops.py    ← export crops for fine-tuning
  FINETUNE_V2M.md         ← optional train-your-own notes
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` not found | Reinstall Python with PATH enabled; try `python3` |
| `ffmpeg` not found | Install ffmpeg, **new** terminal, check `ffmpeg -version` |
| `No module named …` | Activate `venv`, then `pip install -r requirements.txt` |
| Missing `license-plate-finetune-v1m.pt` | Copy/download into project root (step 6) |
| CUDA / GPU errors | `PLATE_DEVICE=cpu` or install a PyTorch build that matches your GPU |
| Extremely slow | Normal on CPU; use short `--start/--end`, `detect_scale = 0.5`, GPU machine, or `--detect-cache` |
| Cache mismatch error | Delete the JSONL or pass a **new** `--detect-cache` path after changing detection settings |
| Blur too big / missing | Tune `[tracking]` in `config.toml`; prefer `--zone-filter kalman` |

---

## For LLMs / AI assistants

Paste this into ChatGPT, Claude, Cursor, etc. when helping a user set up or run the tool:

```text
You are helping set up and run https://github.com/fra00/plates-blur
(license-plate blur for dashcam / motorcycle video).

Goals:
1) Discover OS, Python version, ffmpeg on PATH, NVIDIA GPU — do not ask; check.
2) Guide clone → venv → torch (CUDA if usable else CPU) → pip install -r requirements.txt.
3) Confirm license-plate-finetune-v1m.pt and yolov8n.pt exist in the project root
   (they are committed in the repo, ~45 MB total).
4) Never recommend copying .engine TensorRT files across machines.
5) First verify with a 5s --debug run, then a real blur with --zone-filter kalman.
6) Prefer --start/--end and --detect-cache when iterating on a slow CPU.
7) Tunables live in config.toml; entry point is blur_plates.py.
8) Give one short copy-pasteable step at a time; diagnose failures before continuing.

Integration: callable as a shell/subprocess:
  python blur_plates.py <input> <output> [--start MM:SS] [--end MM:SS] [--zone-filter kalman]
```

---

## Optional: fine-tune on your own footage

See `FINETUNE_V2M.md` and the scripts `extract_training_frames.py`,
`review_annotations.py`, `augment_training_data.py`, `finetune.py`.
Only needed if the shipped model systematically misses your camera/angles.

---

## License & credits

- Detection: Ultralytics YOLO + SAHI-style tiling  
- Baseline plate family:
  [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)  
- This repo includes the fine-tuned plate weights (`license-plate-finetune-v1m.pt`)
  and the vehicle detector (`yolov8n.pt`) so a clone is runnable after `pip install`.
