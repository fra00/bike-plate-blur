# Agent instructions for https://github.com/fra00/bike-plate-blur

## What this repo is
CLI that detects and blurs license plates in video (`blur_plates.py`).
Motorcycle-oriented: letterbox vehicle crops, extra rear ROI on motos.
Plates are kept only inside vehicle boxes; blur zones are interpolated from
the full detection cache. No full-frame plate scan, no SAHI, no geometric
moto-circle fallback. Config: `config.toml`. Architecture: `docs/pipeline.md`.

## Setup checklist (do not skip)
1. Python 3.11/3.12 + ffmpeg/ffprobe on PATH
2. `python -m venv venv` → activate → install torch (CUDA if usable else CPU)
3. `pip install -r requirements.txt`
4. Confirm `license-plate-finetune-v1m.pt` and `yolov8s.pt` are in the project root (shipped in git)
5. Do not rely on `.engine` files across machines; use `.pt`
6. Smoke test: short `--start/--end` with `--debug`, then without debug

## Commands
```bash
python blur_plates.py input.mp4 output.mp4
python blur_plates.py input.mp4 output.mp4 --start 2:00 --end 2:30 --detect-cache cache/run.jsonl
```

## Env
- `PLATE_DEVICE=cpu` or `cuda` to force device

## Do not commit
Videos, `cache/`, `venv/`, `dataset/`, `runs/`, `training_data*`, `*.onnx`,
`*.engine`, and extra local `*.pt` experiments — see `.gitignore`. The two
shipped checkpoints (`license-plate-finetune-v1m.pt`, `yolov8s.pt`) **are**
tracked.
