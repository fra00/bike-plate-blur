# Agent instructions for https://github.com/fra00/bike-plate-blur

## What this repo is
CLI tool that detects and blurs license plates in video (`blur_plates.py`),
with motorcycle-oriented detection (letterbox crops, rear ROI) and temporal
tracking (prefer `--zone-filter kalman`). Config: `config.toml`.

## Setup checklist (do not skip)
1. Python 3.11/3.12 + ffmpeg/ffprobe on PATH
2. `python -m venv venv` → activate → install torch (CUDA if usable else CPU)
3. `pip install -r requirements.txt`
4. Confirm `license-plate-finetune-v1m.pt` and `yolov8n.pt` are in the project root (shipped in git)
5. Do not rely on `.engine` files across machines; use `.pt`
6. Smoke test: short `--start/--end` with `--debug`, then without debug + kalman

## Commands
```bash
python blur_plates.py input.mp4 output.mp4 --zone-filter kalman
python blur_plates.py input.mp4 output.mp4 --start 2:00 --end 2:30 --detect-cache cache/run.jsonl --zone-filter kalman
# Offline anti-blink (needs existing cache; keep kalman for quality):
python blur_plates.py input.mp4 output.mp4 --start 2:00 --end 2:30 --detect-cache cache/run.jsonl --zone-filter kalman --offline-zones
```

## Env
- `PLATE_DEVICE=cpu` or `cuda` to force device

## Do not commit
Videos, `cache/`, `venv/`, `dataset/`, `*.onnx`, `*.engine`, and extra local
`*.pt` experiments — see `.gitignore`. The two shipped checkpoints
(`license-plate-finetune-v1m.pt`, `yolov8n.pt`) **are** tracked.
