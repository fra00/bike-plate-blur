# Fine-tune plan — license-plate-finetune-v2m

Use only after detection letterbox + moto ROI hit a recall ceiling on the
reference clip (`cache/ref_200_230_det_v2.jsonl` A/B vs baseline).

## 1. Export crops

```text
python export_moto_crops.py testvideo/montage_20260808_224940.mp4 \
  cache/ref_200_230_det_v2.jsonl --out dataset/v2m \
  --start 2:00 --end 2:30 --max-pos 500 --max-neg 250 --every 1
```

Current export on the ref clip: **307 positives + 250 hard-negatives**, split in
`dataset/v2m/` (`data.yaml`, train/val by later frame window — no leakage).
Target overall remains **300–500** annotated positives (add other clips if needed).

## 2. Dataset layout

```text
dataset/v2m/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt   # YOLO class 0 = plate, coords relative to crop
```

Split without leakage: no shared absolute frame indices between train and val
(e.g. even frames → train, odd → val is weak; prefer disjoint time windows).

Hard-negatives (`images/neg`) have no labels — use as background images in the
Ultralytics data YAML (`background` / empty-label images).

## 3. Train

```text
yolo detect train model=yolo11n.pt data=dataset/v2m/data.yaml \
  imgsz=640 epochs=80 batch=16 \
  hsv_h=0.015 hsv_s=0.5 hsv_v=0.4 degrees=10 \
  blur=0.3 mosaic=0.5
```

Or resume from `license-plate-finetune-v1m.pt` if the architecture matches.

## 4. Deploy

1. Copy best weights → `license-plate-finetune-v2m.pt`
2. Point `PLATE_MODEL_PATH` in `plates/constants.py`
3. Rebuild detection cache as `cache/ref_200_230_det_v3.jsonl`
4. Re-run `eval_detection.py` vs baseline / det_v2

TensorRT `.engine` is optional and machine-specific (`plates/models.py`
`_prefer_engine`).
