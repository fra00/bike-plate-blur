# Fine-tune the plate model

Use this only after letterbox crops + motorcycle rear ROI still miss plates
on **your** camera. The shipped `license-plate-finetune-v1m.pt` is the
production plate detector; `yolov8s.pt` is stock COCO and is not fine-tuned
here.

`dataset/`, extra `*.pt`, and `runs/` are gitignored.

## 1. Detection cache

Build (or reuse) a JSONL cache of the clip you will harvest crops from:

```bash
python blur_plates.py your_clip.mp4 _tmp.mp4 --start 2:00 --end 2:30 \
  --detect-cache cache/your_clip.jsonl
```

Changing crop size, ROI, or CLAHE requires a **new** cache path.

## 2. Export crops

```bash
python export_moto_crops.py your_clip.mp4 cache/your_clip.jsonl \
  --out dataset/v2m --start 2:00 --end 2:30 --max-pos 500 --max-neg 250 --every 1
```

Aim for a few hundred annotated positives; add other clips if one angle
dominates. Review / split with `review_annotations.py` and
`extract_training_frames.py` if you are labelling full frames instead of
exported rear crops.

## 3. Dataset layout

```text
dataset/v2m/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt   # YOLO class 0 = plate, coords relative to the crop
```

Split without leakage: disjoint time windows, not “even vs odd frames”.
Hard-negatives (`images/neg`) have no labels — pass them as background images
in the Ultralytics data YAML.

## 4. Train

```bash
yolo detect train model=yolo11n.pt data=dataset/v2m/data.yaml \
  imgsz=640 epochs=80 batch=16 \
  hsv_h=0.015 hsv_s=0.5 hsv_v=0.4 degrees=10 \
  blur=0.3 mosaic=0.5
```

Or resume from `license-plate-finetune-v1m.pt` if the architecture matches.
`augment_training_data.py` and `finetune.py` in the repo root wrap a similar
loop.

## 5. Deploy

1. Copy best weights next to `blur_plates.py` (gitignored unless you whitelist
   the name).
2. Point `PLATE_MODEL_PATH` in `plates/constants.py`.
3. Write a **new** `--detect-cache` JSONL (header includes the model path).
4. Compare with `eval_detection.py cache/new.jsonl` (detection-only metrics;
   tracking / blur settings do not change those numbers).

TensorRT `.engine` is optional and machine-specific (`plates/models.py`
`_prefer_engine`). Do not copy `.engine` files between GPUs.
