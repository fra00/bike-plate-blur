# Debug overlay

`--debug` applies the same redaction as production, then draws boxes.

| Colour | Meaning |
|--------|---------|
| Green | Vehicle YOLO; motorcycle is large enough for plate zones |
| Magenta | Motorcycle below `moto_min_blur_box_h_frac` (with size hysteresis) |
| Blue | Plate-model hit that became a blur zone (`plate 0.xx`) |
| Grey | Plate-model hit that tracking dropped (`skip 0.xx`) |
| Yellow | Interpolated plate (gap fill, not a detection) |
| Orange | `--own-plate` rectangle |

`--debug-overlay` is the same idea with source tags (`CROP`, `CROP_MOTO`,
`BRIDGE`). `--debug-hud` adds a ~320 px side panel (frame index, counts,
timings).

There is no cyan “base” circle. If a motorcycle has no plate hit and no
interpolated tracklet, it is not redacted.
