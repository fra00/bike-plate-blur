# ─── End-of-run summary printing ───────────────────────────────────────────
import os

from plates.common import _format_duration


def _hardware_label(device: str) -> str:
    """Short string describing the compute device used (GPU name + VRAM, or CPU)."""
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU - {name} ({total_gb:.1f} GB VRAM)"
        except Exception:
            return "GPU (CUDA)"
    # CPU fallback
    try:
        import platform
        cpu = platform.processor() or platform.machine() or "CPU"
        n   = os.cpu_count() or "?"
        return f"CPU - {cpu} ({n} threads)"
    except Exception:
        return "CPU"

def _print_run_summary(*, elapsed_total, elapsed_process, device, info,
                       input_path, output_path, frame_num, total_plates,
                       redact_mode, redact_color, redact_image_path,
                       vehicle_filter, total_quads=0):
    """Print a compact one-page summary at the end of a video run."""
    bar = "=" * 60

    in_name  = os.path.basename(input_path)
    out_name = os.path.basename(output_path)
    res      = f"{info['width']}x{info['height']}"
    in_fps   = info["fps"]
    in_codec = info.get("codec", "?")
    in_dur   = info.get("duration", 0.0)
    out_size = os.path.getsize(output_path) / 1024 / 1024 if os.path.exists(output_path) else 0.0

    proc_fps   = frame_num / elapsed_process if elapsed_process > 0 else 0.0
    speed_x    = (frame_num / in_fps) / elapsed_total if in_fps and elapsed_total > 0 else 0.0
    plates_pf  = total_plates / frame_num if frame_num else 0.0
    quads_pf   = total_quads / frame_num if frame_num else 0.0

    mode_desc = redact_mode
    if redact_mode == "color":
        b, g, r = redact_color
        mode_desc = f"color (R={r} G={g} B={b})"
    elif redact_mode == "image":
        mode_desc = f"image ({os.path.basename(redact_image_path)})" if redact_image_path else "image"

    print(f"\n{bar}")
    print(f"  Run summary")
    print(f"{bar}")
    print(f"  Hardware       :  {_hardware_label(device)}")
    print(f"  Input          :  {in_name}")
    print(f"                    {res} @ {in_fps:.2f} fps  |  {in_codec}  |  {_format_duration(in_dur)}")
    print(f"  Output         :  {out_name}")
    print(f"                    {out_size:.1f} MB  |  HEVC (lossless intermediate -> visually lossless mux)")
    print(f"  Redaction      :  {mode_desc}")
    if vehicle_filter and vehicle_filter != "all":
        print(f"  Vehicle filter :  {vehicle_filter} only")
    print(f"  Frames         :  {frame_num:,} processed")
    print(f"  Plates         :  {total_plates:,} redacted   ({plates_pf:.2f} per frame)")
    if total_quads:
        print(f"  Rotated quads  :  {total_quads:,} refined blur zones   ({quads_pf:.2f} per frame)")
    print(f"  Throughput     :  {proc_fps:.1f} fps processing  |  {speed_x:.2f}x realtime")
    print(f"  Total time     :  {_format_duration(elapsed_total)}")
    print(f"{bar}\n")
