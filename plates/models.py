# ─── Model loading & fixed-batch inference ─────────────────────────────────
import os

from sahi import AutoDetectionModel

from plates.constants import PLATE_MODEL_PATH, TRT_BATCH, VEHICLE_MODEL_PATH


# Set by load_models(): True when a fixed-batch TensorRT engine was loaded, so
# _infer_batched() knows whether the batch padding is actually required.
_TRT_ACTIVE = False


def _cuda_usable() -> bool:
    """True when this PyTorch build has kernels for the installed GPU.

    torch.cuda.is_available() only reports that a driver and device exist: a
    GPU whose compute capability is older than every architecture the wheel was
    compiled for still raises `no kernel image is available for execution on
    the device` on the first kernel launch. Comparing the device capability
    against torch.cuda.get_arch_list() catches that before any model is loaded.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    device_cc = major * 10 + minor
    archs = []
    for arch in torch.cuda.get_arch_list():       # 'sm_86', 'compute_86', ...
        num = arch.split("_")[-1]
        if num.isdigit():
            archs.append(int(num))
    if not archs:
        return True     # build info unavailable — trust torch
    # A binary built for sm_N runs on hardware of the same major version with a
    # capability >= N (CUDA minor-version compatibility).
    return any(a // 10 == major and a <= device_cc for a in archs)


def _pick_device() -> str:
    """Select the inference device, honouring the PLATE_DEVICE override.

    PLATE_DEVICE=cpu forces CPU inference (useful to sidestep an unsupported
    GPU); PLATE_DEVICE=cuda restores the unconditional GPU behaviour.
    """
    forced = os.environ.get("PLATE_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    return "cuda" if _cuda_usable() else "cpu"


def _prefer_engine(path: str, device: str = "cuda") -> str:
    """Return the .engine sibling of *path* when it exists, else the .pt.

    A TensorRT engine only runs on the GPU it was built for, so on CPU the
    .pt weights are the only usable option.
    """
    if device != "cuda":
        return path
    eng = os.path.splitext(path)[0] + ".engine"
    return eng if os.path.exists(eng) else path


def _pad_to_batch(images: list, batch: int = TRT_BATCH) -> list:
    """Pad *images* to a multiple of *batch* by repeating the last image.

    TensorRT engines exported with a fixed batch size reject smaller inputs,
    so duplicate tiles are appended and their results discarded afterwards.
    """
    rem = len(images) % batch
    if rem == 0:
        return images
    pad = images[-1]
    return images + [pad] * (batch - rem)


def _infer_batched(model, images: list, conf: float, **kwargs) -> list:
    """Run *images* through the plate model, chunked to TRT_BATCH.

    Returns one result per input image (padding duplicates are dropped).
    The padding is skipped for .pt weights, which accept any batch size: a
    1080p frame yields 2 tiles, so padding them to 4 would double the work.
    """
    out = []
    for start in range(0, len(images), TRT_BATCH):
        chunk = images[start:start + TRT_BATCH]
        batch = _pad_to_batch(chunk, TRT_BATCH) if _TRT_ACTIVE else chunk
        results = model(batch, conf=conf, verbose=False, **kwargs)
        out.extend(results[:len(chunk)])
    return out


def load_models(plate_conf: float = 0.07):
    """Load YOLOv8 vehicle detector + SAHI-wrapped license plate detector.

    plate_conf is set to the lowest threshold that will be used so SAHI doesn't
    discard low-confidence detections before context-aware filtering can run.
    """
    global _TRT_ACTIVE
    import torch
    from ultralytics import YOLO

    device = _pick_device()
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"  Device : {device.upper()}" + (f"  ({gpu_name})" if device == "cuda" else " (install CUDA PyTorch for GPU acceleration)"))

    plate_path = _prefer_engine(PLATE_MODEL_PATH, device)
    _TRT_ACTIVE = plate_path.endswith(".engine")

    vehicle_model = YOLO(_prefer_engine(VEHICLE_MODEL_PATH, device))
    if str(getattr(vehicle_model, "ckpt_path", "")).endswith(".pt"):
        vehicle_model.to(device)

    plate_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=plate_path,
        confidence_threshold=plate_conf,
        device=device,
    )

    return vehicle_model, plate_model, device
