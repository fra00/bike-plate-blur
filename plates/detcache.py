# ─── Detection cache (JSONL) ────────────────────────────────────────────────
"""
Per-frame cache of raw detector output, so tracking / smoothing / rendering can
be re-run without paying for inference again.

Detection dominates the runtime (seconds per frame on CPU), while tracking and
redaction are microseconds. Caching what the *detector* produced — before any
tracking — means every experiment on the temporal filters replays a 30 s clip in
seconds instead of tens of minutes, and compares runs against byte-identical
detections instead of a moving baseline.

Layout: one JSON object per line. The first line is a header holding the
detection parameters; a run whose parameters differ from the header is refused,
because rendering with stale detections would silently invalidate the result.
The last line is an end-of-file marker with the frame count, so an interrupted
write is detected as partial rather than treated as complete.
"""
import json
import os


_CACHE_VERSION = 1

# Frames between flushes while writing (~1 s of footage).
_FLUSH_EVERY = 30

# Parameters that change what the detector outputs. Anything outside this set
# (blur strength, tracking thresholds, encoder settings, ...) is applied after
# detection and therefore does not invalidate a cache.
_META_KEYS = (
    "input", "input_size", "start", "end", "width", "height",
    "vehicle_conf", "vehicle_filter", "plate_conf", "plate_conf_in_vehicle",
    "sahi_slice_size", "sahi_overlap", "detect_scale",
    "sharpen", "sharpen_amount", "sharpen_sigma", "vehicle_crop_scale",
    "moto_crop_scale", "moto_crop_bottom_frac", "moto_crop_side_pad_frac",
    "plate_crop_imgsz",
    "crop_clahe", "crop_clahe_clip", "crop_clahe_grid",
    "moto_close_conf", "plate_model", "vehicle_model",
)


def build_meta(**kwargs) -> dict:
    """Normalise the detection parameters into a comparable header dict."""
    meta = {"version": _CACHE_VERSION}
    for key in _META_KEYS:
        value = kwargs.get(key)
        if isinstance(value, float):
            value = round(value, 6)     # avoid float repr noise across runs
        meta[key] = value
    return meta


class CacheMismatch(RuntimeError):
    """Raised when an existing cache was produced with different parameters."""


def _is_zero_bound(v) -> bool:
    return v is None or v == 0 or v == 0.0


def _range_compatible(stored: dict, requested: dict) -> bool:
    """True if clip bounds match, or a full-video cache is replaying a subclip."""
    s_start, s_end = stored.get("start"), stored.get("end")
    r_start, r_end = requested.get("start"), requested.get("end")
    if s_start == r_start and s_end == r_end:
        return True
    full_cache = _is_zero_bound(s_start) and s_end is None
    return full_cache


class DetectionCache:
    """Read-or-write cache of (plates, vehicles) keyed by frame index.

    An existing file is read; a missing one is written as the run proceeds.
    """

    def __init__(self, path: str, meta: dict):
        self.path     = path
        self.meta     = meta
        self.reading  = os.path.exists(path)
        self.partial  = False
        self.frames   = 0
        self._data    = {}
        self._fh      = None
        self._written = 0
        self.stored_start = meta.get("start")
        self.stored_end = meta.get("end")

        if self.reading:
            self._load()
        else:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")
            self._fh.write(json.dumps(meta) + "\n")

    # ── reading ───────────────────────────────────────────────────────────────

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as fh:
            header = fh.readline()
            if not header.strip():
                raise CacheMismatch(f"empty detection cache: {self.path}")
            stored = json.loads(header)
            diff = [k for k in _META_KEYS
                    if k not in ("start", "end")
                    and stored.get(k) != self.meta.get(k)]
            if not _range_compatible(stored, self.meta):
                diff.extend(
                    k for k in ("start", "end")
                    if stored.get(k) != self.meta.get(k)
                )
            if stored.get("version") != _CACHE_VERSION or diff:
                raise CacheMismatch(
                    f"detection cache {os.path.basename(self.path)} was built with "
                    f"different settings ({', '.join(diff) or 'version'}) — delete it "
                    f"or pass a different --detect-cache path"
                )
            self.stored_start = stored.get("start")
            self.stored_end = stored.get("end")
            eof = False
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "eof" in entry:
                    eof = True
                    continue
                self._data[int(entry["f"])] = (
                    [tuple(p) for p in entry["p"]],
                    [tuple(v) for v in entry["v"]],
                )
        self.frames  = len(self._data)
        self.partial = not eof

    def replay_index(self, clip_frame: int, fps: float) -> int:
        """Map a clip-local frame to the cache index (full-video cache + --start)."""
        if not self.reading:
            return clip_frame
        if not (_is_zero_bound(self.stored_start) and self.stored_end is None):
            return clip_frame
        req_start = self.meta.get("start")
        if _is_zero_bound(req_start):
            return clip_frame
        return int(clip_frame) + int(round(float(req_start) * float(fps)))

    def get(self, frame_idx: int):
        """Return (plates, vehicles) for *frame_idx*, or None when absent."""
        return self._data.get(frame_idx)

    def frame_indices(self) -> list:
        """Sorted frame indices held by a cache being read."""
        return sorted(self._data)

    # ── writing ───────────────────────────────────────────────────────────────

    def put(self, frame_idx: int, plates: list, vehicles: list) -> None:
        stored = (
            [tuple(p) for p in plates],
            [tuple(v) for v in vehicles],
        )
        self._data[frame_idx] = stored
        if self._fh is None:
            return
        self._fh.write(json.dumps({
            "f": frame_idx,
            "p": [list(p) for p in plates],
            "v": [list(v) for v in vehicles],
        }) + "\n")
        self._written += 1
        # Populating a cache costs tens of minutes of inference. Flushing
        # regularly means an interrupted run leaves a usable partial cache (the
        # reader already reports and tolerates one) instead of losing everything
        # still sitting in the buffer — and lets a long run be evaluated while
        # it is still going.
        if self._written % _FLUSH_EVERY == 0:
            self._fh.flush()

    def finish_write(self) -> None:
        """Close a write-mode cache so the in-memory frames can be interpolated."""
        self.close()
        self.reading = True
        self.partial = False
        self.frames = len(self._data)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps({"eof": self._written}) + "\n")
            self._fh.close()
            self._fh = None
            self.frames = self._written if self._written else len(self._data)


class DetectionStore:
    """In-memory (plates, vehicles) store with the DetectionCache read API."""

    def __init__(self):
        self._data = {}
        self.reading = False
        self.partial = False
        self.frames = 0

    def put(self, frame_idx: int, plates: list, vehicles: list) -> None:
        self._data[frame_idx] = (
            [tuple(p) for p in plates],
            [tuple(v) for v in vehicles],
        )

    def get(self, frame_idx: int):
        return self._data.get(frame_idx)

    def replay_index(self, clip_frame: int, fps: float) -> int:
        return clip_frame

    def frame_indices(self) -> list:
        return sorted(self._data)

    def finish_write(self) -> None:
        self.reading = True
        self.frames = len(self._data)

    def close(self) -> None:
        self.finish_write()

