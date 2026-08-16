# ─── FFmpeg pipeline helpers ────────────────────────────────────────────────
import json
import os
import subprocess
import tempfile
import threading

from tqdm import tqdm

from plates.constants import _CUVID_DECODERS


def get_video_info(video_path: str) -> dict:
    """Return width, height, fps, codec via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    vs = next(s for s in streams if s["codec_type"] == "video")
    num, den = map(int, vs["r_frame_rate"].split("/"))
    duration = float(vs.get("duration") or
                     json.loads(subprocess.run(
                         ["ffprobe", "-v", "quiet", "-print_format", "json",
                          "-show_format", video_path],
                         capture_output=True, text=True).stdout)["format"]["duration"])

    stored_w = int(vs["width"])
    stored_h = int(vs["height"])

    # Detect rotation metadata — ffmpeg auto-rotates output by default.
    # For 90°/270° clips (e.g. iPhone, GoPro portrait) the display dimensions
    # are swapped vs the stored stream dimensions.
    rotate = int(vs.get("tags", {}).get("rotate", 0))
    for sd in vs.get("side_data_list", []):          # newer ffmpeg uses display matrix
        if sd.get("side_data_type") == "Display Matrix":
            rotate = -int(sd.get("rotation", 0))
            break
    if abs(rotate) in (90, 270):
        stored_w, stored_h = stored_h, stored_w

    # Apply the same even-rounding the scale filter uses, so frame_size is exact.
    width  = (stored_w // 2) * 2
    height = (stored_h // 2) * 2

    return {
        "width":    width,
        "height":   height,
        "fps":      num / den,
        "codec":    vs["codec_name"],
        "pix_fmt":  vs.get("pix_fmt", "yuv420p"),
        "duration": duration,
    }

def build_ffmpeg_extract(input_path, start_sec, end_sec, codec=None):
    """Build ffmpeg command that pipes raw BGR frames to stdout.

    Uses NVIDIA CUVID hardware decode when the input codec is supported,
    falling back to software decode silently.
    """
    cmd = ["ffmpeg", "-y"]

    hw_decoder = _CUVID_DECODERS.get(codec or "")
    if hw_decoder:
        # CUVID decoder must come before -i; seek with -ss after -i for accuracy
        cmd += ["-c:v", hw_decoder]
        cmd += ["-i", input_path]
        if start_sec is not None:
            cmd += ["-ss", f"{start_sec:.6f}"]
    else:
        if start_sec is not None:
            cmd += ["-ss", f"{start_sec:.6f}"]
        cmd += ["-i", input_path]

    if end_sec is not None:
        duration = end_sec - (start_sec or 0.0)
        cmd += ["-t", f"{duration:.6f}"]

    # CUVID decoders output nv12 and sometimes emit wrong color-range metadata,
    # causing orange-cast corruption when scale converts nv12→bgr24 directly.
    # Inserting format=yuv420p forces a clean software nv12→yuv420p step first.
    vf = ("format=yuv420p," if hw_decoder else "") + "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", vf,
        "pipe:1",
    ]
    return cmd

def build_ffmpeg_encode_lossless(width, height, fps, out_path):
    """Build ffmpeg command that reads raw BGR frames from stdin → FFV1 lossless."""
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "ffv1",
        "-level", "3",
        "-threads", "0",
        out_path,
    ]

def _ffmpeg_with_progress(cmd, total_frames, desc="  Encoding"):
    """Run an ffmpeg command and show a tqdm frame progress bar. Returns (returncode, stderr)."""
    # Insert -progress pipe:1 -nostats right after 'ffmpeg'
    cmd = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]

    stderr_lines = []

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]"
    with tqdm(total=total_frames, unit="frame", desc=desc,
              dynamic_ncols=True, bar_format=bar_fmt) as pbar:
        current = 0
        for line in proc.stdout:
            if line.startswith("frame="):
                try:
                    new = int(line.split("=", 1)[1])
                    if new > current:
                        pbar.update(new - current)
                        current = new
                except ValueError:
                    pass

    t.join()
    proc.wait()
    return proc.returncode, "".join(stderr_lines)

def _source_color_args(source_path):
    """
    Probe the source video for color metadata (primaries, transfer, space,
    range) and return ffmpeg flags that preserve it on the output.

    Without this, the raw-BGR pipe in the middle of the pipeline strips all
    colorimetry — leaving the final encoded HEVC with color_primaries=unknown
    etc.  Different players then guess different colorspaces, which is exactly
    the colour-shift artefact users see on iPhone footage (BT.709 source ←→
    BT.601 default-guess).
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=color_primaries,color_transfer,color_space,color_range",
             "-of", "default=nw=1:nk=0", source_path],
            capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return []
    # Parse key=value pairs (order from ffprobe is alphabetical, not request order).
    fields = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        fields[k.strip()] = v.strip()

    args = []
    primaries = fields.get("color_primaries", "")
    transfer  = fields.get("color_transfer",  "")
    space     = fields.get("color_space",     "")
    rng       = fields.get("color_range",     "")
    if primaries and primaries != "unknown":
        args += ["-color_primaries", primaries]
    if transfer and transfer != "unknown":
        args += ["-color_trc", transfer]
    if space and space != "unknown":
        args += ["-colorspace", space]
    if rng and rng != "unknown":
        # ffmpeg's -color_range only accepts 'tv'/'pc' (or 1/2), not 'mpeg'/'jpeg'
        # — both refer to the same thing; pass the ffprobe label through as-is.
        args += ["-color_range", rng]
    return args

def mux_audio(video_only_path, original_path, output_path,
              start_sec, end_sec, fps, total_frames=None,
              preset="medium", tmp_dir="D:/pip-tmp", quality=18):
    """
    Combine processed video with original audio.
    Audio is extracted to a temp file first (resets timestamps to 0)
    so it stays in sync with the processed video regardless of clip position.
    Final encode: HEVC CRF 0 (lossless) video + original audio.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    # ── Extract audio segment to temp file (timestamps start at 0) ───────────
    with tempfile.NamedTemporaryFile(suffix=".mov", delete=False, dir=tmp_dir) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        audio_cmd = ["ffmpeg", "-y"]
        if start_sec is not None:
            audio_cmd += ["-ss", f"{start_sec:.6f}"]
        audio_cmd += ["-i", original_path]
        if end_sec is not None:
            duration = end_sec - (start_sec or 0.0)
            audio_cmd += ["-t", f"{duration:.6f}"]
        audio_cmd += [
            "-vn",                    # no video
            "-c:a", "copy",
            "-reset_timestamps", "1", # force PTS to start at 0
            tmp_audio_path,
        ]
        subprocess.run(audio_cmd, capture_output=True, check=True)

        # ── Mux video (FFV1, t=0) + extracted audio (t=0) → final output ─────
        # Use hevc_nvenc (GPU) if available, fall back to libx265 (CPU)
        def _nvenc_available():
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc",
                 "-t", "0", "-c:v", "hevc_nvenc", "-f", "null", "-"],
                capture_output=True)
            return r.returncode == 0

        # Capture source colorimetry so we can re-tag the output with the same
        # BT.709 (or whatever the source had) values.  The raw-BGR pipe strips
        # this metadata mid-pipeline and players have to guess otherwise.
        color_args = _source_color_args(original_path)

        if _nvenc_available():
            video_codec_args = [
                "-c:v", "hevc_nvenc",
                # Visually-lossless constant quality.  QP 0 used to produce
                # ~400 Mbps output that many players refused to open or froze
                # mid-playback; cq 18 is indistinguishable to the eye and
                # ~10× smaller.
                "-rc", "vbr",
                "-cq", str(quality),
                "-preset", "p4",    # p1=fastest … p7=slowest (GPU-side)
                "-bf", "0",         # match iPhone source (no B-frames)
                "-tag:v", "hvc1",   # QuickTime / macOS compatible
                "-pix_fmt", "yuv420p",
                *color_args,
            ]
            enc_label = "  Encoding (HEVC NVENC GPU + audio)"
        else:
            video_codec_args = [
                "-c:v", "libx265",
                "-crf", str(quality),   # visually lossless, ~10× smaller than crf 0
                "-preset", preset,
                "-tag:v", "hvc1",
                "-pix_fmt", "yuv420p",
                *color_args,
            ]
            enc_label = "  Encoding (HEVC CPU + audio)"

        mux_cmd = [
            "ffmpeg", "-y",
            "-i", video_only_path,   # processed video, PTS 0..N
            "-i", tmp_audio_path,    # audio, PTS 0..N
            "-map", "0:v:0",
            "-map", "1:a:0",
            *video_codec_args,
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        rc, stderr = _ffmpeg_with_progress(mux_cmd, total_frames, enc_label)
        if rc != 0:
            # Retry without audio
            print("  Warning: audio mux failed, retrying video-only...")
            print(stderr[-400:])
            mux_cmd_no_audio = [
                "ffmpeg", "-y",
                "-i", video_only_path,
                *video_codec_args,
                "-movflags", "+faststart",
                output_path,
            ]
            rc2, _ = _ffmpeg_with_progress(mux_cmd_no_audio, total_frames, "  Encoding (video-only)")
            if rc2 != 0:
                raise subprocess.CalledProcessError(rc2, mux_cmd_no_audio)

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

def estimate_frame_count(info, start_sec, end_sec):
    """Estimate total frames for the progress bar."""
    total_duration = info["duration"]
    clip_start = start_sec or 0.0
    clip_end   = end_sec   or total_duration
    return max(1, int((clip_end - clip_start) * info["fps"]))
