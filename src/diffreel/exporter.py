"""FFmpeg pipeline: raw BGR frames -> MP4 (H.264, 60 fps, crf 18) -> GitHub-friendly GIF (two-stage palette)."""

from __future__ import annotations

import contextlib
import functools
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

GIF_START_WIDTH = 960
GIF_MIN_WIDTH = 720
GIF_MIN_FPS = 12


class ExportError(RuntimeError):
    pass


@dataclass(slots=True)
class GifResult:
    path: Path
    width: int
    fps: int
    size_mb: float
    fits: bool


def _runs(exe: str) -> bool:
    try:
        return subprocess.run([exe, "-version"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else os.environ.get("XDG_CACHE_HOME")
    return Path(base) / "diffreel" if base else Path.home() / ".cache" / "diffreel"


@functools.cache
def find_ffmpeg() -> str:
    """System ffmpeg if present, otherwise the static build bundled with imageio-ffmpeg (zero setup).

    On Windows, a bundled binary living in a very deep virtualenv cannot be spawned (CreateProcess hits MAX_PATH),
    so it is copied once to a short cache path.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    with contextlib.suppress(RuntimeError):
        return imageio_ffmpeg.get_ffmpeg_exe()
    bundled = sorted((Path(imageio_ffmpeg.__file__).parent / "binaries").glob("ffmpeg*"))
    bundled = [p for p in bundled if p.suffix in ("", ".exe")]
    if not bundled:
        raise ExportError("ffmpeg not found: install it (e.g. `winget install ffmpeg`, `brew install ffmpeg`).")
    src = bundled[0]
    dst = _cache_dir() / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if not _runs(str(dst)):
        raise ExportError(f"bundled ffmpeg at {dst} does not run; install ffmpeg on your PATH.")
    return str(dst)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-2500:]
        raise ExportError(f"ffmpeg failed ({proc.returncode}):\n{tail}")


def export_mp4(
    frames: Iterable[NDArray[np.uint8]],
    size: tuple[int, int],
    fps: int,
    out_path: Path,
    *,
    crf: int = 18,
    on_frame: Callable[[], None] | None = None,
) -> Path:
    w, h = size
    if w % 2 or h % 2:
        raise ExportError(f"frame size must be even for yuv420p, got {w}x{h}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-tune", "animation",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            if frame.shape[1] != w or frame.shape[0] != h:
                raise ExportError(f"frame is {frame.shape[1]}x{frame.shape[0]}, expected {w}x{h}")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if on_frame:
                on_frame()
    except BrokenPipeError:
        pass
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        if not proc.stdin.closed:
            proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise ExportError(f"ffmpeg (mp4) failed:\n{stderr[-2500:]}")
    return out_path


def export_gif(mp4_path: Path, out_path: Path, *, fps: int = 30, max_mb: float = 5.0) -> GifResult:
    """Two-stage palette GIF (palettegen stats_mode=diff + bayer dithering), shrunk until it fits ``max_mb``.

    GIF size grows roughly with width^2 * fps, so after the first pass the next settings are estimated from the
    budget: width shrinks first (down to GIF_MIN_WIDTH), then frame rate (down to GIF_MIN_FPS).
    """
    width, rate = GIF_START_WIDTH, fps
    result: GifResult | None = None
    for _ in range(4):
        vf = (
            f"fps={rate},scale={width}:-2:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=stats_mode=diff[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        )
        _run([find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4_path),
              "-vf", vf, "-loop", "0", str(out_path)])  # fmt: skip
        size_mb = out_path.stat().st_size / 1_048_576
        result = GifResult(out_path, width, rate, size_mb, size_mb <= max_mb)
        if result.fits or (width <= GIF_MIN_WIDTH and rate <= GIF_MIN_FPS):
            break
        budget = 0.9 * max_mb / size_mb  # how much of the current pixel*fps rate we can keep
        new_w = width * math.sqrt(budget)
        if new_w >= GIF_MIN_WIDTH:
            width = int(new_w) // 2 * 2
        else:
            budget *= (width / GIF_MIN_WIDTH) ** 2
            width = GIF_MIN_WIDTH
            rate = max(GIF_MIN_FPS, int(rate * budget))
    assert result is not None
    return result


def export_poster(frame: NDArray[np.uint8], path: Path) -> Path:
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise ExportError("could not encode poster PNG")
    buf.tofile(str(path))
    return path


def probe(path: Path) -> str:
    """Return ffmpeg's stream description line for ``path`` (for diagnostics)."""
    proc = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True)
    lines = proc.stderr.decode("utf-8", "replace").splitlines()
    return next((ln.strip() for ln in lines if "Video:" in ln), "")
