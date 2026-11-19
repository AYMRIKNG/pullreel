"""Build the README before/after GIFs from one kept recording.

    uv run pullreel demo --out docs --keep-frames
    uv run python scripts/make_readme_gifs.py <kept-dir> docs

"before.gif" is what a plain headless capture gives you: the same screencast frames, no cursor, no camera, no frame.
"after.gif" is PullReel's render of the same recording. Both share width and frame rate so they sit side by side.
"""

from __future__ import annotations

import subprocess
import sys
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np

from pullreel.cinema_engine import CinemaEngine
from pullreel.exporter import export_mp4, find_ffmpeg
from pullreel.models import Recording, StudioTheme

WIDTH, FPS = 640, 24


def raw_frames(rec: Recording, fps: int):
    ts = [f.t for f in rec.frames]
    cached: tuple[int, np.ndarray] | None = None
    n = int((rec.t_end - rec.t_start) * fps)
    for i in range(n):
        idx = max(0, bisect_right(ts, rec.t_start + i / fps) - 1)
        if cached is None or cached[0] != idx:
            img = cv2.imdecode(np.fromfile(str(rec.frames[idx].path), dtype=np.uint8), cv2.IMREAD_COLOR)
            cached = (idx, cv2.resize(img, rec.viewport, interpolation=cv2.INTER_AREA))
        yield cached[1]


def to_gif(mp4: Path, gif: Path) -> None:
    vf = (f"fps={FPS},scale={WIDTH}:-2:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
          "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")  # fmt: skip
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)],
                   check=True)  # fmt: skip
    print(f"{gif}: {gif.stat().st_size / 1_048_576:.2f} MB")


def main() -> None:
    kept, out = Path(sys.argv[1]), Path(sys.argv[2])
    rec = Recording.load_json(kept / "recording.json")
    out.mkdir(parents=True, exist_ok=True)
    before_mp4 = out / "_before.mp4"
    export_mp4(raw_frames(rec, 30), rec.viewport, 30, before_mp4)
    to_gif(before_mp4, out / "before.gif")
    before_mp4.unlink()
    engine = CinemaEngine(rec, StudioTheme())
    after_mp4 = out / "_after.mp4"
    export_mp4(engine.frames(), engine.frame_size, 60, after_mp4)
    to_gif(after_mp4, out / "after.gif")
    after_mp4.unlink()


if __name__ == "__main__":
    main()
