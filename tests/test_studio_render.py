"""Tests for Work Package 1: Studio rendering quality, overlays, smart reframing, formats, and parallel rendering.

Zero network required: uses a small synthetic recording generated in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pullreel.cinema_engine import CinemaEngine, CursorSprite, resolve_canvas_size
from pullreel.exporter import (
    export_gif,
    export_mp4,
    export_poster,
    export_screenshot_sheet,
    export_webm,
    export_webp,
    probe,
)
from pullreel.models import (
    CaptionEvent,
    ClickEvent,
    KeyEvent,
    Recording,
    RenderSettings,
    SourceFrame,
    StudioTheme,
    TelemetryFrame,
)
from pullreel.overlays import bundled_font_path, format_keystroke_badges, load_font


def _make_synthetic_recording(tmp_path: Path, duration_s: float = 1.2) -> Recording:
    """Create a tiny 2x Retina synthetic recording with clicks, keystrokes, captions, and focus transitions."""
    frames: list[SourceFrame] = []
    for k in range(6):
        img = np.full((1600, 2560, 3), (18, 18, 24), dtype=np.uint8)
        # Simulated UI card + button + table
        cv2.rectangle(img, (200, 200), (1400, 900), (36, 36, 48), -1)
        cv2.rectangle(img, (260, 260), (620, 360), (139, 92, 246), -1)
        cv2.rectangle(img, (1500, 500), (2300, 1200), (45, 70, 95), -1)
        cv2.circle(img, (320 + k * 80, 310), 22, (240, 240, 255), -1)
        p = tmp_path / f"src_{k:03d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append(SourceFrame(t=k * (duration_s / 6.0), path=p))

    tel: list[TelemetryFrame] = []
    steps = int(duration_s * 60) + 1
    for i in range(steps):
        t = i / 60.0
        if t < 0.3:
            fb, z, mode, ctype = None, 1.0, "wide", "arrow"
        elif t < 0.7:
            fb, z, mode, ctype = (130.0, 130.0, 180.0, 50.0), 1.75, "focus", "text"
        else:
            fb, z, mode, ctype = (750.0, 250.0, 400.0, 350.0), 1.45, "focus", "pointer"
        tel.append(
            TelemetryFrame(
                t=t,
                mouse_x=180.0 + i * 10.0,
                mouse_y=155.0 + i * 3.5,
                is_clicking=(abs(t - 0.5) < 0.05),
                target_zoom=z,
                focus_box=fb,
                camera_mode=mode,  # type: ignore[arg-type]
                cursor_type=ctype,  # type: ignore[arg-type]
            )
        )

    keys = [
        KeyEvent(t=0.35, key="P"),
        KeyEvent(t=0.40, key="r"),
        KeyEvent(t=0.45, key="o"),
        KeyEvent(t=0.52, key="Enter", is_shortcut=True),
    ]
    captions = [
        CaptionEvent(t_start=0.25, t_end=0.68, text="Enter a realistic player name", step_index=1, action="fill",
                     focus_box=(130.0, 130.0, 180.0, 50.0)),
        CaptionEvent(t_start=0.70, t_end=1.15, text="Inspect the updated roster table", step_index=2, action="show",
                     focus_box=(750.0, 250.0, 400.0, 350.0)),
    ]
    return Recording(
        viewport=(1280, 800),
        scale=2.0,
        frames=frames,
        telemetry=tel,
        clicks=[ClickEvent(t=0.5, x=480.0, y=260.0)],
        keys=keys,
        captions=captions,
        url="https://app.example.com/roster",
        feature_summary="Add a new player to the roster",
        t_start=0.0,
        t_end=duration_s,
    )


def test_bundled_inter_font_loads_with_weights() -> None:
    path = bundled_font_path()
    assert path is not None and path.is_file()
    f_reg = load_font(24, weight=400)
    f_bold = load_font(24, weight=700)
    assert f_reg.getbbox("PullReel") is not None
    assert f_bold.getbbox("PullReel") is not None


def test_cursor_sprites_build_all_three_types() -> None:
    for kind in ("arrow", "pointer", "text"):
        sp = CursorSprite.build(kind)  # type: ignore[arg-type]
        assert sp.rgb.ndim == 3 and sp.rgb.shape[2] == 3
        assert sp.alpha.shape[:2] == sp.rgb.shape[:2]
        assert float(sp.alpha.max()) > 0.9


@pytest.mark.parametrize(
    ("aspect", "resolution", "expected"),
    [
        ("16:9", "1080p", (1920, 1080)),
        ("1:1", "1080p", (1080, 1080)),
        ("9:16", "1080p", (1080, 1920)),
        ("4:5", "1080p", (1080, 1350)),
        ("16:9", "1440p", (2560, 1440)),
        ("16:9", "4k", (3840, 2160)),
    ],
)
def test_resolution_and_aspect_targets(aspect: str, resolution: str, expected: tuple[int, int]) -> None:
    th = StudioTheme(aspect=aspect, resolution=resolution)  # type: ignore[arg-type]
    assert resolve_canvas_size((1280, 800), th, 38) == expected


def test_backgrounds_chrome_url_bar_and_zero_padding(tmp_path: Path) -> None:
    rec = _make_synthetic_recording(tmp_path, duration_s=0.4)
    # Create a small wallpaper image
    wp = tmp_path / "wall.png"
    wall = np.zeros((300, 400, 3), dtype=np.uint8)
    wall[:, :200] = (180, 80, 40)
    wall[:, 200:] = (40, 160, 120)
    cv2.imwrite(str(wp), wall)

    configs = [
        StudioTheme(background_preset="midnight", window_chrome="macos", show_url_bar=True, output_fps=15),
        StudioTheme(background_preset="light", window_chrome="windows", show_url_bar=True, output_fps=15),
        StudioTheme(background_style="wallpaper", wallpaper_path=str(wp), wallpaper_blur=18,
                    window_chrome="none", padding_px=0, output_fps=15),
        StudioTheme(background_style="solid", background="#101018", radius_px=0, border_px=0, output_fps=15),
    ]
    for th in configs:
        eng = CinemaEngine(rec, th, RenderSettings(workers=1))
        first = next(eng.frames())
        assert first.shape == (eng.frame_size[1], eng.frame_size[0], 3)
        assert eng.frame_size[0] % 2 == 0 and eng.frame_size[1] % 2 == 0


def test_smart_reframing_follows_focus_box_in_vertical_aspect(tmp_path: Path) -> None:
    rec = _make_synthetic_recording(tmp_path, duration_s=1.0)
    th = StudioTheme(aspect="9:16", resolution="native", output_fps=20)
    eng = CinemaEngine(rec, th, RenderSettings(workers=1))
    assert eng.frame_size[0] < eng.frame_size[1]  # vertical
    assert eng.vis_vw < eng.vw  # cropped horizontally to avoid stretching
    # Early focus is on the left (x~220), late focus is on the right (x~950): camera x must pan right.
    early_x = eng._camera_plan[8].cam_x
    late_x = eng._camera_plan[-1].cam_x
    assert late_x > early_x + 150.0


def test_focus_padding_prevents_clipping_large_subject(tmp_path: Path) -> None:
    rec = _make_synthetic_recording(tmp_path, duration_s=0.5)
    th = StudioTheme(output_fps=20)
    eng = CinemaEngine(rec, th, RenderSettings(focus_padding_px=64.0, workers=1))
    # For the late focus box (w=400, h=350) + 2*64 padding -> fit zoom must keep full box inside viewport
    late = eng._camera_plan[-1]
    vis_w = eng.vis_vw / late.zoom
    vis_h = eng.vis_vh / late.zoom
    assert vis_w >= 400.0 + 64.0
    assert vis_h >= 350.0 + 64.0


def test_overlays_intro_outro_captions_keystrokes_and_screenshots(tmp_path: Path) -> None:
    rec = _make_synthetic_recording(tmp_path, duration_s=1.0)
    th = StudioTheme(
        output_fps=20,
        intro_enabled=True,
        intro_title="Roster Builder",
        intro_subtitle="Create and compare players",
        intro_duration_s=0.4,
        outro_enabled=True,
        outro_title="Try PullReel",
        outro_cta="pullreel.dev",
        outro_duration_s=0.4,
        show_captions=True,
        show_keystrokes=True,
        screenshots=True,
    )
    eng = CinemaEngine(rec, th, RenderSettings(workers=2))
    all_frames = list(eng.frames())
    assert len(all_frames) == eng.frame_count
    assert eng.intro_frames == 8 and eng.outro_frames == 8
    assert eng.best_frame is not None
    assert len(eng.key_screenshots) >= 1

    badges, alpha = format_keystroke_badges(rec.keys, 0.6)
    assert "Pro" in badges and any("Enter" in b for b in badges) and alpha > 0.5

    sheet_path, indiv = export_screenshot_sheet(eng.key_screenshots, tmp_path / "demo-screenshots.png", th)
    assert sheet_path.is_file() and sheet_path.stat().st_size > 1000
    assert len(indiv) == len(eng.key_screenshots) and indiv[0].is_file()


def test_parallel_and_serial_rendering_produce_identical_frames(tmp_path: Path) -> None:
    rec = _make_synthetic_recording(tmp_path, duration_s=0.6)
    th = StudioTheme(output_fps=20, window_chrome="macos")
    serial = list(CinemaEngine(rec, th, RenderSettings(workers=1)).frames())
    parallel = list(CinemaEngine(rec, th, RenderSettings(workers=3)).frames())
    assert len(serial) == len(parallel)
    for a, b in zip(serial, parallel):
        assert np.array_equal(a, b)


def test_export_all_formats_mp4_webm_gif_webp_poster(tmp_path: Path) -> None:
    rec = _make_synthetic_recording(tmp_path, duration_s=0.5)
    th, st = StudioTheme(aspect="16:9", resolution="native").with_quality("draft")
    th.output_fps = 15
    eng = CinemaEngine(rec, th, st)

    mp4 = export_mp4(eng.frames(), eng.frame_size, th.output_fps, tmp_path / "clip.mp4", crf=th.crf, preset=th.x264_preset)
    assert mp4.is_file() and mp4.stat().st_size > 1000
    assert "h264" in probe(mp4).lower()

    poster = export_poster(eng.best_frame if eng.best_frame is not None else eng.last_frame, tmp_path / "clip-poster.png")  # type: ignore[arg-type]
    assert poster.is_file() and poster.stat().st_size > 1000

    gif_res = export_gif(mp4, tmp_path / "clip.gif", fps=10, max_mb=5.0)
    assert gif_res.path.is_file() and gif_res.fits

    webp = export_webp(mp4, tmp_path / "clip.webp", fps=12, quality=75, max_width=640)
    assert webp.is_file() and webp.stat().st_size > 500

    webm = export_webm(mp4, tmp_path / "clip.webm", crf=32, cpu_used=6)
    assert webm.is_file() and webm.stat().st_size > 500
    assert "vp9" in probe(webm).lower()
