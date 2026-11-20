"""Turn a raw recording into studio-grade 60 fps frames.

Upgraded cinema pipeline:
* Configurable studio chassis: solid / gradient / presets / blurred wallpaper, macOS / Windows / none window chrome,
  optional browser URL bar, adjustable corner radius, soft drop shadow, and safe padding.
* Smart reframing across aspect ratios (16:9, 1:1, 9:16, 4:5) and resolutions (native, 1080p, 1440p, 4k), following
  focus_box and cursor without stretching or clipping.
* Critically damped spring camera with smooth focus-box transitions, anticipation lookahead, focus padding, and
  speed-proportional temporal motion blur.
* High-resolution vector cursors (arrow, pointer hand, text I-beam), press squash (0.90), subtle drop shadow,
  fast-movement motion trail, and clean dual-ring click ripples.
* Animated overlays using the bundled Inter OFL font: intro title card, per-step captions, keystroke HUD, outro CTA.
* Money-shot best-frame tracker and annotated key screenshots for documentation.
* Parallel frame rendering (ThreadPoolExecutor + LRU source frame cache) piped directly to FFmpeg with zero disk PNGs.
"""

from __future__ import annotations

import math
import os
import threading
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from pullreel.models import (
    AspectRatio,
    Box,
    CursorType,
    Recording,
    RenderSettings,
    ResolutionPreset,
    StudioTheme,
    TelemetryFrame,
)
from pullreel.overlays import (
    annotate_screenshot,
    draw_caption,
    draw_keystrokes,
    ease_in_out_cubic,
    ease_out_cubic,
    hex_to_bgr,
    hex_to_rgb,
    is_light_color,
    load_font,
    render_title_card,
)

Frame = NDArray[np.uint8]
FloatImg = NDArray[np.float32]

# Vector cursor shapes in 1x "points".
_ARROW = [(0.0, 0.0), (0.0, 17.0), (4.2, 13.2), (7.0, 19.5), (9.6, 18.4), (6.9, 12.3), (12.3, 12.3)]
# Clean pointing hand cursor, fingertip hotspot near (5.5, 0.5).
_POINTER = [
    (5.5, 0.0), (7.5, 0.0), (7.8, 7.2),
    (9.5, 6.2), (11.2, 6.8), (11.5, 8.8),
    (13.0, 7.8), (14.6, 8.6), (14.8, 10.6),
    (16.0, 10.2), (17.2, 11.2), (16.8, 16.5),
    (14.5, 20.0), (6.8, 20.0), (3.6, 16.5),
    (0.8, 12.2), (1.8, 10.4), (3.8, 11.2), (5.0, 12.6),
    (4.6, 1.4),
]
_SPRITE_RES = 8  # sprite pixels per cursor point
_SPRITE_MARGIN = 4.0  # points of room for outline + shadow

_RESOLUTION_TARGETS: dict[ResolutionPreset, dict[AspectRatio, tuple[int, int]]] = {
    "1080p": {"16:9": (1920, 1080), "1:1": (1080, 1080), "9:16": (1080, 1920), "4:5": (1080, 1350)},
    "1440p": {"16:9": (2560, 1440), "1:1": (1440, 1440), "9:16": (1440, 2560), "4:5": (1440, 1800)},
    "4k": {"16:9": (3840, 2160), "1:1": (2160, 2160), "9:16": (2160, 3840), "4:5": (2160, 2700)},
}
_NATIVE_ASPECTS: dict[AspectRatio, tuple[int, int]] = {
    "16:9": (1440, 810),
    "1:1": (960, 960),
    "9:16": (720, 1280),
    "4:5": (864, 1080),
}


def _even(n: float) -> int:
    v = max(2, int(round(n)))
    return v if v % 2 == 0 else v + 1


def _rounded_rect_alpha(size: tuple[int, int], rect: tuple[float, float, float, float], radius: float) -> FloatImg:
    """Anti-aliased rounded-rectangle coverage mask (4x supersampled), float32 in [0, 1]."""
    w, h = size
    x0, y0, x1, y1 = rect
    if radius <= 0.5:
        mask = np.zeros((h, w), dtype=np.float32)
        ix0, iy0 = max(0, int(round(x0))), max(0, int(round(y0)))
        ix1, iy1 = min(w, int(round(x1))), min(h, int(round(y1)))
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = 1.0
        return mask
    ss = 4 if max(w, h) <= 2600 else 2
    img = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(img).rounded_rectangle(
        (x0 * ss, y0 * ss, x1 * ss - 1, y1 * ss - 1), radius=radius * ss, fill=255
    )
    img = img.resize((w, h), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0


@dataclass(slots=True)
class CursorSprite:
    rgb: FloatImg  # premultiplied BGR, float32 0..255
    alpha: FloatImg  # (h, w, 1) float32 0..1
    hotspot: tuple[float, float]  # sprite pixel coordinates of the hotspot

    @classmethod
    def _from_fill(cls, fill_a: FloatImg, hotspot: tuple[float, float], outline_pts: float = 1.5) -> CursorSprite:
        h, w = fill_a.shape[:2]
        r = _SPRITE_RES
        rad = max(1, int(round(outline_pts * r)))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        outline_a = cv2.dilate(fill_a, disc)

        shift = np.float32([[1, 0, 1.0 * r], [0, 1, 2.0 * r]])
        shadow_a = cv2.warpAffine(outline_a, shift, (w, h))
        shadow_a = cv2.GaussianBlur(shadow_a, (0, 0), sigmaX=1.4 * r) * 0.35

        dark = np.array(hex_to_bgr("#111113"), dtype=np.float32)
        white = np.array((255.0, 255.0, 255.0), dtype=np.float32)
        black = np.zeros(3, dtype=np.float32)

        rgb = np.zeros((h, w, 3), dtype=np.float32)
        a = np.zeros((h, w), dtype=np.float32)
        for layer_a, color in ((shadow_a, black), (outline_a, white), (fill_a, dark)):
            la = layer_a[..., None]
            rgb = color * la + rgb * (1.0 - la)
            a = layer_a + a * (1.0 - layer_a)
        return cls(rgb=rgb.astype(np.float32), alpha=a[..., None].astype(np.float32), hotspot=hotspot)

    @classmethod
    def build(cls, kind: CursorType = "arrow") -> CursorSprite:
        r, m = _SPRITE_RES, _SPRITE_MARGIN
        if kind == "pointer":
            w = int((18.0 + 2 * m) * r)
            h = int((21.0 + 2 * m) * r)
            pts = [((x + m) * r, (y + m) * r) for x, y in _POINTER]
            fill = Image.new("L", (w, h), 0)
            ImageDraw.Draw(fill).polygon(pts, fill=255)
            fill_a = np.asarray(fill, dtype=np.float32) / 255.0
            return cls._from_fill(fill_a, hotspot=((6.2 + m) * r, (0.6 + m) * r), outline_pts=1.35)
        if kind == "text":
            w = int((10.0 + 2 * m) * r)
            h = int((20.0 + 2 * m) * r)
            fill = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(fill)
            cx = (5.0 + m) * r
            top = (1.5 + m) * r
            bot = (18.5 + m) * r
            stem = 0.85 * r
            cap_w = 3.2 * r
            cap_h = 1.2 * r
            # Vertical stem + top/bottom I-beam serifs
            d.rectangle((cx - stem, top, cx + stem, bot), fill=255)
            d.rectangle((cx - cap_w, top, cx + cap_w, top + cap_h), fill=255)
            d.rectangle((cx - cap_w, bot - cap_h, cx + cap_w, bot), fill=255)
            fill_a = np.asarray(fill, dtype=np.float32) / 255.0
            return cls._from_fill(fill_a, hotspot=(cx, (top + bot) / 2.0), outline_pts=1.15)
        # Default: arrow
        w = int((13.0 + 2 * m) * r)
        h = int((20.0 + 2 * m) * r)
        pts = [((x + m) * r, (y + m) * r) for x, y in _ARROW]
        fill = Image.new("L", (w, h), 0)
        ImageDraw.Draw(fill).polygon(pts, fill=255)
        fill_a = np.asarray(fill, dtype=np.float32) / 255.0
        return cls._from_fill(fill_a, hotspot=(m * r, m * r), outline_pts=1.6)


class _Spring:
    """Critically damped (by default) 1-D spring, semi-implicit Euler."""

    __slots__ = ("pos", "vel", "omega", "zeta")

    def __init__(self, pos: float, omega: float, zeta: float) -> None:
        self.pos, self.vel, self.omega, self.zeta = pos, 0.0, omega, zeta

    def step(self, target: float, dt: float, substeps: int) -> float:
        h = dt / substeps
        w = self.omega
        for _ in range(substeps):
            acc = (w * w) * (target - self.pos) - 2.0 * self.zeta * w * self.vel
            self.vel += acc * h
            self.pos += self.vel * h
        return self.pos


class _Telemetry:
    """Fast interpolation over the 60 Hz telemetry log."""

    def __init__(self, frames: list[TelemetryFrame]) -> None:
        if not frames:
            raise ValueError("recording has no telemetry")
        self.frames = frames
        self.t = np.array([f.t for f in frames], dtype=np.float64)
        self.x = np.array([f.mouse_x for f in frames], dtype=np.float64)
        self.y = np.array([f.mouse_y for f in frames], dtype=np.float64)

    def at(self, t: float) -> tuple[float, float, TelemetryFrame]:
        ts = self.t
        if t <= ts[0]:
            return float(self.x[0]), float(self.y[0]), self.frames[0]
        if t >= ts[-1]:
            return float(self.x[-1]), float(self.y[-1]), self.frames[-1]
        i = int(np.searchsorted(ts, t, side="right")) - 1
        span = ts[i + 1] - ts[i]
        u = 0.0 if span <= 0 else (t - ts[i]) / span
        x = self.x[i] + (self.x[i + 1] - self.x[i]) * u
        y = self.y[i] + (self.y[i + 1] - self.y[i]) * u
        return float(x), float(y), self.frames[i]


@dataclass(slots=True)
class _CameraSample:
    index: int
    t: float
    cam_x: float
    cam_y: float
    zoom: float
    vx: float
    vy: float
    vz: float
    settled: bool
    focus_box: Box | None
    camera_mode: str


def resolve_canvas_size(
    viewport: tuple[int, int],
    theme: StudioTheme,
    chrome_h_base: int,
) -> tuple[int, int]:
    """Resolve final even canvas ``(cw, ch)`` from aspect, resolution, and quality preset."""
    if theme.resolution != "native":
        aspect: AspectRatio = theme.aspect or "16:9"
        cw, ch = _RESOLUTION_TARGETS[theme.resolution][aspect]
        if theme.quality == "draft":
            cw, ch = _even(cw * 0.67), _even(ch * 0.67)
        return _even(cw), _even(ch)
    if theme.aspect is not None:
        cw, ch = _NATIVE_ASPECTS[theme.aspect]
        if theme.quality == "draft":
            cw, ch = _even(cw * 0.75), _even(ch * 0.75)
        return _even(cw), _even(ch)
    vw, vh = viewport
    pad = theme.padding_px
    cw = _even(vw + 2 * pad)
    ch = _even(vh + chrome_h_base + 2 * pad)
    if theme.quality == "draft":
        cw, ch = _even(cw * 0.75), _even(ch * 0.75)
    return cw, ch


def _display_url(raw: str | None) -> str:
    if not raw:
        return "app.local"
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        path = parsed.path.rstrip("/")
        return f"{parsed.netloc}{path}" if path else parsed.netloc
    if parsed.scheme == "file":
        return parsed.path.rsplit("/", 1)[-1] or "demo.html"
    return raw[:64]


class CinemaEngine:
    def __init__(
        self,
        recording: Recording,
        theme: StudioTheme | None = None,
        settings: RenderSettings | None = None,
    ) -> None:
        self.rec = recording
        self.theme = theme or StudioTheme()
        self.settings = settings or RenderSettings()
        self.vw, self.vh = recording.viewport  # logical CSS px
        self.fps = self.theme.output_fps
        self.tel = _Telemetry(recording.telemetry)
        self.frame_ts = [f.t for f in recording.frames]

        has_chrome = self.theme.window_chrome != "none" or self.theme.show_url_bar
        base_chrome_h = 38 if has_chrome else 0
        self.cw, self.ch = resolve_canvas_size((self.vw, self.vh), self.theme, base_chrome_h)

        # Scale chassis proportions gracefully to the target canvas.
        self.ui_scale = min(self.cw / 1392.0, self.ch / 912.0)
        self.pad_x = max(0, int(round(self.theme.padding_px * self.ui_scale)))
        self.pad_y = max(0, int(round(self.theme.padding_px * self.ui_scale)))
        # Ensure window stays non-empty even on small or narrow canvases.
        self.pad_x = min(self.pad_x, max(0, (self.cw - 120) // 4))
        self.pad_y = min(self.pad_y, max(0, (self.ch - 120) // 4))
        self.chrome_h = max(24, int(round(38 * self.ui_scale))) if has_chrome else 0
        self.win_w = max(2, self.cw - 2 * self.pad_x)
        self.win_h = max(2, self.ch - 2 * self.pad_y)
        self.content_w = self.win_w
        self.content_h = max(2, self.win_h - self.chrome_h)
        self.content_x = self.pad_x
        self.content_y = self.pad_y + self.chrome_h

        # Smart reframing geometry: map logical viewport (vw, vh) to (content_w, content_h) with zero distortion.
        self.base_scale = max(self.content_w / self.vw, self.content_h / self.vh)
        self.vis_vw = min(float(self.vw), self.content_w / self.base_scale)
        self.vis_vh = min(float(self.vh), self.content_h / self.base_scale)

        self.cursors: dict[CursorType, CursorSprite] = {
            "arrow": CursorSprite.build("arrow"),
            "pointer": CursorSprite.build("pointer"),
            "text": CursorSprite.build("text"),
        }
        self._build_backdrop()
        self._build_chassis()

        # Thread-safe LRU cache for decoded source frames.
        self._src_lock = threading.Lock()
        self._src_cache: OrderedDict[int, Frame] = OrderedDict()
        self._src_cache_max = 10

        self.intro_frames = (
            max(0, int(round(self.theme.intro_duration_s * self.fps)))
            if (self.theme.intro_enabled and (self.theme.intro_title or self.rec.feature_summary))
            else 0
        )
        self.outro_frames = (
            max(0, int(round(self.theme.outro_duration_s * self.fps)))
            if (self.theme.outro_enabled and (self.theme.outro_title or self.theme.brand_name or self.theme.outro_cta))
            else 0
        )
        self.rec_frame_count = max(1, math.ceil(max(self.rec.duration, 1.0 / self.fps) * self.fps))

        self.last_frame: Frame | None = None
        self.best_frame: Frame | None = None
        self._best_score: float = -1.0
        self.key_screenshots: list[tuple[int, str, Frame]] = []
        self._camera_plan: list[_CameraSample] = self._plan_camera()

    # ------------------------------------------------------------------ sizes
    @property
    def frame_size(self) -> tuple[int, int]:
        return self.cw, self.ch

    @property
    def frame_count(self) -> int:
        return self.intro_frames + self.rec_frame_count + self.outro_frames

    # ------------------------------------------------------------------ backdrop & chassis
    def _build_backdrop(self) -> None:
        th = self.theme
        cw, ch = self.cw, self.ch

        if th.background_style == "wallpaper" and th.wallpaper_path:
            data = np.fromfile(th.wallpaper_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                ih, iw = img.shape[:2]
                scale = max(cw / iw, ch / ih)
                nw, nh = max(cw, int(round(iw * scale))), max(ch, int(round(ih * scale)))
                resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                x0, y0 = (nw - cw) // 2, (nh - ch) // 2
                crop = resized[y0 : y0 + ch, x0 : x0 + cw]
                if th.wallpaper_blur > 0:
                    crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=max(1.0, th.wallpaper_blur * self.ui_scale))
                # Subtle studio vignette over wallpaper
                yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
                d = np.sqrt(((xx - cw / 2) / (cw * 0.65)) ** 2 + ((yy - ch / 2) / (ch * 0.65)) ** 2)
                vig = np.clip(1.0 - 0.22 * d, 0.72, 1.0)[..., None]
                self.backdrop: Frame = np.clip(crop.astype(np.float32) * vig, 0, 255).astype(np.uint8)
                return

        bg1 = np.array(hex_to_bgr(th.background), dtype=np.float32)
        if th.background_style == "solid":
            self.backdrop = np.full((ch, cw, 3), bg1, dtype=np.uint8)
            return

        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        if th.background_style == "gradient" and th.background_end:
            bg2 = np.array(hex_to_bgr(th.background_end), dtype=np.float32)
            rad = math.radians(th.background_angle)
            nx = (xx / max(cw - 1, 1)) - 0.5
            ny = (yy / max(ch - 1, 1)) - 0.5
            proj = nx * math.cos(rad) + ny * math.sin(rad) + 0.5
            t_g = np.clip(proj, 0.0, 1.0)[..., None]
            base = bg1 * (1.0 - t_g) + bg2 * t_g
            # Soft radial studio highlight for depth
            d = np.sqrt(((xx - cw * 0.5) / (cw * 0.65)) ** 2 + ((yy - ch * 0.35) / (ch * 0.65)) ** 2)
            lift = np.clip(1.0 - d, 0.0, 1.0)[..., None] * 14.0
            self.backdrop = np.clip(base + lift + 0.5, 0, 255).astype(np.uint8)
            return

        # Default "studio": clean dark/light backdrop with a subtle studio center light.
        d = np.sqrt(((xx - cw / 2) / (cw * 0.6)) ** 2 + ((yy - ch / 2) / (ch * 0.6)) ** 2)
        lift = np.clip(1.0 - d, 0.0, 1.0)[..., None] * 9.0
        self.backdrop = np.clip(bg1 + lift + 0.5, 0, 255).astype(np.uint8)

    def _draw_chrome_bar(self, width: int, height: int) -> Frame:
        """Draw macOS or Windows window titlebar + optional browser URL pill."""
        th = self.theme
        light = is_light_color(th.background)
        bar_bg = (238, 238, 242) if light else (22, 22, 27)
        sep_col = (212, 212, 218) if light else (42, 42, 50)
        bar = np.full((height, width, 3), (bar_bg[2], bar_bg[1], bar_bg[0]), dtype=np.uint8)
        bar[-1, :] = (sep_col[2], sep_col[1], sep_col[0])

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        s = height / 38.0
        cy = height / 2.0

        if th.window_chrome == "macos":
            r = max(4.0, 6.0 * s)
            gap = 20.0 * s
            x_start = 18.0 * s
            for idx, col in enumerate(((255, 95, 86, 255), (255, 189, 46, 255), (39, 201, 63, 255))):
                cx = x_start + idx * gap
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
        elif th.window_chrome == "windows":
            fg = (60, 60, 70, 220) if light else (190, 190, 202, 220)
            w_btn = 34.0 * s
            # Close (x)
            cx = width - 18.0 * s
            d = 4.5 * s
            lw = max(1, int(round(1.5 * s)))
            draw.line((cx - d, cy - d, cx + d, cy + d), fill=fg, width=lw)
            draw.line((cx - d, cy + d, cx + d, cy - d), fill=fg, width=lw)
            # Maximize (square)
            mx = cx - w_btn
            draw.rectangle((mx - d, cy - d, mx + d, cy + d), outline=fg, width=lw)
            # Minimize (line)
            nx = mx - w_btn
            draw.line((nx - d, cy + 1, nx + d, cy + 1), fill=fg, width=lw)

        if th.show_url_bar:
            url_text = _display_url(th.browser_url or self.rec.url)
            font = load_font(max(11, int(round(12.5 * s))), weight=500, custom_path=th.font_path)
            pill_w = min(int(width * 0.52), max(int(220 * s), int(len(url_text) * 8.2 * s + 48 * s)))
            pill_h = max(18, int(round(24 * s)))
            px0 = (width - pill_w) // 2 if th.window_chrome != "windows" else max(16, int(18 * s))
            py0 = (height - pill_h) // 2
            pill_bg = (226, 226, 232, 255) if light else (33, 33, 41, 255)
            pill_border = (204, 204, 212, 255) if light else (54, 54, 66, 255)
            text_fg = (72, 72, 82, 255) if light else (178, 178, 192, 255)
            draw.rounded_rectangle(
                (px0, py0, px0 + pill_w, py0 + pill_h),
                radius=max(6, int(round(7 * s))),
                fill=pill_bg,
                outline=pill_border,
                width=1,
            )
            # Small lock dot + URL string
            bbox = draw.textbbox((0, 0), url_text, font=font)
            tw, th_t = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = px0 + (pill_w - tw) // 2
            ty = py0 + (pill_h - th_t) // 2 - 1
            draw.text((tx, ty), url_text, font=font, fill=text_fg)

        arr = np.asarray(overlay, dtype=np.float32)
        a = arr[..., 3:4] / 255.0
        bgr = arr[..., :3][..., ::-1]
        out = bar.astype(np.float32) * (1.0 - a) + bgr * a
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def _build_chassis(self) -> None:
        th = self.theme
        cw, ch = self.cw, self.ch
        px, py = self.pad_x, self.pad_y
        ww, wh = self.win_w, self.win_h
        radius = min(th.radius_px * self.ui_scale, min(ww, wh) / 4.0) if (px > 0 or py > 0) else 0.0
        border_px = max(0, int(round(th.border_px * self.ui_scale)))

        base = self.backdrop.astype(np.float32)
        win = (float(px), float(py), float(px + ww), float(py + wh))

        if th.shadow_alpha > 0 and th.shadow_blur > 0 and (px > 0 or py > 0):
            off_y = th.shadow_offset_y * self.ui_scale
            shadow = _rounded_rect_alpha((cw, ch), (win[0], win[1] + off_y, win[2], win[3] + off_y), radius)
            sigma = max(1.0, (th.shadow_blur * self.ui_scale) / 2.0)
            shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=sigma) * th.shadow_alpha
            base = base * (1.0 - shadow[..., None])
        self.base: Frame = np.clip(base + 0.5, 0, 255).astype(np.uint8)

        self.chrome_bar: Frame | None = (
            self._draw_chrome_bar(self.win_w, self.chrome_h) if self.chrome_h > 0 else None
        )

        win_a = _rounded_rect_alpha((cw, ch), win, radius)
        inner = (win[0] + border_px, win[1] + border_px, win[2] - border_px, win[3] - border_px)
        inner_a = _rounded_rect_alpha((cw, ch), inner, max(0.0, radius - border_px))
        border_a = np.clip(win_a - inner_a, 0.0, 1.0)
        self.border_color = np.array(hex_to_bgr(th.border), dtype=np.float32)

        e = int(math.ceil(radius)) + border_px + 3
        # Precompute (sl, win_alpha_eff, premult_bg_and_border) so per-frame blending is one fused multiply-add.
        self.strips: list[tuple[tuple[slice, slice], FloatImg, FloatImg]] = []
        if radius > 0.5 or border_px > 0:
            y_top0, y_top1 = max(0, py - 2), min(ch, py + e)
            y_bot0, y_bot1 = max(0, py + wh - e), min(ch, py + wh + 2)
            x_l0, x_l1 = max(0, px - 2), min(cw, px + e)
            x_r0, x_r1 = max(0, px + ww - e), min(cw, px + ww + 2)
            x_all = slice(max(0, px - 2), min(cw, px + ww + 2))
            y_mid = slice(y_top1, max(y_top1, y_bot0))
            regions = [
                (slice(y_top0, y_top1), x_all),
                (slice(y_bot0, y_bot1), x_all),
                (y_mid, slice(x_l0, x_l1)),
                (y_mid, slice(x_r0, x_r1)),
            ]
            for sl in regions:
                if sl[0].stop > sl[0].start and sl[1].stop > sl[1].start:
                    wa = win_a[sl][..., None]
                    ba = border_a[sl][..., None]
                    content_mul = (wa * (1.0 - ba)).astype(np.float32)
                    bg_term = (
                        self.base[sl].astype(np.float32) * (1.0 - wa) * (1.0 - ba)
                        + self.border_color * ba
                        + 0.5
                    ).astype(np.float32)
                    self.strips.append((sl, content_mul, bg_term))

        self.inner_rect = (self.content_x, self.content_y, self.content_x + self.content_w, self.content_y + self.content_h)

    # ------------------------------------------------------------------ sources
    def _source(self, t: float) -> tuple[int, Frame]:
        i = max(0, bisect_right(self.frame_ts, t) - 1)
        with self._src_lock:
            cached = self._src_cache.get(i)
            if cached is not None:
                self._src_cache.move_to_end(i)
                return i, cached
        path = self.rec.frames[i].path
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not decode source frame {path}")
        with self._src_lock:
            self._src_cache[i] = img
            self._src_cache.move_to_end(i)
            while len(self._src_cache) > self._src_cache_max:
                self._src_cache.popitem(last=False)
        return i, img

    def _crop_window(self, src: Frame, cx: float, cy: float, zoom: float, settled: bool) -> Frame:
        """Extract camera window centred on ``(cx, cy)`` in CSS px and render to ``(content_w, content_h)``."""
        cw_out, ch_out = self.content_w, self.content_h
        sh, sw = src.shape[:2]
        src_per_css_x = sw / self.vw
        src_per_css_y = sh / self.vh
        # Visible logical CSS dimensions at this zoom
        vis_w = self.vis_vw / zoom
        vis_h = self.vis_vh / zoom
        left = min(max(cx - vis_w / 2.0, 0.0), max(0.0, self.vw - vis_w))
        top = min(max(cy - vis_h / 2.0, 0.0), max(0.0, self.vh - vis_h))

        # Source ROI in source pixels (only resize the cropped ROI -> much faster than resizing the full 2560x1600!)
        sx0 = left * src_per_css_x
        sy0 = top * src_per_css_y
        s_w = vis_w * src_per_css_x
        s_h = vis_h * src_per_css_y

        if settled:
            ix0 = max(0, min(sw - 1, int(round(sx0))))
            iy0 = max(0, min(sh - 1, int(round(sy0))))
            ix1 = max(ix0 + 1, min(sw, int(round(sx0 + s_w))))
            iy1 = max(iy0 + 1, min(sh, int(round(sy0 + s_h))))
            roi = src[iy0:iy1, ix0:ix1]
            if roi.shape[1] == cw_out and roi.shape[0] == ch_out:
                return roi
            interp = cv2.INTER_AREA if (cw_out < roi.shape[1] and ch_out < roi.shape[0]) else cv2.INTER_CUBIC
            return cv2.resize(roi, (cw_out, ch_out), interpolation=interp)

        # Slice tight source ROI first so warpAffine never touches pixels outside the crop.
        pad_px = 2
        ix0 = max(0, int(math.floor(sx0)) - pad_px)
        iy0 = max(0, int(math.floor(sy0)) - pad_px)
        ix1 = min(sw, int(math.ceil(sx0 + s_w)) + pad_px)
        iy1 = min(sh, int(math.ceil(sy0 + s_h)) + pad_px)
        roi = src[iy0:iy1, ix0:ix1]
        scale_x = cw_out / max(s_w, 1e-3)
        scale_y = ch_out / max(s_h, 1e-3)
        m = np.float32([[scale_x, 0.0, -(sx0 - ix0) * scale_x], [0.0, scale_y, -(sy0 - iy0) * scale_y]])
        flags = cv2.INTER_AREA if (scale_x < 0.95 and scale_y < 0.95) else cv2.INTER_LINEAR
        return cv2.warpAffine(roi, m, (cw_out, ch_out), flags=flags, borderMode=cv2.BORDER_REPLICATE)

    def _content_with_blur(self, src: Frame, sample: _CameraSample) -> Frame:
        """Render camera content with speed-proportional temporal motion blur when panning/zooming fast."""
        st = self.settings
        speed = math.hypot(sample.vx, sample.vy) * sample.zoom * self.base_scale + abs(sample.vz) * 360.0
        if st.motion_blur <= 0.0 or st.motion_blur_samples <= 1 or sample.settled or speed < 110.0:
            return self._crop_window(src, sample.cam_x, sample.cam_y, sample.zoom, sample.settled)

        n = max(2, min(5, st.motion_blur_samples))
        shutter_s = min(0.85 / self.fps, (speed / 1800.0) * st.motion_blur / self.fps)
        offsets = np.linspace(-0.5 * shutter_s, 0.5 * shutter_s, n)
        if n == 3:
            weights = (0.25, 0.50, 0.25)
        else:
            weights = tuple(1.0 / n for _ in range(n))

        acc = np.zeros((self.content_h, self.content_w, 3), dtype=np.float32)
        for off, w in zip(offsets, weights):
            z = min(2.0, max(1.0, sample.zoom + sample.vz * off))
            cx, cy = self._clamp_center(sample.cam_x + sample.vx * off, sample.cam_y + sample.vy * off, z)
            sub = self._crop_window(src, cx, cy, z, False)
            acc += sub.astype(np.float32) * w
        return np.clip(acc + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ camera
    def _clamp_center(self, cx: float, cy: float, zoom: float) -> tuple[float, float]:
        hw = min(self.vw / 2.0, self.vis_vw / (2.0 * zoom))
        hh = min(self.vh / 2.0, self.vis_vh / (2.0 * zoom))
        return min(max(cx, hw), self.vw - hw), min(max(cy, hh), self.vh - hh)

    def _raw_target(self, t: float, prev: tuple[float, float]) -> tuple[float, float, float, Box | None, str]:
        mx, my, tf = self.tel.at(t + self.settings.lookahead_s)
        zoom = min(2.0, max(1.0, tf.target_zoom))
        pad_f = self.settings.focus_padding_px
        if tf.focus_box is not None:
            bx, by, bw, bh = tf.focus_box
            # Generous padding so the focused element is never cut off at the frame edge.
            padded_w = bw + 2.0 * pad_f
            padded_h = bh + 2.0 * pad_f
            fit = min(0.88 * self.vis_vw / max(padded_w, 1.0), 0.88 * self.vis_vh / max(padded_h, 1.0))
            zoom = max(1.0, min(zoom, fit))
            pull = 0.18 if bw * bh < 0.08 * self.vw * self.vh else 0.0
            cx = (1.0 - pull) * (bx + bw / 2.0) + pull * mx
            cy = (1.0 - pull) * (by + bh / 2.0) + pull * my
        elif tf.camera_mode == "pan" and zoom > 1.0:
            cx, cy = prev
            dzx = self.settings.pan_deadzone * self.vis_vw / zoom
            dzy = self.settings.pan_deadzone * self.vis_vh / zoom
            if mx > cx + dzx:
                cx = mx - dzx
            elif mx < cx - dzx:
                cx = mx + dzx
            if my > cy + dzy:
                cy = my - dzy
            elif my < cy - dzy:
                cy = my + dzy
        else:
            # In cropped aspect ratios (1:1, 9:16, 4:5), even wide mode smoothly follows cursor/focus horizontally.
            if self.vis_vw < self.vw - 4.0 or self.vis_vh < self.vh - 4.0:
                cx, cy = mx, my
            else:
                cx, cy = self.vw / 2.0, self.vh / 2.0
        cx, cy = self._clamp_center(cx, cy, zoom)
        return cx, cy, zoom, tf.focus_box, tf.camera_mode

    def _plan_camera(self) -> list[_CameraSample]:
        """Pre-compute the spring camera path for all recording frames with smooth focus transitions."""
        st = self.settings
        dt = 1.0 / self.fps
        t0 = self.rec.t_start
        n = self.rec_frame_count

        # Pass 1: collect raw targets
        raw_x = np.empty(n, dtype=np.float64)
        raw_y = np.empty(n, dtype=np.float64)
        raw_z = np.empty(n, dtype=np.float64)
        fboxes: list[Box | None] = []
        modes: list[str] = []
        prev = (self.vw / 2.0, self.vh / 2.0)
        for i in range(n):
            t = t0 + i * dt
            tx, ty, tz, fb, mode = self._raw_target(t, prev)
            raw_x[i], raw_y[i], raw_z[i] = tx, ty, tz
            fboxes.append(fb)
            modes.append(mode)
            prev = (tx, ty)

        # Pass 2: smooth target jumps between distinct focus boxes + zoom-dip on long pans
        win = max(1, int(round(st.focus_transition_s * self.fps)))
        if win > 1 and n > win:
            kernel = np.hanning(2 * win + 1)
            kernel /= kernel.sum()
            sm_x = np.convolve(np.pad(raw_x, win, mode="edge"), kernel, mode="valid")
            sm_y = np.convolve(np.pad(raw_y, win, mode="edge"), kernel, mode="valid")
            sm_z = np.convolve(np.pad(raw_z, win, mode="edge"), kernel, mode="valid")
            # Gentle zoom pullback when target pans quickly over a long distance.
            pan_speed = np.hypot(np.gradient(sm_x, dt), np.gradient(sm_y, dt))
            dip = np.clip((pan_speed - 380.0) / 1800.0, 0.0, 0.22)
            sm_z = np.maximum(1.0, sm_z * (1.0 - dip))
        else:
            sm_x, sm_y, sm_z = raw_x, raw_y, raw_z

        # Pass 3: integrate critically damped springs
        sx = _Spring(float(sm_x[0]), st.omega, st.zeta)
        sy = _Spring(float(sm_y[0]), st.omega, st.zeta)
        sz = _Spring(float(sm_z[0]), st.omega, st.zeta)

        samples: list[_CameraSample] = []
        for i in range(n):
            t = t0 + i * dt
            if i > 0:
                sx.step(float(sm_x[i]), dt, st.substeps)
                sy.step(float(sm_y[i]), dt, st.substeps)
                sz.step(float(sm_z[i]), dt, st.substeps)
            zoom = min(2.0, max(1.0, sz.pos))
            cam_x, cam_y = self._clamp_center(sx.pos, sy.pos, zoom)
            settled = abs(sx.vel) * zoom < 0.4 and abs(sy.vel) * zoom < 0.4 and abs(sz.vel) < 1e-3
            samples.append(_CameraSample(i, t, cam_x, cam_y, zoom, sx.vel, sy.vel, sz.vel, settled, fboxes[i], modes[i]))
        return samples

    # ------------------------------------------------------------------ overlays
    def _press_amount(self, t: float) -> float:
        best = 0.0
        for c in self.rec.clicks:
            age = t - c.t
            if -0.03 <= age < 0.0:
                v = 1.0 - (-age / 0.03)
            elif 0.0 <= age <= 0.09:
                v = 1.0
            elif 0.09 < age <= 0.19:
                v = 1.0 - (age - 0.09) / 0.10
            else:
                continue
            best = max(best, v * v * (3 - 2 * v))
        return best

    def _draw_ripples(self, canvas: Frame, t: float, to_canvas: _Mapper, zoom: float) -> None:
        th = self.theme
        dur = th.ripple_ms / 1000.0
        color = tuple(int(c) for c in reversed(th.ripple_color))  # RGB -> BGR
        accent_bgr = hex_to_bgr(th.accent)
        x0, y0, x1, y1 = self.inner_rect
        zf = (0.65 + 0.35 * zoom) * self.ui_scale
        for c in self.rec.clicks:
            age = t - c.t
            if not 0.0 <= age <= dur:
                continue
            u = age / dur
            e = ease_out_cubic(u)
            radius = (6.0 + 30.0 * e) * zf
            inner_r = (3.0 + 16.0 * e) * zf
            alpha_outer = 0.60 * (1.0 - u)
            alpha_fill = 0.18 * (1.0 - u) * (1.0 - u)
            px, py = to_canvas(c.x, c.y)
            pad = int(radius + 6)
            rx0, ry0 = max(x0, int(px) - pad), max(y0, int(py) - pad)
            rx1, ry1 = min(x1, int(px) + pad + 1), min(y1, int(py) + pad + 1)
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            roi = canvas[ry0:ry1, rx0:rx1]
            shift = 4
            center = (int(round((px - rx0) * (1 << shift))), int(round((py - ry0) * (1 << shift))))
            if alpha_fill > 0.01:
                fill_ov = roi.copy()
                cv2.circle(fill_ov, center, int(round(inner_r * (1 << shift))), accent_bgr, -1, cv2.LINE_AA, shift)
                cv2.addWeighted(fill_ov, alpha_fill, roi, 1.0 - alpha_fill, 0.0, dst=roi)
            ring_ov = roi.copy()
            thickness = max(2, int(round(2.0 * self.ui_scale)))
            cv2.circle(ring_ov, center, int(round(radius * (1 << shift))), color, thickness, cv2.LINE_AA, shift)
            cv2.addWeighted(ring_ov, alpha_outer, roi, 1.0 - alpha_outer, 0.0, dst=roi)

    def _draw_cursor(
        self,
        canvas: Frame,
        px: float,
        py: float,
        scale: float,
        kind: CursorType = "arrow",
        alpha_mul: float = 1.0,
    ) -> None:
        if alpha_mul <= 1e-3:
            return
        sp = self.cursors.get(kind, self.cursors["arrow"])
        f = scale / _SPRITE_RES
        h0, w0 = sp.alpha.shape[:2]
        w, h = max(2, round(w0 * f)), max(2, round(h0 * f))
        fx, fy = w / w0, h / h0
        rgb = cv2.resize(sp.rgb, (w, h), interpolation=cv2.INTER_AREA)
        a = cv2.resize(sp.alpha, (w, h), interpolation=cv2.INTER_AREA)
        hx, hy = sp.hotspot[0] * fx, sp.hotspot[1] * fy
        ox, oy = px - hx, py - hy
        ix, iy = math.floor(ox), math.floor(oy)
        sx, sy = ox - ix, oy - iy
        m = np.float32([[1, 0, sx], [0, 1, sy]])
        rgb = cv2.warpAffine(rgb, m, (w + 1, h + 1), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(a, m, (w + 1, h + 1), flags=cv2.INTER_LINEAR)
        if a.ndim == 2:
            a = a[..., None]
        if alpha_mul < 0.999:
            rgb = rgb * alpha_mul
            a = a * alpha_mul
        x0, y0, x1, y1 = self.inner_rect
        cx0, cy0 = max(ix, x0), max(iy, y0)
        cx1, cy1 = min(ix + w + 1, x1), min(iy + h + 1, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sl_s = (slice(cy0 - iy, cy1 - iy), slice(cx0 - ix, cx1 - ix))
        roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
        out = roi * (1.0 - a[sl_s]) + rgb[sl_s]
        canvas[cy0:cy1, cx0:cx1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def _draw_active_caption(self, canvas: Frame, t: float) -> None:
        if not self.theme.show_captions or not self.rec.captions:
            return
        for cap in self.rec.captions:
            if cap.t_start <= t <= cap.t_end and cap.text.strip():
                dur = max(0.2, cap.t_end - cap.t_start)
                fade = min(0.24, dur * 0.25)
                age = t - cap.t_start
                rem = cap.t_end - t
                a_in = ease_out_cubic(age / fade) if age < fade else 1.0
                a_out = ease_out_cubic(rem / fade) if rem < fade else 1.0
                draw_caption(canvas, cap.text, min(a_in, a_out), a_in, self.theme)
                break

    # ------------------------------------------------------------------ per-frame renderer
    def _render_recording_frame(self, sample: _CameraSample) -> tuple[Frame, float, tuple[int, str, Frame] | None]:
        th = self.theme
        t = sample.t
        _idx, src = self._source(t)
        content = self._content_with_blur(src, sample)

        canvas = self.base.copy()
        if self.chrome_bar is not None:
            canvas[self.pad_y : self.pad_y + self.chrome_h, self.pad_x : self.pad_x + self.win_w] = self.chrome_bar
        canvas[self.content_y : self.content_y + self.content_h, self.content_x : self.content_x + self.content_w] = content

        for sl, content_mul, bg_term in self.strips:
            region = canvas[sl].astype(np.float32) * content_mul + bg_term
            np.clip(region, 0, 255, out=region)
            canvas[sl] = region.astype(np.uint8)

        mapper = _Mapper(
            sample.cam_x, sample.cam_y, sample.zoom,
            self.vis_vw, self.vis_vh, self.vw, self.vh,
            self.content_x, self.content_y, self.content_w, self.content_h,
        )
        self._draw_ripples(canvas, t, mapper, sample.zoom)

        mx, my, tf = self.tel.at(t)
        px, py = mapper(mx, my)
        press = self._press_amount(t)
        press_scale = 1.0 - (1.0 - th.click_press_scale) * press
        cursor_scale = 1.25 * (0.65 + 0.35 * sample.zoom) * th.cursor_scale * press_scale * self.ui_scale

        # Subtle motion trail when the cursor sweeps fast across the screen.
        if th.cursor_trail:
            prev_mx, prev_my, _ = self.tel.at(t - 1.0 / self.fps)
            ppx, ppy = mapper(prev_mx, prev_my)
            speed_c = math.hypot(px - ppx, py - ppy) * self.fps
            if speed_c > 420.0 * self.ui_scale:
                for step_back, trail_alpha in ((2, 0.08), (1, 0.16)):
                    tmx, tmy, _ = self.tel.at(t - step_back / self.fps)
                    tpx, tpy = mapper(tmx, tmy)
                    self._draw_cursor(canvas, tpx, tpy, cursor_scale * 0.96, tf.cursor_type, alpha_mul=trail_alpha)

        self._draw_cursor(canvas, px, py, cursor_scale, tf.cursor_type)

        # Overlays (captions + keystroke HUD)
        self._draw_active_caption(canvas, t)
        if th.show_keystrokes and self.rec.keys:
            draw_keystrokes(canvas, self.rec.keys, t, th)

        # Money-shot score: prefer settled focused/show moments in the middle-to-late story.
        rel_t = (t - self.rec.t_start) / max(self.rec.duration, 1e-3)
        score = 0.0
        if sample.settled:
            score += 2.0
        if sample.focus_box is not None:
            score += 3.0 + (sample.zoom - 1.0) * 2.0
        if 0.25 <= rel_t <= 0.88:
            score += 1.5 + rel_t
        if tf.is_clicking:
            score -= 0.5  # avoid mid-click frames as poster

        # Key screenshot capture (one per meaningful step right near its settled end)
        shot: tuple[int, str, Frame] | None = None
        if th.screenshots and self.rec.captions:
            for cap in self.rec.captions:
                if cap.action in ("click", "fill", "show"):
                    target_t = cap.t_start + 0.72 * max(0.1, cap.t_end - cap.t_start)
                    if abs(t - target_t) <= (0.55 / self.fps):
                        cbox = None
                        fb = cap.focus_box or sample.focus_box
                        if fb is not None:
                            bx, by, bw, bh = fb
                            x0, y0 = mapper(bx, by)
                            x1, y1 = mapper(bx + bw, by + bh)
                            cbox = (x0, y0, max(4.0, x1 - x0), max(4.0, y1 - y0))
                        annotated = annotate_screenshot(canvas, cbox, cap.step_index + 1, cap.text, th)
                        shot = (cap.step_index + 1, cap.text, annotated)
                        break

        return canvas, score, shot

    # ------------------------------------------------------------------ main loop
    def frames(self) -> Iterator[Frame]:
        th, st = self.theme, self.settings
        n_rec = self.rec_frame_count
        self.key_screenshots = []
        seen_shot_steps: set[int] = set()

        # Pre-render first recording frame if intro cross-fade needs it
        first_rec_frame: Frame | None = None
        if self.intro_frames > 0:
            first_rec_frame, _, _ = self._render_recording_frame(self._camera_plan[0])
            intro_title = th.intro_title or self.rec.feature_summary or "Product Demo"
            fade_n = max(1, min(self.intro_frames // 3, int(round(0.35 * self.fps))))
            for i in range(self.intro_frames):
                prog = min(1.0, (i + 1) / max(1, self.intro_frames * 0.45))
                card = render_title_card(
                    self.backdrop,
                    title=intro_title,
                    subtitle=th.intro_subtitle,
                    theme=th,
                    progress=prog,
                    is_outro=False,
                )
                rem = self.intro_frames - 1 - i
                if rem < fade_n:
                    u = ease_in_out_cubic(1.0 - rem / fade_n)
                    card = cv2.addWeighted(first_rec_frame, u, card, 1.0 - u, 0.0)
                self.last_frame = card
                yield card

        # Parallel rendering of recording frames via ThreadPoolExecutor (OpenCV/NumPy release GIL).
        workers = st.workers if st.workers > 0 else min(4, max(1, (os.cpu_count() or 2)))
        if workers <= 1 or n_rec < 8:
            for i, sample in enumerate(self._camera_plan):
                if i == 0 and first_rec_frame is not None:
                    frame, score, shot = first_rec_frame, 1.0, None
                else:
                    frame, score, shot = self._render_recording_frame(sample)
                if score > self._best_score:
                    self._best_score = score
                    self.best_frame = frame
                if shot is not None and shot[0] not in seen_shot_steps:
                    seen_shot_steps.add(shot[0])
                    self.key_screenshots.append(shot)
                self.last_frame = frame
                yield frame
        else:
            max_in_flight = workers * 3
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending: OrderedDict[int, Future[tuple[Frame, float, tuple[int, str, Frame] | None]]] = OrderedDict()
                submit_idx = 0

                def _fill() -> None:
                    nonlocal submit_idx
                    while submit_idx < n_rec and len(pending) < max_in_flight:
                        if submit_idx == 0 and first_rec_frame is not None:
                            f: Future[tuple[Frame, float, tuple[int, str, Frame] | None]] = Future()
                            f.set_result((first_rec_frame, 1.0, None))
                            pending[submit_idx] = f
                        else:
                            pending[submit_idx] = pool.submit(self._render_recording_frame, self._camera_plan[submit_idx])
                        submit_idx += 1

                _fill()
                while pending:
                    _idx, fut = pending.popitem(last=False)
                    _fill()
                    frame, score, shot = fut.result()
                    if score > self._best_score:
                        self._best_score = score
                        self.best_frame = frame
                    if shot is not None and shot[0] not in seen_shot_steps:
                        seen_shot_steps.add(shot[0])
                        self.key_screenshots.append(shot)
                    self.last_frame = frame
                    yield frame

        # Outro card
        if self.outro_frames > 0 and self.last_frame is not None:
            tail_frame = self.last_frame
            outro_title = th.outro_title or th.brand_name or self.rec.feature_summary or "Try it today"
            fade_n = max(1, min(self.outro_frames // 3, int(round(0.35 * self.fps))))
            for i in range(self.outro_frames):
                prog = min(1.0, (i + 1) / max(1, self.outro_frames * 0.45))
                card = render_title_card(
                    self.backdrop,
                    title=outro_title,
                    subtitle=None,
                    cta=th.outro_cta,
                    theme=th,
                    progress=prog,
                    is_outro=True,
                )
                if i < fade_n:
                    u = ease_in_out_cubic((i + 1) / fade_n)
                    card = cv2.addWeighted(card, u, tail_frame, 1.0 - u, 0.0)
                self.last_frame = card
                yield card


@dataclass(slots=True)
class _Mapper:
    """CSS px -> canvas px for the current camera and smart-reframed window."""

    cam_x: float
    cam_y: float
    zoom: float
    vis_vw: float
    vis_vh: float
    vw: int
    vh: int
    content_x: int
    content_y: int
    content_w: int
    content_h: int

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        vis_w = self.vis_vw / self.zoom
        vis_h = self.vis_vh / self.zoom
        left = min(max(self.cam_x - vis_w / 2.0, 0.0), max(0.0, self.vw - vis_w))
        top = min(max(self.cam_y - vis_h / 2.0, 0.0), max(0.0, self.vh - vis_h))
        sx = self.content_w / max(vis_w, 1e-6)
        sy = self.content_h / max(vis_h, 1e-6)
        return (x - left) * sx + self.content_x, (y - top) * sy + self.content_y
