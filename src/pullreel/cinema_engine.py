"""Turn a raw recording into studio-grade 60 fps frames.

* Camera = critically damped spring (zeta=1, omega=6.5 rad/s) on (x, y, zoom), integrated with semi-implicit Euler.
  It reads the telemetry 250 ms in the future, so zooms start before the click instead of chasing it.
* Zoom never pixelates: the source is 2560x1600 (Retina 2x) and the output window is 1280x800, so up to 2x zoom
  we only ever *downsample* (area filter) or copy 1:1.
* Headless Chromium draws no cursor, so we composite a vector-quality arrow (8x supersampled, premultiplied alpha,
  sub-pixel placement), a press squash (0.88) and a fading click ripple.
* Everything sits in a minimal dark studio chassis: #09090B backdrop, soft shadow, 16 px rounded corners, 1 px border.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from pullreel.models import Recording, RenderSettings, StudioTheme, TelemetryFrame

Frame = NDArray[np.uint8]
FloatImg = NDArray[np.float32]

# Arrow cursor polygon in 1x "points", hotspot (tip) at (0, 0).
_ARROW = [(0.0, 0.0), (0.0, 17.0), (4.2, 13.2), (7.0, 19.5), (9.6, 18.4), (6.9, 12.3), (12.3, 12.3)]
_SPRITE_RES = 8  # sprite pixels per cursor point
_SPRITE_MARGIN = 4.0  # points of room for outline + shadow


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return b, g, r


def _rounded_rect_alpha(size: tuple[int, int], rect: tuple[float, float, float, float], radius: float) -> FloatImg:
    """Anti-aliased rounded-rectangle coverage mask (4x supersampled), float32 in [0, 1]."""
    ss = 4
    w, h = size
    img = Image.new("L", (w * ss, h * ss), 0)
    x0, y0, x1, y1 = rect
    ImageDraw.Draw(img).rounded_rectangle(
        (x0 * ss, y0 * ss, x1 * ss - 1, y1 * ss - 1), radius=radius * ss, fill=255
    )
    img = img.resize((w, h), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0


@dataclass(slots=True)
class CursorSprite:
    rgb: FloatImg  # premultiplied BGR, float32 0..255
    alpha: FloatImg  # (h, w, 1) float32 0..1
    hotspot: tuple[float, float]  # sprite pixel coordinates of the tip

    @classmethod
    def build(cls) -> CursorSprite:
        r, m = _SPRITE_RES, _SPRITE_MARGIN
        w = int((13.0 + 2 * m) * r)
        h = int((20.0 + 2 * m) * r)
        pts = [((x + m) * r, (y + m) * r) for x, y in _ARROW]

        fill = Image.new("L", (w, h), 0)
        ImageDraw.Draw(fill).polygon(pts, fill=255)
        fill_a = np.asarray(fill, dtype=np.float32) / 255.0

        # White outline = fill dilated by a disc (round joins, crisp tip).
        rad = int(round(1.6 * r))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        outline_a = cv2.dilate(fill_a, disc)

        # Soft drop shadow.
        shift = np.float32([[1, 0, 1.0 * r], [0, 1, 2.0 * r]])
        shadow_a = cv2.warpAffine(outline_a, shift, (w, h))
        shadow_a = cv2.GaussianBlur(shadow_a, (0, 0), sigmaX=1.4 * r) * 0.35

        dark = np.array(_hex_to_bgr("#111113"), dtype=np.float32)
        white = np.array((255.0, 255.0, 255.0), dtype=np.float32)
        black = np.zeros(3, dtype=np.float32)

        # Porter-Duff "over", premultiplied: shadow, then white outline, then dark fill.
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        a = np.zeros((h, w), dtype=np.float32)
        for layer_a, color in ((shadow_a, black), (outline_a, white), (fill_a, dark)):
            la = layer_a[..., None]
            rgb = color * la + rgb * (1.0 - la)
            a = layer_a + a * (1.0 - layer_a)
        return cls(rgb=rgb.astype(np.float32), alpha=a[..., None].astype(np.float32), hotspot=(m * r, m * r))


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
        self.vw, self.vh = recording.viewport  # CSS px == output content px
        pad = self.theme.padding_px
        self.pad = pad
        self.cw, self.ch = self.vw + 2 * pad, self.vh + 2 * pad
        self.fps = self.theme.output_fps
        self.tel = _Telemetry(recording.telemetry)
        self.frame_ts = [f.t for f in recording.frames]
        self.cursor = CursorSprite.build()
        self._build_chassis()
        self._src_idx = -1
        self._src: Frame | None = None
        self._scaled_key: tuple[int, float] | None = None
        self._scaled: Frame | None = None
        self.last_frame: Frame | None = None

    # ------------------------------------------------------------------ sizes
    @property
    def frame_size(self) -> tuple[int, int]:
        return self.cw, self.ch

    @property
    def frame_count(self) -> int:
        return max(1, math.ceil(self.rec.duration * self.fps))

    # ------------------------------------------------------------------ chassis
    def _build_chassis(self) -> None:
        th, pad = self.theme, self.pad
        cw, ch, vw, vh = self.cw, self.ch, self.vw, self.vh
        bg = np.array(_hex_to_bgr(th.background), dtype=np.float32)

        # Backdrop: flat #09090B with a barely-there lift in the centre (reads as a studio light).
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        d = np.sqrt(((xx - cw / 2) / (cw * 0.6)) ** 2 + ((yy - ch / 2) / (ch * 0.6)) ** 2)
        lift = np.clip(1.0 - d, 0.0, 1.0)[..., None] * 7.0
        base = np.clip(bg + lift, 0, 255)

        # Soft drop shadow under the window.
        win = (pad, pad, pad + vw, pad + vh)
        shadow = _rounded_rect_alpha((cw, ch), (win[0], win[1] + th.shadow_offset_y, win[2], win[3] + th.shadow_offset_y), th.radius_px)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=th.shadow_blur / 2.0) * th.shadow_alpha
        base = base * (1.0 - shadow[..., None])
        self.base: Frame = np.clip(base + 0.5, 0, 255).astype(np.uint8)

        win_a = _rounded_rect_alpha((cw, ch), win, th.radius_px)
        inner = (win[0] + th.border_px, win[1] + th.border_px, win[2] - th.border_px, win[3] - th.border_px)
        inner_a = _rounded_rect_alpha((cw, ch), inner, max(0.0, th.radius_px - th.border_px))
        border_a = np.clip(win_a - inner_a, 0.0, 1.0)
        self.border_color = np.array(_hex_to_bgr(th.border), dtype=np.float32)

        # Only the edge strips need blending; the interior is a plain copy.
        e = th.radius_px + 3
        self.strips: list[tuple[tuple[slice, slice], FloatImg, FloatImg]] = []
        regions = [
            (slice(pad - 2, pad + e), slice(pad - 2, pad + vw + 2)),  # top
            (slice(pad + vh - e, pad + vh + 2), slice(pad - 2, pad + vw + 2)),  # bottom
            (slice(pad + e, pad + vh - e), slice(pad - 2, pad + e)),  # left
            (slice(pad + e, pad + vh - e), slice(pad + vw - e, pad + vw + 2)),  # right
        ]
        for sl in regions:
            self.strips.append((sl, win_a[sl][..., None].copy(), border_a[sl][..., None].copy()))
        self.inner_rect = (pad, pad, pad + vw, pad + vh)

    # ------------------------------------------------------------------ sources
    def _source(self, t: float) -> Frame:
        i = max(0, bisect_right(self.frame_ts, t) - 1)
        if i != self._src_idx or self._src is None:
            path = self.rec.frames[i].path
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"could not decode source frame {path}")
            self._src, self._src_idx = img, i
            self._scaled_key = None
        assert self._src is not None
        return self._src

    def _content(self, src: Frame, cx: float, cy: float, zoom: float, settled: bool) -> Frame:
        """Crop the camera window (centre cx, cy in CSS px) and render it at exactly vw x vh."""
        vw, vh = self.vw, self.vh
        sh, sw = src.shape[:2]
        k = zoom * vw / sw  # source px -> output px
        zkey = round(zoom, 3)
        if self._scaled_key != (self._src_idx, zkey) or self._scaled is None:
            if abs(k - 1.0) < 1e-3:
                scaled = src
            else:
                size = (max(vw, round(sw * k)), max(vh, round(sh * k)))
                interp = cv2.INTER_AREA if k < 1.0 else cv2.INTER_CUBIC
                scaled = cv2.resize(src, size, interpolation=interp)
            self._scaled, self._scaled_key = scaled, (self._src_idx, zkey)
        scaled = self._scaled
        r = scaled.shape[1] / vw  # output px per CSS px at this zoom
        ox = (cx - vw / (2 * zoom)) * r
        oy = (cy - vh / (2 * zoom)) * r
        ox = min(max(ox, 0.0), scaled.shape[1] - vw)
        oy = min(max(oy, 0.0), scaled.shape[0] - vh)
        if settled or (abs(ox - round(ox)) < 1e-3 and abs(oy - round(oy)) < 1e-3):
            ix, iy = int(round(ox)), int(round(oy))
            return scaled[iy : iy + vh, ix : ix + vw]
        m = np.float32([[1, 0, -ox], [0, 1, -oy]])
        return cv2.warpAffine(scaled, m, (vw, vh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------ camera
    def _clamp_center(self, cx: float, cy: float, zoom: float) -> tuple[float, float]:
        hw, hh = self.vw / (2 * zoom), self.vh / (2 * zoom)
        return min(max(cx, hw), self.vw - hw), min(max(cy, hh), self.vh - hh)

    def _camera_target(self, t: float, prev: tuple[float, float]) -> tuple[float, float, float]:
        mx, my, tf = self.tel.at(t + self.settings.lookahead_s)
        zoom = min(2.0, max(1.0, tf.target_zoom))
        if tf.focus_box is not None:
            bx, by, bw, bh = tf.focus_box
            # Never crop the subject: a big result area (table, chart) gets a wider shot that fits it whole.
            fit = min(0.94 * self.vw / max(bw, 1.0), 0.94 * self.vh / max(bh, 1.0))
            zoom = max(1.0, min(zoom, fit))
            pull = 0.2 if bw * bh < 0.08 * self.vw * self.vh else 0.0  # big areas: stay centred on them
            cx = (1 - pull) * (bx + bw / 2) + pull * mx
            cy = (1 - pull) * (by + bh / 2) + pull * my
        elif tf.camera_mode == "pan" and zoom > 1.0:
            cx, cy = prev
            dzx = self.settings.pan_deadzone * self.vw / zoom
            dzy = self.settings.pan_deadzone * self.vh / zoom
            if mx > cx + dzx:
                cx = mx - dzx
            elif mx < cx - dzx:
                cx = mx + dzx
            if my > cy + dzy:
                cy = my - dzy
            elif my < cy - dzy:
                cy = my + dzy
        else:
            cx, cy = self.vw / 2, self.vh / 2
        cx, cy = self._clamp_center(cx, cy, zoom)
        return cx, cy, zoom

    # ------------------------------------------------------------------ overlays
    def _press_amount(self, t: float) -> float:
        """0 -> 1 envelope around each click (fast in, slightly slower out)."""
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
        x0, y0, x1, y1 = self.inner_rect
        zf = 0.65 + 0.35 * zoom
        for c in self.rec.clicks:
            age = t - c.t
            if not 0.0 <= age <= dur:
                continue
            u = age / dur
            e = 1.0 - (1.0 - u) ** 3
            radius = (6.0 + 28.0 * e) * zf
            alpha = 0.55 * (1.0 - u)
            px, py = to_canvas(c.x, c.y)
            pad = int(radius + 4)
            rx0, ry0 = max(x0, int(px) - pad), max(y0, int(py) - pad)
            rx1, ry1 = min(x1, int(px) + pad + 1), min(y1, int(py) + pad + 1)
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            roi = canvas[ry0:ry1, rx0:rx1]
            overlay = roi.copy()
            shift = 4
            center = (int(round((px - rx0) * (1 << shift))), int(round((py - ry0) * (1 << shift))))
            cv2.circle(overlay, center, int(round(radius * (1 << shift))), color, 2, cv2.LINE_AA, shift)
            cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    def _draw_cursor(self, canvas: Frame, px: float, py: float, scale: float) -> None:
        sp = self.cursor
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
        # Clip to the window interior.
        x0, y0, x1, y1 = self.inner_rect
        cx0, cy0 = max(ix, x0), max(iy, y0)
        cx1, cy1 = min(ix + w + 1, x1), min(iy + h + 1, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sl_s = (slice(cy0 - iy, cy1 - iy), slice(cx0 - ix, cx1 - ix))
        roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
        out = roi * (1.0 - a[sl_s]) + rgb[sl_s]
        canvas[cy0:cy1, cx0:cx1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ main loop
    def frames(self) -> Iterator[Frame]:
        st, th = self.settings, self.theme
        dt = 1.0 / self.fps
        t0 = self.rec.t_start
        cx, cy, z = self._camera_target(t0, (self.vw / 2, self.vh / 2))
        sx, sy, sz = _Spring(cx, st.omega, st.zeta), _Spring(cy, st.omega, st.zeta), _Spring(z, st.omega, st.zeta)
        prev_target = (cx, cy)
        pad = self.pad
        for i in range(self.frame_count):
            t = t0 + i * dt
            tx, ty, tz = self._camera_target(t, prev_target)
            prev_target = (tx, ty)
            if i > 0:
                sx.step(tx, dt, st.substeps)
                sy.step(ty, dt, st.substeps)
                sz.step(tz, dt, st.substeps)
            zoom = min(2.0, max(1.0, sz.pos))
            cam_x, cam_y = self._clamp_center(sx.pos, sy.pos, zoom)
            settled = abs(sx.vel) * zoom < 0.4 and abs(sy.vel) * zoom < 0.4 and abs(sz.vel) < 1e-3

            src = self._source(t)
            content = self._content(src, cam_x, cam_y, zoom, settled)

            canvas = self.base.copy()
            canvas[pad : pad + self.vh, pad : pad + self.vw] = content
            for sl, wa, ba in self.strips:
                region = canvas[sl].astype(np.float32)
                region = self.base[sl].astype(np.float32) * (1.0 - wa) + region * wa
                region = region * (1.0 - ba) + self.border_color * ba
                canvas[sl] = np.clip(region + 0.5, 0, 255).astype(np.uint8)

            mapper = _Mapper(cam_x, cam_y, zoom, self.vw, self.vh, pad)
            self._draw_ripples(canvas, t, mapper, zoom)
            mx, my, _ = self.tel.at(t)
            px, py = mapper(mx, my)
            press = self._press_amount(t)
            press_scale = 1.0 - (1.0 - th.click_press_scale) * press
            cursor_scale = 1.25 * (0.65 + 0.35 * zoom) * th.cursor_scale * press_scale
            self._draw_cursor(canvas, px, py, cursor_scale)
            self.last_frame = canvas
            yield canvas


@dataclass(slots=True)
class _Mapper:
    """CSS px -> canvas px for the current camera."""

    cam_x: float
    cam_y: float
    zoom: float
    vw: int
    vh: int
    pad: int

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        left = self.cam_x - self.vw / (2 * self.zoom)
        top = self.cam_y - self.vh / (2 * self.zoom)
        return (x - left) * self.zoom + self.pad, (y - top) * self.zoom + self.pad
