"""Typography, animated overlays (intro/outro cards, step captions, keystroke HUD) and annotated screenshot sheets.

All text uses the bundled Inter variable font (SIL Open Font License 1.1) with optional user font override.
"""

from __future__ import annotations

import functools
import math
import textwrap
from importlib import resources
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from pullreel.models import KeyEvent, StudioTheme

Frame = NDArray[np.uint8]


def ease_out_cubic(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 3


def ease_in_out_cubic(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 4.0 * u * u * u if u < 0.5 else 1.0 - ((-2.0 * u + 2.0) ** 3) / 2.0


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return (139, 92, 246)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    r, g, b = hex_to_rgb(color)
    return b, g, r


def is_light_color(color: str) -> bool:
    r, g, b = hex_to_rgb(color)
    return (0.299 * r + 0.587 * g + 0.114 * b) > 155.0


@functools.cache
def bundled_font_path() -> Path | None:
    """Locate the bundled Inter OFL variable font."""
    try:
        ref = resources.files("pullreel.assets").joinpath("fonts", "Inter-Variable.ttf")
        path = Path(str(ref))
        if path.is_file():
            return path
    except Exception:
        pass
    local = Path(__file__).parent / "assets" / "fonts" / "Inter-Variable.ttf"
    return local if local.is_file() else None


@functools.cache
def load_font(size: int, weight: int = 500, custom_path: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load Inter (or ``custom_path``) at ``size`` px and variable ``weight`` (100..900)."""
    size = max(10, int(round(size)))
    candidates: list[Path] = []
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            candidates.append(p)
    bundled = bundled_font_path()
    if bundled is not None:
        candidates.append(bundled)
    for path in candidates:
        try:
            font = ImageFont.truetype(str(path), size)
            try:
                axes = font.get_variation_axes()
                if axes:
                    vals: list[float] = []
                    for ax in axes:
                        name = ax.get("name", b"").decode("utf-8", "ignore").lower()
                        lo, hi = float(ax["minimum"]), float(ax["maximum"])
                        if "weight" in name:
                            vals.append(min(hi, max(lo, float(weight))))
                        elif "optical" in name:
                            vals.append(min(hi, max(lo, float(size))))
                        else:
                            vals.append(float(ax["default"]))
                    font.set_variation_by_axes(vals)
            except Exception:
                pass
            return font
        except OSError:
            continue
    return ImageFont.load_default()


def _composite_rgba(canvas_bgr: Frame, overlay_rgba: Image.Image, alpha_mul: float = 1.0, offset: tuple[int, int] = (0, 0)) -> None:
    """Alpha-composite a PIL RGBA overlay onto a BGR OpenCV frame in place at ``offset=(x, y)``."""
    if alpha_mul <= 1e-4:
        return
    ch, cw = canvas_bgr.shape[:2]
    ox, oy = offset
    ow, oh = overlay_rgba.size
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(cw, ox + ow), min(ch, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return
    arr = np.asarray(overlay_rgba, dtype=np.float32)
    sub = arr[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    a = (sub[..., 3:4] / 255.0) * float(alpha_mul)
    bgr = sub[..., :3][..., ::-1]
    roi = canvas_bgr[y0:y1, x0:x1]
    out = roi.astype(np.float32) * (1.0 - a) + bgr * a
    np.clip(out + 0.5, 0, 255, out=out)
    roi[:] = out.astype(np.uint8)


def _load_logo(logo_path: str | None, max_h: int) -> Image.Image | None:
    if not logo_path:
        return None
    p = Path(logo_path)
    if not p.is_file():
        return None
    try:
        img = Image.open(p).convert("RGBA")
        w, h = img.size
        if h <= 0 or w <= 0:
            return None
        scale = min(1.0, max_h / h)
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        return img
    except OSError:
        return None


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


@functools.lru_cache(maxsize=8)
def _cached_title_overlay(
    w: int,
    h: int,
    title: str,
    subtitle: str | None,
    cta: str | None,
    background: str,
    accent: str,
    brand_name: str | None,
    logo_path: str | None,
    font_path: str | None,
    is_outro: bool,
) -> Image.Image:
    """Pre-render the full-res title card RGBA layer once; animated frames simply shift & fade it."""
    scale = min(w / 1280.0, h / 720.0)
    light = is_light_color(background)
    fg = (18, 18, 22) if light else (250, 250, 252)
    muted = (92, 92, 104) if light else (168, 168, 180)
    accent_rgb = hex_to_rgb(accent)

    # Render accent glow at 1/8 resolution so Gaussian blur takes <1 ms even at 4K.
    gw, gh = max(32, w // 8), max(32, h // 8)
    glow_small = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow_small)
    rx, ry = int(gw * 0.24), int(gh * 0.22)
    gdraw.ellipse((gw // 2 - rx, gh // 2 - ry, gw // 2 + rx, gh // 2 + ry), fill=(*accent_rgb, 42 if not light else 26))
    glow_arr = cv2.GaussianBlur(np.asarray(glow_small), (0, 0), sigmaX=max(2.0, gw * 0.08))
    glow_full = cv2.resize(glow_arr, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = Image.fromarray(glow_full, mode="RGBA")
    draw = ImageDraw.Draw(overlay)

    cx = w // 2
    logo = _load_logo(logo_path, max(36, int(round(56 * scale))))
    title_font = load_font(int(round(48 * scale)), weight=700, custom_path=font_path)
    sub_font = load_font(int(round(22 * scale)), weight=400, custom_path=font_path)
    badge_font = load_font(int(round(16 * scale)), weight=600, custom_path=font_path)

    max_chars = max(20, int(w / max(1, 26 * scale)))
    title_lines = textwrap.wrap(title.strip(), width=max_chars)[:2] or [title.strip()]
    sub_lines = textwrap.wrap(subtitle.strip(), width=int(max_chars * 1.5))[:2] if subtitle else []

    line_gap = int(round(10 * scale))
    block_h = 0
    if logo is not None:
        block_h += logo.height + int(round(22 * scale))
    elif brand_name:
        block_h += int(round(36 * scale))
    t_heights = [_text_size(draw, ln, title_font)[1] for ln in title_lines]
    block_h += sum(t_heights) + max(0, len(t_heights) - 1) * line_gap
    if sub_lines:
        block_h += int(round(18 * scale))
        s_heights = [_text_size(draw, ln, sub_font)[1] for ln in sub_lines]
        block_h += sum(s_heights) + max(0, len(s_heights) - 1) * int(round(6 * scale))
    if cta:
        block_h += int(round(56 * scale))

    cur_y = (h - block_h) // 2

    if logo is not None:
        overlay.alpha_composite(logo, (cx - logo.width // 2, cur_y))
        cur_y += logo.height + int(round(22 * scale))
    elif brand_name:
        bw, bh = _text_size(draw, brand_name, badge_font)
        pad_x, pad_y = int(round(14 * scale)), int(round(6 * scale))
        bx0, by0 = cx - bw // 2 - pad_x, cur_y
        bx1, by1 = cx + bw // 2 + pad_x, cur_y + bh + 2 * pad_y
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=max(8, int(round(14 * scale))),
                               fill=(*accent_rgb, 36), outline=(*accent_rgb, 110), width=max(1, int(round(scale))))
        draw.text((cx - bw // 2, by0 + pad_y - 1), brand_name, font=badge_font, fill=(*accent_rgb, 255))
        cur_y = by1 + int(round(16 * scale))

    for ln, lh in zip(title_lines, t_heights):
        tw, _ = _text_size(draw, ln, title_font)
        draw.text((cx - tw // 2, cur_y), ln, font=title_font, fill=(*fg, 255))
        cur_y += lh + line_gap

    if sub_lines:
        cur_y += int(round(8 * scale))
        for ln in sub_lines:
            sw, sh = _text_size(draw, ln, sub_font)
            draw.text((cx - sw // 2, cur_y), ln, font=sub_font, fill=(*muted, 240))
            cur_y += sh + int(round(6 * scale))

    if cta:
        cur_y += int(round(22 * scale))
        cw, ch_t = _text_size(draw, cta, badge_font)
        pad_x, pad_y = int(round(22 * scale)), int(round(11 * scale))
        bx0, by0 = cx - cw // 2 - pad_x, cur_y
        bx1, by1 = cx + cw // 2 + pad_x, cur_y + ch_t + 2 * pad_y
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=max(10, int(round(18 * scale))),
                               fill=(*accent_rgb, 235 if is_outro else 210))
        draw.text((cx - cw // 2, by0 + pad_y - 1), cta, font=badge_font, fill=(255, 255, 255, 255))

    return overlay


def render_title_card(
    backdrop_bgr: Frame,
    *,
    title: str,
    subtitle: str | None = None,
    cta: str | None = None,
    theme: StudioTheme,
    progress: float = 1.0,
    is_outro: bool = False,
) -> Frame:
    """Render an animated intro or outro card on top of ``backdrop_bgr`` (cached layer + fast shift/fade)."""
    h, w = backdrop_bgr.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    card = backdrop_bgr.copy()
    e = ease_out_cubic(progress)
    y_shift = int(round((1.0 - e) * 24.0 * scale))
    overlay = _cached_title_overlay(
        w, h, title, subtitle, cta,
        theme.background, theme.accent, theme.brand_name, theme.logo_path, theme.font_path, is_outro,
    )
    _composite_rgba(card, overlay, alpha_mul=e, offset=(0, y_shift))
    return card


@functools.lru_cache(maxsize=64)
def _cached_caption_pill(
    text: str,
    w: int,
    scale_q: int,
    background: str,
    accent: str,
    font_path: str | None,
) -> Image.Image:
    """Pre-render a tight RGBA pill image for a caption string."""
    scale = scale_q / 100.0
    font = load_font(int(round(19 * scale)), weight=500, custom_path=font_path)
    tmp_draw = ImageDraw.Draw(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))

    max_chars = max(24, int(w / max(1, 14 * scale)))
    lines = textwrap.wrap(text.strip(), width=max_chars)[:2]
    sizes = [_text_size(tmp_draw, ln, font) for ln in lines]
    tw = max(s[0] for s in sizes)
    th = sum(s[1] for s in sizes) + max(0, len(lines) - 1) * int(round(5 * scale))

    pad_x = int(round(22 * scale))
    pad_y = int(round(11 * scale))
    box_w = tw + 2 * pad_x + int(round(12 * scale))
    box_h = th + 2 * pad_y

    pill = Image.new("RGBA", (box_w + 2, box_h + 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pill)

    light = is_light_color(background)
    pill_bg = (255, 255, 255, 228) if light else (14, 14, 18, 218)
    pill_border = (212, 212, 216, 220) if light else (63, 63, 74, 210)
    fg = (18, 18, 22, 255) if light else (250, 250, 252, 255)
    accent_rgb = hex_to_rgb(accent)

    rad = min(box_h // 2, int(round(14 * scale)))
    draw.rounded_rectangle((1, 1, box_w, box_h), radius=rad, fill=pill_bg, outline=pill_border, width=max(1, int(round(scale))))
    dot_r = max(3, int(round(3.5 * scale)))
    dot_x = int(round(14 * scale))
    dot_y = (box_h + 2) // 2
    draw.ellipse((dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r), fill=(*accent_rgb, 255))

    cur_y = pad_y
    text_shift = int(round(6 * scale))
    for ln, (lw, lh) in zip(lines, sizes):
        lx = (box_w - lw) // 2 + text_shift
        draw.text((lx, cur_y), ln, font=font, fill=fg)
        cur_y += lh + int(round(5 * scale))
    return pill


def draw_caption(
    canvas_bgr: Frame,
    text: str,
    alpha: float,
    slide: float,
    theme: StudioTheme,
) -> None:
    """Draw a sleek bottom-center caption pill with ease-out slide + fade (tight ROI compositing)."""
    if not text or alpha <= 1e-3:
        return
    h, w = canvas_bgr.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    pill = _cached_caption_pill(text.strip(), w, int(round(scale * 100)), theme.background, theme.accent, theme.font_path)
    margin_b = max(16, int(round(22 * scale)))
    y_offset = int(round((1.0 - slide) * 14.0 * scale))
    x0 = (w - pill.width) // 2
    y0 = h - margin_b - pill.height + y_offset
    _composite_rgba(canvas_bgr, pill, alpha_mul=alpha, offset=(x0, y0))


_KEY_LABELS: dict[str, str] = {
    "Enter": "↵ Enter",
    "Return": "↵ Enter",
    "Backspace": "⌫",
    "Tab": "⇥ Tab",
    "Escape": "Esc",
    "ControlOrMeta+A": "Ctrl+A",
    "Control+A": "Ctrl+A",
    "Meta+A": "Cmd+A",
    "ArrowDown": "↓",
    "ArrowUp": "↑",
    "ArrowLeft": "←",
    "ArrowRight": "→",
    "Space": "Space",
}


def format_keystroke_badges(keys: Sequence[KeyEvent], t: float, window_s: float = 1.4) -> tuple[list[str], float]:
    """Return ``(badges, alpha)`` for keystrokes active around time ``t``."""
    recent = [k for k in keys if 0.0 <= (t - k.t) <= window_s]
    if not recent:
        return [], 0.0
    last_age = t - recent[-1].t
    first_age = t - recent[0].t
    fade_in = ease_out_cubic(min(1.0, first_age / 0.14)) if first_age < 0.14 else 1.0
    fade_out = 1.0 - ease_out_cubic((last_age - (window_s - 0.35)) / 0.35) if last_age > (window_s - 0.35) else 1.0
    alpha = max(0.0, min(1.0, fade_in * fade_out))

    badges: list[str] = []
    typed_buf: list[str] = []
    for ev in recent:
        if ev.is_shortcut or len(ev.key) > 1:
            if typed_buf:
                badges.append("".join(typed_buf)[-22:])
                typed_buf.clear()
            badges.append(_KEY_LABELS.get(ev.key, ev.key))
        else:
            typed_buf.append(ev.key)
    if typed_buf:
        badges.append("".join(typed_buf)[-22:])
    return badges[-4:], alpha


def draw_keystrokes(canvas_bgr: Frame, keys: Sequence[KeyEvent], t: float, theme: StudioTheme) -> None:
    """Draw a tactile keystroke HUD in the bottom-right corner (tight ROI compositing)."""
    badges, alpha = format_keystroke_badges(keys, t)
    if not badges or alpha <= 1e-3:
        return
    h, w = canvas_bgr.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    font = load_font(int(round(16 * scale)), weight=600, custom_path=theme.font_path)
    tmp_draw = ImageDraw.Draw(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))

    pad_x = int(round(12 * scale))
    pad_y = int(round(7 * scale))
    gap = int(round(8 * scale))
    margin_r = max(18, int(round(26 * scale)))
    margin_b = max(18, int(round(24 * scale)))
    bevel = max(2, int(round(2 * scale)))

    dims = [_text_size(tmp_draw, b, font) for b in badges]
    total_w = sum(dw + 2 * pad_x for dw, _ in dims) + gap * max(0, len(badges) - 1)
    box_h = max(dh for _, dh in dims) + 2 * pad_y

    hud = Image.new("RGBA", (total_w + 4, box_h + bevel + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hud)

    light = is_light_color(theme.background)
    cap_bg = (255, 255, 255, 232) if light else (22, 22, 28, 226)
    cap_border = (200, 200, 208, 230) if light else (78, 78, 92, 220)
    cap_fg = (24, 24, 28, 255) if light else (244, 244, 248, 255)
    rad = max(6, int(round(8 * scale)))

    cur_x = 1
    for badge, (dw, dh) in zip(badges, dims):
        bw = dw + 2 * pad_x
        draw.rounded_rectangle((cur_x, 1 + bevel, cur_x + bw, box_h + bevel), radius=rad, fill=(0, 0, 0, 95))
        draw.rounded_rectangle((cur_x, 1, cur_x + bw, box_h), radius=rad, fill=cap_bg, outline=cap_border,
                               width=max(1, int(round(scale))))
        draw.text((cur_x + pad_x, 1 + (box_h - dh) // 2 - 1), badge, font=font, fill=cap_fg)
        cur_x += bw + gap

    _composite_rgba(canvas_bgr, hud, alpha_mul=alpha, offset=(w - margin_r - total_w, h - margin_b - box_h))


def annotate_screenshot(
    frame_bgr: Frame,
    focus_canvas_box: tuple[float, float, float, float] | None,
    step_num: int,
    caption: str,
    theme: StudioTheme,
) -> Frame:
    """Return a copy of ``frame_bgr`` annotated with a highlight circle/ring and step badge for documentation."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    accent = hex_to_rgb(theme.accent)

    if focus_canvas_box is not None:
        fx, fy, fw, fh = focus_canvas_box
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        radius = max(24.0 * scale, min(max(fw, fh) * 0.58 + 14.0 * scale, min(w, h) * 0.36))
        ring_w = max(3, int(round(4 * scale)))
        draw.ellipse((cx - radius - 4, cy - radius - 4, cx + radius + 4, cy + radius + 4),
                     outline=(*accent, 90), width=ring_w + 4)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     outline=(*accent, 245), width=ring_w)

    badge_font = load_font(int(round(18 * scale)), weight=700, custom_path=theme.font_path)
    label = f"{step_num}. {caption}" if caption else f"Step {step_num}"
    if len(label) > 64:
        label = label[:61].rstrip() + "..."
    tw, th = _text_size(draw, label, badge_font)
    px, py = int(round(16 * scale)), int(round(10 * scale))
    m = max(16, int(round(20 * scale)))
    draw.rounded_rectangle((m, m, m + tw + 2 * px, m + th + 2 * py), radius=max(8, int(round(10 * scale))),
                           fill=(14, 14, 18, 224), outline=(*accent, 220), width=max(1, int(round(1.5 * scale))))
    draw.text((m + px, m + py - 1), label, font=badge_font, fill=(250, 250, 252, 255))

    _composite_rgba(out, overlay, alpha_mul=1.0)
    return out


def build_screenshot_sheet(
    shots: Sequence[tuple[int, str, Frame]],
    theme: StudioTheme,
    *,
    cols: int = 2,
    tile_width: int = 800,
) -> Frame:
    """Assemble annotated screenshots into a clean multi-panel contact sheet PNG."""
    if not shots:
        raise ValueError("no screenshots to assemble")
    cols = max(1, min(cols, len(shots)))
    rows = math.ceil(len(shots) / cols)
    first_h, first_w = shots[0][2].shape[:2]
    tile_h = max(200, int(round(tile_width * first_h / max(1, first_w))))
    gap = 28
    pad = 40
    header_h = 64
    sheet_w = pad * 2 + cols * tile_width + (cols - 1) * gap
    sheet_h = pad * 2 + header_h + rows * tile_h + (rows - 1) * gap

    bg = np.full((sheet_h, sheet_w, 3), hex_to_bgr(theme.background), dtype=np.uint8)
    overlay = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    light = is_light_color(theme.background)
    fg = (18, 18, 22, 255) if light else (250, 250, 252, 255)
    title_font = load_font(26, weight=700, custom_path=theme.font_path)
    heading = theme.intro_title or theme.brand_name or "Key Screenshots"
    draw.text((pad, pad), heading, font=title_font, fill=fg)
    _composite_rgba(bg, overlay, 1.0)

    for idx, (_step_num, _cap, frame) in enumerate(shots):
        r, c = divmod(idx, cols)
        x = pad + c * (tile_width + gap)
        y = pad + header_h + r * (tile_h + gap)
        resized = cv2.resize(frame, (tile_width, tile_h), interpolation=cv2.INTER_AREA)
        bg[y : y + tile_h, x : x + tile_width] = resized
        cv2.rectangle(bg, (x, y), (x + tile_width - 1, y + tile_h - 1), hex_to_bgr(theme.border), 1)
    return bg
