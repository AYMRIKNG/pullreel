"""Project-level configuration via ``pullreel.toml`` (auto-discovered at the repository root).

Validated with Pydantic so typos or bad values fail early with a clear, friendly message.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pullreel.models import (
    AspectRatio,
    BackgroundStyle,
    ExportFormat,
    QualityPreset,
    RenderSettings,
    ResolutionPreset,
    StudioTheme,
    WindowChrome,
)

CONFIG_FILENAME = "pullreel.toml"


class ConfigError(ValueError):
    """Raised when ``pullreel.toml`` cannot be read or fails schema validation."""


class BrandConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    primary_color: str = "#09090B"
    accent_color: str = "#8B5CF6"
    logo: str | None = None
    font: str | None = None


class ThemeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background_style: BackgroundStyle | None = None
    background: str | None = None
    background_end: str | None = None
    background_angle: float | None = None
    background_preset: str | None = None
    wallpaper: str | None = None
    wallpaper_blur: int | None = Field(default=None, ge=0, le=200)
    border: str | None = None
    border_px: int | None = Field(default=None, ge=0, le=12)
    radius_px: int | None = Field(default=None, ge=0, le=64)
    padding_px: int | None = Field(default=None, ge=0, le=240)
    shadow_blur: int | None = Field(default=None, ge=0, le=160)
    shadow_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow_offset_y: int | None = Field(default=None, ge=-60, le=100)
    window_chrome: WindowChrome | None = None
    show_url_bar: bool | None = None
    cursor_scale: float | None = Field(default=None, ge=0.5, le=2.5)
    cursor_trail: bool | None = None
    ripple_color: tuple[int, int, int] | None = None
    show_captions: bool | None = None
    show_keystrokes: bool | None = None


class IntroConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    title: str | None = None
    subtitle: str | None = None
    duration_s: float = Field(default=1.6, ge=0.0, le=6.0)


class OutroConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    title: str | None = None
    cta: str | None = None
    duration_s: float = Field(default=1.6, ge=0.0, le=6.0)


class RenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspect: AspectRatio | None = None
    resolution: ResolutionPreset = "native"
    quality: QualityPreset = "standard"
    formats: list[ExportFormat] = Field(default_factory=lambda: ["mp4", "gif"])
    screenshots: bool = False
    fps: int | None = Field(default=None, ge=10, le=120)
    crf: int | None = Field(default=None, ge=0, le=51)
    motion_blur: float | None = Field(default=None, ge=0.0, le=1.0)
    gif_max_mb: float | None = Field(default=None, gt=0.5, le=100.0)

    @field_validator("formats")
    @classmethod
    def _ensure_mp4(cls, v: list[ExportFormat]) -> list[ExportFormat]:
        if not v:
            return ["mp4"]
        return list(dict.fromkeys(v))


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_mutations: bool = True
    dry_rehearsal: bool = False
    extra_blocklist: list[str] = Field(default_factory=list)


class PullReelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: BrandConfig = Field(default_factory=BrandConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    intro: IntroConfig = Field(default_factory=IntroConfig)
    outro: OutroConfig = Field(default_factory=OutroConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    source_path: Path | None = Field(default=None, exclude=True)


def discover_config(repo: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    candidate = repo / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def _resolve_asset(base_dir: Path | None, value: str | None) -> str | None:
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute() and base_dir is not None:
        p = (base_dir / p).resolve()
    return str(p)


def load_project_config(repo: Path, explicit: Path | None = None) -> PullReelConfig:
    path = discover_config(repo, explicit)
    if path is None:
        return PullReelConfig()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"invalid {path.name}: {exc}") from exc
    try:
        cfg = PullReelConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
        raise ConfigError(f"invalid {path.name}: {details}") from exc
    cfg.source_path = path
    return cfg


def build_render_profile(
    cfg: PullReelConfig,
    *,
    aspect: AspectRatio | None = None,
    resolution: ResolutionPreset | None = None,
    quality: QualityPreset | None = None,
    chrome: WindowChrome | None = None,
    url_bar: bool | None = None,
    background_preset: str | None = None,
    intro: bool | None = None,
    outro: bool | None = None,
    screenshots: bool | None = None,
    formats: list[ExportFormat] | None = None,
    feature_title: str | None = None,
    browser_url: str | None = None,
) -> tuple[StudioTheme, RenderSettings]:
    """Combine ``pullreel.toml`` and CLI overrides into a validated ``(StudioTheme, RenderSettings)``."""
    base_dir = cfg.source_path.parent if cfg.source_path is not None else None
    t = cfg.theme
    r = cfg.render
    b = cfg.brand

    eff_quality: QualityPreset = quality or r.quality
    theme_kwargs: dict[str, object] = {
        "accent": b.accent_color,
        "brand_name": b.name,
        "logo_path": _resolve_asset(base_dir, b.logo),
        "font_path": _resolve_asset(base_dir, b.font),
        "aspect": aspect if aspect is not None else r.aspect,
        "resolution": resolution if resolution is not None else r.resolution,
        "formats": list(formats) if formats is not None else list(r.formats),
        "screenshots": screenshots if screenshots is not None else r.screenshots,
        "intro_enabled": intro if intro is not None else cfg.intro.enabled,
        "intro_title": cfg.intro.title or feature_title,
        "intro_subtitle": cfg.intro.subtitle,
        "intro_duration_s": cfg.intro.duration_s,
        "outro_enabled": outro if outro is not None else cfg.outro.enabled,
        "outro_title": cfg.outro.title or b.name or feature_title,
        "outro_cta": cfg.outro.cta,
        "outro_duration_s": cfg.outro.duration_s,
        "browser_url": browser_url,
    }
    if t.background is not None:
        theme_kwargs["background"] = t.background
    elif b.primary_color != "#09090B":
        theme_kwargs["background"] = b.primary_color
    for src_field, dst_field in (
        ("background_style", "background_style"),
        ("background_end", "background_end"),
        ("background_angle", "background_angle"),
        ("wallpaper_blur", "wallpaper_blur"),
        ("border", "border"),
        ("border_px", "border_px"),
        ("radius_px", "radius_px"),
        ("padding_px", "padding_px"),
        ("shadow_blur", "shadow_blur"),
        ("shadow_alpha", "shadow_alpha"),
        ("shadow_offset_y", "shadow_offset_y"),
        ("cursor_scale", "cursor_scale"),
        ("cursor_trail", "cursor_trail"),
        ("ripple_color", "ripple_color"),
        ("show_captions", "show_captions"),
        ("show_keystrokes", "show_keystrokes"),
    ):
        val = getattr(t, src_field)
        if val is not None:
            theme_kwargs[dst_field] = val
    if t.wallpaper:
        theme_kwargs["wallpaper_path"] = _resolve_asset(base_dir, t.wallpaper)
        if t.background_style is None:
            theme_kwargs["background_style"] = "wallpaper"
    if background_preset is not None:
        theme_kwargs["background_preset"] = background_preset
    elif t.background_preset is not None:
        theme_kwargs["background_preset"] = t.background_preset
    if chrome is not None:
        theme_kwargs["window_chrome"] = chrome
    elif t.window_chrome is not None:
        theme_kwargs["window_chrome"] = t.window_chrome
    if url_bar is not None:
        theme_kwargs["show_url_bar"] = url_bar
    elif t.show_url_bar is not None:
        theme_kwargs["show_url_bar"] = t.show_url_bar
    if r.gif_max_mb is not None:
        theme_kwargs["gif_max_mb"] = r.gif_max_mb

    try:
        base_theme = StudioTheme.model_validate(theme_kwargs)
    except ValidationError as exc:
        raise ConfigError(f"invalid theme configuration: {exc}") from exc

    theme, settings = base_theme.with_quality(eff_quality)
    if r.fps is not None:
        theme.output_fps = r.fps
    if r.crf is not None:
        theme.crf = r.crf
    if r.motion_blur is not None:
        settings.motion_blur = r.motion_blur
    return theme, settings
