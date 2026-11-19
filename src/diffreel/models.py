"""Typed data contracts shared by every DiffReel stage.

``Storyboard`` is the exact JSON contract the storyboard agent (LLM) produces; ``Recording`` is what the
browser runner hands to the cinema engine and can be saved to disk to re-render without re-recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

LocatorStrategy = Literal["role", "label", "placeholder", "text", "testid", "css"]
CameraMode = Literal["wide", "focus", "pan"]
ActionType = Literal["wait", "click", "fill", "press", "hover", "scroll"]

Box = tuple[float, float, float, float]  # x, y, width, height in CSS pixels


class Locator(BaseModel):
    """Semantic locator. For ``role``: ``value`` is the ARIA role and ``name`` the accessible name."""

    strategy: LocatorStrategy
    value: str = Field(min_length=1)
    name: str | None = None

    def describe(self) -> str:
        if self.strategy == "role":
            return f"role={self.value!r} name={self.name!r}"
        return f"{self.strategy}={self.value!r}"


class Camera(BaseModel):
    mode: CameraMode = "wide"
    zoom: float = 1.0

    @field_validator("zoom")
    @classmethod
    def _clamp_zoom(cls, v: float) -> float:
        if not 1.0 <= v <= 2.0:
            raise ValueError("camera.zoom must be within [1.0, 2.0]")
        return v

    @model_validator(mode="after")
    def _wide_is_unzoomed(self) -> Self:
        if self.mode == "wide":
            self.zoom = 1.0
        return self


class ActionStep(BaseModel):
    action: ActionType
    locator: Locator | None = None
    input_value: str | None = None
    camera: Camera = Field(default_factory=Camera)
    dwell_after_ms: int = Field(default=450, ge=0, le=10_000)

    @model_validator(mode="after")
    def _check_requirements(self) -> Self:
        if self.action in ("click", "fill", "hover") and self.locator is None:
            raise ValueError(f"action {self.action!r} requires a locator")
        if self.action in ("fill", "press") and not self.input_value:
            raise ValueError(f"action {self.action!r} requires input_value")
        if self.action == "scroll" and self.input_value is not None:
            try:
                int(self.input_value)
            except ValueError as exc:
                raise ValueError("scroll input_value must be an integer pixel delta") from exc
        return self


class Storyboard(BaseModel):
    feature_summary: str
    start_path: str = "/"
    steps: list[ActionStep] = Field(min_length=1)


class TelemetryFrame(BaseModel):
    t: float  # wall-clock seconds (same clock as CDP screencast timestamps)
    mouse_x: float  # CSS px
    mouse_y: float
    is_clicking: bool
    target_zoom: float
    focus_box: Box | None
    camera_mode: CameraMode


class ClickEvent(BaseModel):
    t: float
    x: float  # CSS px
    y: float


class SourceFrame(BaseModel):
    t: float
    path: Path


class Recording(BaseModel):
    viewport: tuple[int, int] = (1280, 800)
    scale: float = 2.0
    frames: list[SourceFrame]
    telemetry: list[TelemetryFrame]
    clicks: list[ClickEvent] = Field(default_factory=list)
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def source_size(self) -> tuple[int, int]:
        return round(self.viewport[0] * self.scale), round(self.viewport[1] * self.scale)

    def save_json(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> Recording:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


FileStatus = Literal["added", "modified", "deleted", "renamed"]


class ChangedFile(BaseModel):
    path: str
    status: FileStatus
    is_ui: bool = False  # markup / component / style file
    is_route: bool = False  # defines a page or route
    added: int = 0
    removed: int = 0


class DiffSummary(BaseModel):
    """Compact, LLM-ready digest of a branch's UI changes (produced by ``diff_analyzer``)."""

    base_ref: str
    head_ref: str
    files: list[ChangedFile] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)  # component / function names introduced or touched
    labels: list[str] = Field(default_factory=list)  # user-facing strings: button text, aria-label, placeholder...
    test_ids: list[str] = Field(default_factory=list)  # data-testid / data-test / data-cy values
    routes: list[str] = Field(default_factory=list)  # URL paths that the change adds or touches, best first
    handlers: list[str] = Field(default_factory=list)  # event handlers & state setters (onClick=..., useState)
    excerpt: str = ""  # trimmed added-lines excerpt of the most relevant UI hunks

    @property
    def ui_files(self) -> list[ChangedFile]:
        return [f for f in self.files if f.is_ui or f.is_route]

    @property
    def is_empty(self) -> bool:
        return not self.files

    def to_prompt(self, max_chars: int = 6000) -> str:
        """Render the ``GIT_DIFF_SUMMARY`` block sent to the storyboard LLM, capped at ``max_chars``."""
        lines = [f"Base: {self.base_ref} -> Head: {self.head_ref}", "Changed UI files:"]
        for f in self.ui_files[:25] or self.files[:15]:
            tag = " [route]" if f.is_route else ""
            lines.append(f"  {f.status:<8} {f.path} (+{f.added}/-{f.removed}){tag}")
        for title, items in (
            ("Routes", self.routes),
            ("Components", self.components),
            ("User-facing labels", self.labels),
            ("Test ids", self.test_ids),
            ("Handlers/state", self.handlers),
        ):
            if items:
                lines.append(f"{title}: " + ", ".join(repr(i) for i in items[:30]))
        head = "\n".join(lines)
        room = max_chars - len(head) - 40
        if self.excerpt and room > 200:
            excerpt = self.excerpt if len(self.excerpt) <= room else self.excerpt[:room] + "\n... (truncated)"
            head += "\n\nAdded lines (excerpt):\n" + excerpt
        return head[:max_chars]


class StudioTheme(BaseModel):
    background: str = "#09090B"
    border: str = "#27272A"
    border_px: int = 1
    radius_px: int = 16
    padding_px: int = 56
    shadow_blur: int = 48
    shadow_alpha: float = 0.55
    shadow_offset_y: int = 18
    cursor_scale: float = 1.0
    ripple_color: tuple[int, int, int] = (255, 255, 255)
    ripple_ms: int = 250
    click_press_scale: float = 0.88
    output_fps: int = 60
    gif_fps: int = 30
    gif_max_mb: float = 5.0
    crf: int = 18


class RenderSettings(BaseModel):
    omega: float = 6.5  # rad/s, spring natural frequency
    zeta: float = 1.0  # critical damping
    lookahead_s: float = 0.25  # camera anticipates the cursor by this much
    substeps: int = 4
    pan_deadzone: float = 0.30  # fraction of the crop the cursor may roam before a pan
