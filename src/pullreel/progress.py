"""Structured progress events, emitted by the pipeline and drawn by the CLI (no UI dependency here)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Stage = Literal["diff", "scout", "snapshot", "storyboard", "rehearsal", "recording", "render", "gif"]
State = Literal["start", "update", "done", "fail"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: Stage
    state: State = "update"
    detail: str | None = None  # short human text: "click 'Add task'", "212 tokens · 6.1 tok/s"
    completed: float | None = None  # with total: drives a progress bar
    total: float | None = None


Reporter = Callable[[ProgressEvent], None]


def silent(_event: ProgressEvent) -> None:
    """Default reporter: ignore everything."""


def say(report: Reporter, stage: Stage, state: State = "update", detail: str | None = None,
        completed: float | None = None, total: float | None = None) -> None:  # fmt: skip
    report(ProgressEvent(stage, state, detail, completed, total))
