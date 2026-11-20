"""Live pipeline view for the terminal: one line per stage, updated in place.

     PullReel record · bstate (fix/sync-hors-ligne -> master)
     ✔ Diff         282 files · 88 UI · 34 routes                   0.9s
     ⠸ Storyboard   gemma4:26b · writing · 212 tokens · 6.1 tok/s   41s
     ○ Recording
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import TracebackType

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from pullreel.progress import ProgressEvent, Stage

LABELS: dict[Stage, str] = {
    "diff": "Diff",
    "scout": "Scout",
    "snapshot": "Snapshot",
    "storyboard": "Storyboard",
    "rehearsal": "Rehearsal",
    "recording": "Recording",
    "render": "Render",
    "gif": "GIF",
    "exports": "Exports",
}


@dataclass(slots=True)
class _Row:
    stage: Stage
    state: str = "pending"  # pending | running | done | fail | skipped
    detail: str = ""
    completed: float | None = None
    total: float | None = None
    t0: float | None = None
    t1: float | None = None
    spinner: Spinner = field(default_factory=lambda: Spinner("dots", style="magenta"))

    def elapsed(self) -> str:
        if self.t0 is None:
            return ""
        secs = (self.t1 or time.monotonic()) - self.t0
        return f"{secs:.1f}s" if secs < 10 else f"{secs:.0f}s"


class PipelineView:
    """Context manager that draws the pipeline and receives :class:`ProgressEvent` via :meth:`report`."""

    def __init__(self, console: Console, stages: list[Stage], title: str) -> None:
        self.console = console
        self.title = title
        self.rows: dict[Stage, _Row] = {s: _Row(s) for s in stages}
        self.t0 = time.monotonic()
        utf = (console.encoding or "").lower().replace("-", "").startswith("utf")
        self._glyphs = (
            {"done": "✔", "fail": "✖", "skipped": "–", "pending": "○"}
            if utf
            else {"done": "v", "fail": "x", "skipped": "-", "pending": "."}
        )
        self._live = Live(self, console=console, refresh_per_second=10, transient=False)

    # -- rich protocol -------------------------------------------------------------------------------------------
    def __rich__(self) -> RenderableType:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=2)
        grid.add_column(no_wrap=True, min_width=13)
        grid.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        grid.add_column(width=22)
        grid.add_column(justify="right", width=6)
        for row in self.rows.values():
            icon: RenderableType
            if row.state == "running":
                icon = row.spinner
            else:
                glyphs = self._glyphs
                icon = Text(glyphs.get(row.state, glyphs["pending"]),
                            style={"done": "green", "fail": "red"}.get(row.state, "dim"))  # fmt: skip
            label_style = {"pending": "dim", "skipped": "dim", "fail": "bold red"}.get(row.state, "bold")
            bar: RenderableType = ""
            if row.state == "running" and row.total:
                bar = ProgressBar(total=row.total, completed=row.completed or 0, width=20,
                                  complete_style="magenta", finished_style="green")  # fmt: skip
            elif row.state == "running":
                bar = ProgressBar(total=None, width=20, pulse_style="magenta")  # indeterminate
            detail_style = "dim" if row.state in ("pending", "done") else ("red" if row.state == "fail" else "")
            grid.add_row(icon, Text(LABELS[row.stage], style=label_style), Text(row.detail, style=detail_style),
                         bar, Text(row.elapsed(), style="dim"))  # fmt: skip
        total = time.monotonic() - self.t0
        head = Text.from_markup(f"[bold magenta]PullReel[/] {escape(self.title)}  [dim]{total:.0f}s[/]")
        return Group(head, grid)

    # -- events --------------------------------------------------------------------------------------------------
    def report(self, ev: ProgressEvent) -> None:
        row = self.rows.get(ev.stage)
        if row is None:
            return
        now = time.monotonic()
        if ev.state == "start":
            if row.t0 is None or row.state in ("done", "fail"):
                row.t0 = now
            row.state, row.t1 = "running", None
            row.completed, row.total = ev.completed, ev.total
        elif ev.state == "done":
            row.state, row.t1 = "done", now
            if row.t0 is None:
                row.t0 = now
        elif ev.state == "fail":
            row.state, row.t1 = "fail", now
        else:
            if row.state == "pending":
                row.state, row.t0 = "running", now
            if ev.total is not None:
                row.total = ev.total
            if ev.completed is not None:
                row.completed = ev.completed
        if ev.detail is not None:
            row.detail = ev.detail

    def skip(self, stage: Stage, detail: str) -> None:
        row = self.rows.get(stage)
        if row is not None:
            row.state, row.detail = "skipped", detail

    def fail_running(self) -> None:
        """Mark whatever was in flight as failed (an exception escaped the pipeline)."""
        for row in self.rows.values():
            if row.state == "running":
                row.state, row.t1 = "fail", time.monotonic()

    # -- context manager -----------------------------------------------------------------------------------------
    def __enter__(self) -> PipelineView:
        self._live.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is not None:
            self.fail_running()
        self._live.refresh()
        self._live.__exit__(exc_type, exc, tb)
        if not self.console.is_terminal:  # piped/redirected: Live leaves the cursor at the end of the last row
            self.console.line()
