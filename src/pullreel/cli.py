"""PullReel command line: `record`, `plan`, `render`, `diff` and the zero-config `demo`."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Annotated
from urllib.parse import urljoin, urlparse

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from pullreel import __version__
from pullreel.cinema_engine import CinemaEngine
from pullreel.diff_analyzer import GitError, analyze_diff
from pullreel.exporter import ExportError, export_gif, export_mp4, export_poster, find_ffmpeg, probe
from pullreel.human_runner import StepError, record_sync
from pullreel.models import ActionStep, Camera, DiffSummary, Locator, Recording, Storyboard, StudioTheme
from pullreel.storyboard_agent import LLMError, PlanError, PlanResult, parse_model_spec, plan_storyboard_sync

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help="[bold]PullReel[/] - turn a git diff into a cinematic UI demo for your Pull Request.",
)
console = Console()

DEFAULT_MODEL = "ollama/qwen2.5-coder:7b"
KNOWN_ERRORS = (StepError, ExportError, GitError, LLMError, PlanError, RuntimeError, ValueError)

# Shared option types ------------------------------------------------------------------------------------------
UrlOpt = Annotated[str, typer.Option("--url", "-u", help="Page of your running app, e.g. http://localhost:3000.")]
RepoOpt = Annotated[Path, typer.Option("--repo", help="Git repository to diff.", file_okay=False)]
BaseOpt = Annotated[str, typer.Option("--base", "-b", help="Base branch to diff against.")]
ModelOpt = Annotated[
    str,
    typer.Option("--model", "-m", help="LLM: ollama/<model> (default, local), openai/<model>, lmstudio/<model>, "
                 "openrouter/<model>, groq/<model>.", envvar="PULLREEL_MODEL"),
]  # fmt: skip
BaseUrlOpt = Annotated[str | None, typer.Option("--base-url", help="Override the LLM endpoint URL.")]
ApiKeyOpt = Annotated[
    str | None, typer.Option("--api-key", help="API key (else PULLREEL_API_KEY / OPENAI_API_KEY...).", show_default=False)
]
HeadedOpt = Annotated[bool, typer.Option(help="Show the browser.")]


def _print_version(value: bool) -> None:
    if value:
        console.print(f"pullreel {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the version and exit.", callback=_print_version, is_eager=True),
    ] = False,
) -> None:
    """PullReel - turn a git diff into a cinematic UI demo for your Pull Request."""


# Helpers -----------------------------------------------------------------------------------------------------
@contextmanager
def _friendly_errors() -> Iterator[None]:
    try:
        yield
    except KNOWN_ERRORS as exc:
        title = {GitError: "git", LLMError: "LLM", PlanError: "storyboard", StepError: "browser",
                 ExportError: "ffmpeg"}.get(type(exc), "error")  # fmt: skip
        console.print(Panel(escape(str(exc)), title=f"PullReel failed ({title})", border_style="red"))
        raise typer.Exit(1) from exc


def _human_size(path: Path) -> str:
    n = path.stat().st_size
    return f"{n / 1_048_576:.2f} MB" if n >= 1_048_576 else f"{n / 1024:.0f} KB"


def _diff_table(diff: DiffSummary) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column()
    t.add_row("Range", escape(f"{diff.base_ref} -> {diff.head_ref}"))
    ui = diff.ui_files
    t.add_row("Files", f"{len(diff.files)} changed, {len(ui)} UI/route")
    for label, items in (("Routes", diff.routes), ("Components", diff.components), ("Labels", diff.labels),
                         ("Test ids", diff.test_ids)):  # fmt: skip
        if items:
            shown = ", ".join(items[:8]) + (f" (+{len(items) - 8})" if len(items) > 8 else "")
            t.add_row(label, escape(shown))
    return t


def _storyboard_table(sb: Storyboard) -> Table:
    t = Table(title=escape(sb.feature_summary), title_style="bold", header_style="bold magenta", expand=False)
    t.add_column("#", justify="right")
    t.add_column("action")
    t.add_column("target")
    t.add_column("input")
    t.add_column("camera")
    t.add_column("dwell", justify="right")
    for i, s in enumerate(sb.steps):
        cam = s.camera.mode if s.camera.mode == "wide" else f"{s.camera.mode} {s.camera.zoom:.2f}x"
        t.add_row(str(i), s.action, escape(s.locator.describe()) if s.locator else "-", escape(s.input_value or ""),
                  cam, f"{s.dwell_after_ms} ms")  # fmt: skip
    return t


def _load_diff(repo: Path, base: str) -> DiffSummary:
    with console.status(f"[bold]Reading git diff against {base}..."):
        diff = analyze_diff(repo, base)
    if diff.is_empty:
        raise PlanError(f"no changes found between {base} and the working tree of {repo.resolve()}")
    console.print(Panel(_diff_table(diff), title="[bold]Diff", border_style="cyan"))
    if not diff.ui_files:
        console.print("[yellow]No UI files in this diff - the storyboard will lean on the page snapshot.[/]")
    return diff


def _plan(url: str, diff: DiffSummary, model: str, base_url: str | None, api_key: str | None, headed: bool) -> PlanResult:
    cfg = parse_model_spec(model, base_url=base_url, api_key=api_key)
    t0 = time.perf_counter()
    with console.status(f"[bold]Directing with {cfg.model}...") as status:
        res = plan_storyboard_sync(
            url, diff, cfg, headed=headed, progress=lambda m: status.update(f"[bold]Planning[/] - {m}")
        )
    console.print(_storyboard_table(res.storyboard))
    extra = f", {res.attempts} attempts" if res.attempts > 1 else ""
    console.print(f"[dim]planned in {time.perf_counter() - t0:.1f}s with {cfg.label}{extra}[/]")
    for note in res.notes:
        console.print(f"[dim]  - {escape(note)}[/]")
    return res


def _start_url(url: str, sb: Storyboard) -> str:
    """Apply the storyboard's start_path to an http(s) base URL; file:// URLs are used as-is."""
    if urlparse(url).scheme in ("http", "https") and sb.start_path not in ("", "/"):
        return urljoin(url, sb.start_path)
    return url


def _record(url: str, sb: Storyboard, work: Path, *, headed: bool, seed: int) -> Recording:
    with console.status("[bold]Recording in headless Chromium (Retina 2x)...") as status:
        return record_sync(url, sb, work, headed=headed, seed=seed,
                           progress=lambda m: status.update(f"[bold]Recording[/] - {m}"))  # fmt: skip


def render(recording: Recording, mp4: Path, *, gif: Path | None, theme: StudioTheme) -> dict[str, str]:
    engine = CinemaEngine(recording, theme)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Rendering 60 fps", total=engine.frame_count)
        export_mp4(engine.frames(), engine.frame_size, theme.output_fps, mp4, crf=theme.crf,
                   on_frame=lambda: progress.advance(task))  # fmt: skip
    info: dict[str, str] = {
        "MP4": f"{mp4}  ({_human_size(mp4)})",
        "Stream": probe(mp4),
        "Frames": f"{engine.frame_count} @ {theme.output_fps} fps ({recording.duration:.2f}s)",
        "Source captures": f"{len(recording.frames)} "
        f"({len(recording.frames) / max(recording.duration, 1e-6):.1f}/s at 2560x1600)",
    }
    if engine.last_frame is not None:
        info["Poster"] = str(export_poster(engine.last_frame, mp4.with_name(f"{mp4.stem}-poster.png")))
    if gif is not None:
        with console.status("[bold]Encoding GIF (2-pass palette)..."):
            g = export_gif(mp4, gif, fps=theme.gif_fps, max_mb=theme.gif_max_mb)
        flag = "" if g.fits else f"  [yellow](over {theme.gif_max_mb:g} MB budget)[/]"
        info["GIF"] = f"{g.path}  ({g.size_mb:.2f} MB, {g.width}px, {g.fps} fps){flag}"
    return info


def _outputs(out: Path, gif: bool | None) -> tuple[Path, Path | None]:
    """`--out demo.gif` -> (demo.mp4, demo.gif); `--out demo.mp4` -> (demo.mp4, gif only if --gif)."""
    if out.suffix.lower() not in (".gif", ".mp4"):
        out = out / "demo.gif" if (out.is_dir() or not out.suffix) else out.with_suffix(".gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = out.with_suffix(".mp4")
    want_gif = gif if gif is not None else out.suffix.lower() == ".gif"
    return mp4, (out.with_suffix(".gif") if want_gif else None)


def _summary(info: dict[str, str], t0: float, extra: dict[str, str] | None = None) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold magenta", no_wrap=True)
    table.add_column(overflow="fold")
    for k, v in {**info, **(extra or {})}.items():
        table.add_row(k, v)
    table.add_row("Time", f"{time.perf_counter() - t0:.1f}s total")
    console.print(Panel(table, title="[bold green]Demo ready", border_style="green"))


@contextmanager
def _workdir(keep: bool) -> Iterator[Path]:
    work = Path(tempfile.mkdtemp(prefix="pullreel-"))
    try:
        yield work
    finally:
        if keep:
            console.print(f"[dim]raw frames + recording.json kept in {work}  (re-render: pullreel render {work})[/]")
        else:
            shutil.rmtree(work, ignore_errors=True)


# Commands ----------------------------------------------------------------------------------------------------
@app.command()
def record(
    url: UrlOpt,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output .gif or .mp4 (both are written for .gif).")] = Path(
        "pr-demo.gif"
    ),
    repo: RepoOpt = Path("."),
    base: BaseOpt = "main",
    model: ModelOpt = DEFAULT_MODEL,
    base_url: BaseUrlOpt = None,
    api_key: ApiKeyOpt = None,
    storyboard: Annotated[
        Path | None,
        typer.Option("--storyboard", "-s", help="Use this storyboard JSON (from `pullreel plan`) instead of the LLM.",
                     exists=True, dir_okay=False),
    ] = None,  # fmt: skip
    gif: Annotated[bool | None, typer.Option("--gif/--no-gif", help="Force GIF on/off (default: from --out).")] = None,
    seed: Annotated[int, typer.Option(help="Random seed for the human mouse paths.")] = 7,
    headed: HeadedOpt = False,
    keep_frames: Annotated[bool, typer.Option(help="Keep raw frames + recording.json for `pullreel render`.")] = False,
) -> None:
    """Diff -> storyboard (LLM) -> human-paced recording -> cinematic MP4 + GIF for your PR."""
    t0 = time.perf_counter()
    with _friendly_errors():
        find_ffmpeg()
        mp4, gif_path = _outputs(out, gif)
        if storyboard is not None:
            sb = Storyboard.model_validate_json(storyboard.read_text(encoding="utf-8"))
            console.print(_storyboard_table(sb))
            start = _start_url(url, sb)
        else:
            diff = _load_diff(repo, base)
            plan = _plan(url, diff, model, base_url, api_key, headed)
            sb, start = plan.storyboard, plan.url
        sb_path = mp4.with_name(f"{mp4.stem}.storyboard.json")
        if storyboard is None or storyboard.resolve() != sb_path.resolve():
            sb_path.write_text(sb.model_dump_json(indent=2), encoding="utf-8")
        with _workdir(keep_frames) as work:
            rec = _record(start, sb, work, headed=headed, seed=seed)
            info = render(rec, mp4, gif=gif_path, theme=StudioTheme())
        _summary(info, t0, {"Storyboard": f"{sb_path}  (edit it, then: pullreel record -s {sb_path.name} ...)"})


@app.command()
def plan(
    url: UrlOpt,
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the storyboard JSON.")] = Path(
        "storyboard.json"
    ),
    repo: RepoOpt = Path("."),
    base: BaseOpt = "main",
    model: ModelOpt = DEFAULT_MODEL,
    base_url: BaseUrlOpt = None,
    api_key: ApiKeyOpt = None,
    headed: HeadedOpt = False,
) -> None:
    """Dry run: write the storyboard JSON without recording, so you can review or edit it."""
    with _friendly_errors():
        diff = _load_diff(repo, base)
        res = _plan(url, diff, model, base_url, api_key, headed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res.storyboard.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Storyboard written to {out}[/]  ->  [bold]pullreel record --url {url} -s {out}[/]")


@app.command("render")
def render_cmd(
    recording: Annotated[Path, typer.Argument(help="recording.json, or the folder kept by --keep-frames.", exists=True)],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output .gif or .mp4.")] = Path("pr-demo.gif"),
    gif: Annotated[bool | None, typer.Option("--gif/--no-gif")] = None,
) -> None:
    """Re-render a kept recording (new theme/camera settings) without driving the browser again."""
    t0 = time.perf_counter()
    path = recording / "recording.json" if recording.is_dir() else recording
    with _friendly_errors():
        find_ffmpeg()
        rec = Recording.load_json(path)
        mp4, gif_path = _outputs(out, gif)
        info = render(rec, mp4, gif=gif_path, theme=StudioTheme())
    _summary(info, t0)


@app.command()
def diff(repo: RepoOpt = Path("."), base: BaseOpt = "main",
         prompt: Annotated[bool, typer.Option(help="Print the exact text sent to the LLM.")] = False) -> None:  # fmt: skip
    """Show what PullReel understood from your branch (no browser, no LLM)."""
    with _friendly_errors():
        summary = _load_diff(repo, base)
    if prompt:
        console.print(Panel(escape(summary.to_prompt()), title="GIT_DIFF_SUMMARY", border_style="dim"))


def demo_storyboard() -> Storyboard:
    """Hand-written storyboard for the bundled demo page (what the LLM produces in `record`)."""
    return Storyboard(
        feature_summary="Add a task from the composer and complete it",
        start_path="/",
        steps=[
            ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=400),
            ActionStep(
                action="fill",
                locator=Locator(strategy="label", value="New task"),
                input_value="Ship dark mode",
                camera=Camera(mode="focus", zoom=1.8),
                dwell_after_ms=250,
            ),
            ActionStep(
                action="click",
                locator=Locator(strategy="role", value="button", name="Add task"),
                camera=Camera(mode="focus", zoom=1.3),
                dwell_after_ms=750,
            ),
            ActionStep(
                action="click",
                locator=Locator(strategy="role", value="checkbox", name="Complete Ship dark mode"),
                camera=Camera(mode="focus", zoom=1.7),
                dwell_after_ms=550,
            ),
            ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1100),
        ],
    )


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("pullreel-out"),
    headed: HeadedOpt = False,
    gif: Annotated[bool, typer.Option("--gif/--no-gif", help="Also export a GitHub-ready GIF.")] = True,
    seed: Annotated[int, typer.Option(help="Random seed for the human mouse paths.")] = 7,
    keep_frames: Annotated[bool, typer.Option(help="Keep raw screencast frames + recording.json.")] = False,
) -> None:
    """Record the bundled demo page and render a cinematic MP4 + GIF. No LLM, no config."""
    t0 = time.perf_counter()
    console.print(Panel.fit("[bold]PullReel demo[/]  built-in page -> human run -> spring camera -> studio render",
                            border_style="magenta"))  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    with _friendly_errors():
        find_ffmpeg()
        page = resources.files("pullreel.assets").joinpath("demo.html")
        with resources.as_file(page) as html, _workdir(keep_frames) as work:
            rec = _record(html.resolve().as_uri(), demo_storyboard(), work, headed=headed, seed=seed)
            info = render(rec, out / "demo.mp4", gif=(out / "demo.gif") if gif else None, theme=StudioTheme())
    _summary(info, t0)
