"""PullReel command line: `record`, `plan`, `render`, `diff` and the zero-config `demo`."""

from __future__ import annotations

import asyncio
import re
import shutil
import sys
import unicodedata
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from importlib import resources
from pathlib import Path
from typing import Annotated
from urllib.parse import urljoin, urlparse

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from pullreel import __version__, wizard
from pullreel.auth import default_auth_path, login_sync, resolve_auth
from pullreel.cinema_engine import CinemaEngine
from pullreel.diff_analyzer import GitError, analyze_diff
from pullreel.exporter import ExportError, GifResult, export_gif, export_mp4, export_poster, find_ffmpeg, probe
from pullreel.human_runner import StepError, record_sync
from pullreel.models import ActionStep, Camera, DiffSummary, Locator, Recording, Storyboard, StudioTheme
from pullreel.progress import ProgressEvent, Reporter, Stage, say, silent
from pullreel.progress_ui import PipelineView
import httpx

from pullreel.storyboard_agent import (
    PROVIDERS,
    ScoutPick,
    plan_storyboards_sync,
    provider_key,
    scout_features,
    LLMConfig,
    LLMError,
    LoginRequiredError,
    PlanError,
    PlanResult,
    parse_model_spec,
)

app = typer.Typer(
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
    help="[bold]PullReel[/] - turn a git diff into a cinematic UI demo for your Pull Request.",
)

# Labels from real code contain arrows, emoji, accents... A legacy Windows code page (cp1252) cannot encode them and
# would crash the whole command on print. Degrade those characters to "?" instead of failing.
for _stream in (sys.stdout, sys.stderr):
    with suppress(AttributeError, ValueError):
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]

console = Console()

DEFAULT_MODEL = "ollama/qwen2.5-coder:7b"
KNOWN_ERRORS = (StepError, ExportError, GitError, LLMError, PlanError, RuntimeError, ValueError, FileNotFoundError)

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
AuthOpt = Annotated[
    Path | None,
    typer.Option("--auth", help="Saved session from `pullreel login` (default: <repo>/.pullreel/auth.json if it exists).",
                 dir_okay=False),
]


def _print_version(value: bool) -> None:
    if value:
        console.print(f"pullreel {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the version and exit.", callback=_print_version, is_eager=True),
    ] = False,
) -> None:
    """PullReel - turn a git diff into a cinematic UI demo for your Pull Request.

    Run it with no command for the interactive assistant.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(start)


# Helpers -----------------------------------------------------------------------------------------------------
@contextmanager
def _friendly_errors() -> Iterator[None]:
    try:
        yield
    except KNOWN_ERRORS as exc:
        title = {GitError: "git", LLMError: "LLM", PlanError: "storyboard", StepError: "browser",
                 LoginRequiredError: "login required", FileNotFoundError: "missing file",
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


def _diff_detail(diff: DiffSummary) -> str:
    parts = [f"{len(diff.files)} files", f"{len(diff.ui_files)} UI"]
    if diff.routes:
        parts.append(f"{len(diff.routes)} routes (first: {diff.routes[0]})")
    return " · ".join(parts)


def _load_diff(repo: Path, base: str, report: Reporter = silent) -> DiffSummary:
    say(report, "diff", "start", f"git diff against {base}")
    diff = analyze_diff(repo, base)
    if diff.is_empty:
        say(report, "diff", "fail", "no changes")
        raise PlanError(f"no changes found between {base} and the working tree of {repo.resolve()}")
    say(report, "diff", "done", _diff_detail(diff))
    return diff


def _print_plan(res: PlanResult, cfg: LLMConfig) -> None:
    console.print(_storyboard_table(res.storyboard))
    extra = f", {res.attempts} attempts" if res.attempts > 1 else ""
    console.print(f"[dim]directed by {escape(cfg.label)}{extra}[/]")
    for note in res.notes:
        console.print(f"[dim]  - {escape(note)}[/]")


def _start_url(url: str, sb: Storyboard) -> str:
    """Apply the storyboard's start_path to an http(s) base URL; file:// URLs are used as-is."""
    if urlparse(url).scheme in ("http", "https") and sb.start_path not in ("", "/"):
        return urljoin(url, sb.start_path)
    return url


def render(
    recording: Recording, mp4: Path, *, gif: Path | None, theme: StudioTheme, report: Reporter = silent
) -> dict[str, str]:
    engine = CinemaEngine(recording, theme)
    total = engine.frame_count
    say(report, "render", "start", f"spring camera · cursor · studio frame -> {mp4.name}", 0, total)
    done = [0]

    def on_frame() -> None:
        done[0] += 1
        if done[0] % 6 == 0 or done[0] == total:
            say(report, "render", detail=f"frame {done[0]}/{total} · {mp4.name}", completed=done[0], total=total)

    export_mp4(engine.frames(), engine.frame_size, theme.output_fps, mp4, crf=theme.crf, on_frame=on_frame)
    say(report, "render", "done", f"{mp4.name} · {_human_size(mp4)} · {total} frames @ {theme.output_fps} fps")
    info: dict[str, str] = {
        "MP4": f"{mp4}  ({_human_size(mp4)})",
        "Stream": probe(mp4),
        "Frames": f"{total} @ {theme.output_fps} fps ({recording.duration:.2f}s)",
        "Source captures": f"{len(recording.frames)} "
        f"({len(recording.frames) / max(recording.duration, 1e-6):.1f}/s at 2560x1600)",
    }
    if engine.last_frame is not None:
        info["Poster"] = str(export_poster(engine.last_frame, mp4.with_name(f"{mp4.stem}-poster.png")))
    if gif is not None:
        say(report, "gif", "start", "two-pass palette")

        def on_pass(attempt: int, width: int, fps: int, result: GifResult | None) -> None:
            if result is None:
                say(report, "gif", detail=f"pass {attempt} · {width}px {fps} fps · encoding")
            elif not result.fits:
                say(report, "gif", detail=f"pass {attempt} · {width}px -> {result.size_mb:.1f} MB, over "
                    f"{theme.gif_max_mb:g} MB: shrinking")  # fmt: skip

        g = export_gif(mp4, gif, fps=theme.gif_fps, max_mb=theme.gif_max_mb, on_pass=on_pass)
        say(report, "gif", "done", f"{gif.name} · {g.size_mb:.2f} MB · {g.width}px · {g.fps} fps")
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


PLAN_STAGES: list[Stage] = ["diff", "scout", "snapshot", "storyboard", "rehearsal"]
RECORD_STAGES: list[Stage] = [*PLAN_STAGES, "recording", "render", "gif"]

ScenariosOpt = Annotated[
    str, typer.Option("--scenarios", "-n", help="How many features to film (one video each), or 'all'.")
]
DirectorOpt = Annotated[
    str,
    typer.Option("--director", help="Who writes each story: auto (AI, replaced by an auto-composed story when it is "
                 "thin), ai (AI only) or composer (explore the feature and compose, no AI writing - fastest)."),
]


def _count(value: str, diff: DiffSummary) -> int:
    if value.strip().lower() == "all":
        return max(1, min(8, len(diff.features)))
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"--scenarios must be a number or 'all', got {value!r}") from exc
    return max(1, min(8, n))


def _director(value: str) -> str:
    v = value.strip().lower()
    if v not in ("auto", "ai", "composer"):
        raise ValueError(f"--director must be auto, ai or composer, got {value!r}")
    return v
FocusOpt = Annotated[
    str | None,
    typer.Option("--focus", "-f", help='Film this feature instead of letting the scout choose, e.g. "global search".'),
]


def _numbered(path: Path, k: int, n: int) -> Path:
    return path if n == 1 else path.with_name(f"{path.stem}-{k}{path.suffix}")


def _scoped(report: Reporter, k: int, n: int) -> Reporter:
    if n == 1:
        return report

    def inner(ev: ProgressEvent) -> None:
        detail = f"[{k}/{n}] {ev.detail}" if ev.detail else ev.detail
        report(ProgressEvent(ev.stage, ev.state, detail, ev.completed, ev.total))

    return inner


def _print_scenario(k: int, n: int, pick: ScoutPick, res: PlanResult, cfg: LLMConfig) -> None:
    if n > 1 or pick.why:
        head = f"[bold magenta]Scenario {k}/{n}[/] [bold]{escape(pick.title)}[/]" if n > 1 else f"[bold]{escape(pick.title)}[/]"
        console.print(head + (f"  [dim]{escape(pick.why)}[/]" if pick.why else ""))
    _print_plan(res, cfg)


Planned = tuple[ScoutPick | None, PlanResult | None, Storyboard, str]
PathsFn = Callable[[int, int, ScoutPick | None], tuple[Path, Path | None]]


def _slug(text: str, limit: int = 32) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower())
    return s.strip("-")[:limit].strip("-") or "scenario"


def _shoot(planned: list[Planned], paths: PathsFn, view: PipelineView, *, headed: bool, seed: int, keep_frames: bool,
           session: Path | None, source: Path | None = None) -> list[tuple[str, dict[str, str], Path]]:  # fmt: skip
    """Record + render every planned storyboard; returns (title, info, storyboard path) per video."""
    results: list[tuple[str, dict[str, str], Path]] = []
    n = len(planned)
    for k, (pick, _res, sb, start) in enumerate(planned, 1):
        k_mp4, k_gif = paths(k, n, pick)
        sb_path = k_mp4.with_name(f"{k_mp4.stem}.storyboard.json")
        if source is None or source.resolve() != sb_path.resolve():
            sb_path.write_text(sb.model_dump_json(indent=2), encoding="utf-8")
        rep = _scoped(view.report, k, n)
        with _workdir(keep_frames) as work:
            rec = record_sync(start, sb, work, headed=headed, seed=seed, report=rep, auth=session)
            info = render(rec, k_mp4, gif=k_gif, theme=StudioTheme(), report=rep)
        results.append((pick.title if pick else sb.feature_summary, info, sb_path))
    return results


def _report_results(planned: list[Planned], results: list[tuple[str, dict[str, str], Path]], cfg: LLMConfig | None,
                    t0: float) -> None:  # fmt: skip
    for k, (pick, res, sb, _start) in enumerate(planned, 1):
        if pick is not None and res is not None and cfg is not None:
            _print_scenario(k, len(planned), pick, res, cfg)
        else:
            console.print(_storyboard_table(sb))
    if len(results) == 1:
        _title, info, sb_path = results[0]
        _summary(info, t0, {"Storyboard": f"{sb_path}  (edit it, then: pullreel record -s {sb_path.name} ...)"})
    else:
        merged: dict[str, str] = {}
        for k, (title, info, sb_path) in enumerate(results, 1):
            files = " · ".join(v for key, v in info.items() if key in ("MP4", "GIF"))
            merged[f"{k}. {title}"[:28]] = f"{files}  · storyboard {sb_path.name}"
        _summary(merged, t0)


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
    scenarios: ScenariosOpt = "1",
    focus: FocusOpt = None,
    director: DirectorOpt = "auto",
    storyboard: Annotated[
        Path | None,
        typer.Option("--storyboard", "-s", help="Use this storyboard JSON (from `pullreel plan`) instead of the LLM.",
                     exists=True, dir_okay=False),
    ] = None,  # fmt: skip
    gif: Annotated[bool | None, typer.Option("--gif/--no-gif", help="Force GIF on/off (default: from --out).")] = None,
    seed: Annotated[int, typer.Option(help="Random seed for the human mouse paths.")] = 7,
    headed: HeadedOpt = False,
    keep_frames: Annotated[bool, typer.Option(help="Keep raw frames + recording.json for `pullreel render`.")] = False,
    auth: AuthOpt = None,
) -> None:
    """Diff -> scout -> storyboard(s) -> human-paced recording -> cinematic MP4 + GIF (non-interactive)."""
    t0 = time.perf_counter()
    with _friendly_errors():
        find_ffmpeg()
        session = resolve_auth(repo, auth)
        mp4, gif_path = _outputs(out, gif)
        planned: list[Planned] = []
        cfg: LLMConfig | None = None
        with PipelineView(console, RECORD_STAGES, f"record · {repo.resolve().name} · {url}") as view:
            if gif_path is None:
                view.skip("gif", "--no-gif")
            if storyboard is not None:
                sb = Storyboard.model_validate_json(storyboard.read_text(encoding="utf-8"))
                for stage in PLAN_STAGES:
                    view.skip(stage, f"from {storyboard.name}")
                planned.append((None, None, sb, _start_url(url, sb)))
            else:
                cfg = parse_model_spec(model, base_url=base_url, api_key=api_key)
                diff = _load_diff(repo, base, view.report)
                view.title = f"record · {repo.resolve().name} ({diff.head_ref} vs {diff.base_ref})"
                plans = plan_storyboards_sync(url, diff, cfg, scenarios=_count(scenarios, diff), focus=focus,
                                              headed=headed, report=view.report, auth=session,
                                              director=_director(director))  # fmt: skip
                planned += [(pick, res, res.storyboard, res.url) for pick, res in plans]

            def numbered(k: int, n: int, _pick: ScoutPick | None) -> tuple[Path, Path | None]:
                return _numbered(mp4, k, n), (_numbered(gif_path, k, n) if gif_path else None)

            results = _shoot(planned, numbered, view, headed=headed, seed=seed, keep_frames=keep_frames,
                             session=session, source=storyboard)  # fmt: skip
        _report_results(planned, results, cfg, t0)


@app.command()
def start(
    out: Annotated[Path, typer.Option("--out", "-o", help="Folder for the videos.")] = Path("pullreel-out"),
    gif: Annotated[bool, typer.Option("--gif/--no-gif", help="Also export GitHub-ready GIFs.")] = True,
    headed: HeadedOpt = False,
    keep_frames: Annotated[bool, typer.Option(help="Keep raw frames for `pullreel render`.")] = False,
    seed: Annotated[int, typer.Option(help="Random seed for the human mouse paths.")] = 7,
) -> None:
    """Interactive assistant (also what plain `pullreel` runs): pick the AI, review every scenario, film your picks."""
    t0 = time.perf_counter()
    console.print(Panel.fit("[bold magenta]PullReel[/]  let's turn this branch into demo videos", border_style="magenta"))
    try:
        cfg = wizard.ask_ai(console.print)
        director_mode = wizard.ask_director(cfg)
        url = wizard.ask_url()
        repo, base = wizard.ask_repo_and_base(Path.cwd())
    except wizard.Cancelled:
        console.print("[dim]Cancelled.[/]")
        raise typer.Exit(1) from None
    with _friendly_errors():
        find_ffmpeg()
        session = resolve_auth(repo, None)
        with PipelineView(console, ["diff", "scout"], f"scanning {repo.resolve().name} · {cfg.label}") as view:
            diff = _load_diff(repo, base, view.report)
            view.title = f"scanning {repo.resolve().name} ({diff.head_ref} vs {diff.base_ref}) · {cfg.label}"
            n = min(6, max(1, len(diff.features)))
            recommended = asyncio.run(scout_features(diff, cfg, n, view.report))
    try:
        picks = wizard.ask_scenarios(recommended, diff)
    except wizard.Cancelled:
        console.print("[dim]Cancelled.[/]")
        raise typer.Exit(1) from None
    out.mkdir(parents=True, exist_ok=True)

    def by_title(k: int, n: int, pick: ScoutPick | None) -> tuple[Path, Path | None]:
        stem = f"{k:02d}-{_slug(pick.title if pick else 'scenario')}"
        return out / f"{stem}.mp4", (out / f"{stem}.gif") if gif else None

    stages: list[Stage] = ["snapshot", "storyboard", "rehearsal", "recording", "render", "gif"]
    with _friendly_errors():
        while True:
            try:
                with PipelineView(console, stages, f"filming {len(picks)} scenario(s) · {cfg.label}") as view:
                    if not gif:
                        view.skip("gif", "--no-gif")
                    plans = plan_storyboards_sync(url, diff, cfg, picks=picks, headed=headed, report=view.report,
                                                  auth=session, director=director_mode)  # fmt: skip
                    planned: list[Planned] = [(p, r, r.storyboard, r.url) for p, r in plans]
                    results = _shoot(planned, by_title, view, headed=headed, seed=seed, keep_frames=keep_frames,
                                     session=session)  # fmt: skip
                break
            except LoginRequiredError as exc:
                console.print(Panel(escape(str(exc)), title="login required", border_style="yellow"))
                try:
                    again = wizard.confirm("Sign in now in a browser window? (your password never reaches PullReel)")
                except wizard.Cancelled:
                    again = False
                if not again:
                    raise
                origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                console.print("[bold]Sign in in the window that opens, then press Enter here.[/]")
                session = login_sync(origin, default_auth_path(repo)).path
        _report_results(planned, results, cfg, t0)


@app.command()
def plan(
    url: UrlOpt,
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the storyboard JSON (numbered if -n > 1).")] = Path(
        "storyboard.json"
    ),
    repo: RepoOpt = Path("."),
    base: BaseOpt = "main",
    model: ModelOpt = DEFAULT_MODEL,
    base_url: BaseUrlOpt = None,
    api_key: ApiKeyOpt = None,
    scenarios: ScenariosOpt = "1",
    focus: FocusOpt = None,
    director: DirectorOpt = "auto",
    headed: HeadedOpt = False,
    auth: AuthOpt = None,
) -> None:
    """Dry run: scout the branch and write storyboard JSON(s) without recording, to review or edit them."""
    with _friendly_errors():
        cfg = parse_model_spec(model, base_url=base_url, api_key=api_key)
        session = resolve_auth(repo, auth)
        with PipelineView(console, PLAN_STAGES, f"plan · {repo.resolve().name} · {url}") as view:
            diff = _load_diff(repo, base, view.report)
            plans = plan_storyboards_sync(url, diff, cfg, scenarios=_count(scenarios, diff), focus=focus,
                                          headed=headed, report=view.report, auth=session,
                                          director=_director(director))  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for k, (pick, res) in enumerate(plans, 1):
            path = _numbered(out, k, len(plans))
            path.write_text(res.storyboard.model_dump_json(indent=2), encoding="utf-8")
            written.append(path)
            _print_scenario(k, len(plans), pick, res, cfg)
    for path in written:
        console.print(f"[green]Storyboard written to {path}[/]  ->  [bold]pullreel record --url {url} -s {path}[/]")


@app.command()
def models() -> None:
    """Show which AI providers PullReel can use right now (API keys are detected, never displayed)."""
    table = Table(header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("provider", style="bold")
    table.add_column("--model")
    table.add_column("status")
    table.add_column("notes", style="dim")
    for p in PROVIDERS.values():
        spec = f"{p.name}/{p.default_model}" if p.default_model else f"{p.name}/<model>"
        if p.local:
            status = _local_status(p.base_url)
        else:
            _key, env = provider_key(p)
            status = f"[green]ready[/] (key in {env})" if env else f"[dim]set {' or '.join(p.key_envs)}[/]"
        table.add_row(p.name, escape(spec), status, escape(p.note))
    console.print(table)
    console.print(
        "\n[dim]Set a key for this terminal:[/]  [bold]$env:ANTHROPIC_API_KEY = \"sk-ant-...\"[/]  "
        "[dim](PowerShell) or[/] [bold]export ANTHROPIC_API_KEY=...[/] [dim](bash), or pass --api-key.\n"
        "Pick the default model once:[/] [bold]PULLREEL_MODEL=anthropic[/]  [dim]- keys are only read from your "
        "environment, never written to disk by PullReel.[/]"
    )


def _local_status(base_url: str) -> str:
    root = base_url.removesuffix("/v1")
    try:
        if "11434" in base_url:
            names = [m["name"] for m in httpx.get(f"{root}/api/tags", timeout=2).json().get("models", [])]
        else:
            names = [m["id"] for m in httpx.get(f"{base_url}/models", timeout=2).json().get("data", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return "[dim]not running[/]"
    if not names:
        return "[yellow]running, no model installed[/]"
    shown = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
    return f"[green]running[/] ({escape(shown)})"


@app.command()
def login(
    url: UrlOpt,
    repo: RepoOpt = Path("."),
    auth: Annotated[Path | None, typer.Option("--auth", help="Where to save the session (default: <repo>/.pullreel/auth.json).")] = None,
) -> None:
    """Sign in once in a real browser window; PullReel keeps only the session so `record` films you signed in."""
    path = auth or default_auth_path(repo)
    console.print(Panel(
        f"A browser window opens on [bold]{escape(url)}[/].\n"
        "1. Sign in exactly as usual (password, Google, SSO...).\n"
        "2. Once you see your app, come back here and press [bold]Enter[/] (or simply close the window).\n\n"
        f"[dim]PullReel never sees your password: it only saves the session (cookies + storage) to {escape(str(path))}, "
        "in a folder that git ignores. Delete that file to sign out.[/]",
        title="[bold]pullreel login", border_style="magenta"))  # fmt: skip
    with _friendly_errors():
        res = login_sync(url, path)
    if res.still_on_login:
        console.print(f"[yellow]Saved {res.path}, but the page still showed a login form - if `record` lands on the "
                      "login screen, run `pullreel login` again and finish signing in first.[/]")  # fmt: skip
    else:
        console.print(f"[green]Session saved[/] to {escape(str(res.path))} ({res.cookies} cookies, {res.origins} "
                      "origin(s) with storage). `pullreel record` and `pullreel plan` will use it automatically.")  # fmt: skip


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
        with PipelineView(console, ["render", "gif"], f"render · {path.parent.name}") as view:
            if gif_path is None:
                view.skip("gif", "--no-gif")
            info = render(rec, mp4, gif=gif_path, theme=StudioTheme(), report=view.report)
    _summary(info, t0)


@app.command()
def diff(repo: RepoOpt = Path("."), base: BaseOpt = "main",
         prompt: Annotated[bool, typer.Option(help="Print the exact text sent to the LLM.")] = False) -> None:  # fmt: skip
    """Show what PullReel understood from your branch (no browser, no LLM)."""
    with _friendly_errors():
        with console.status(f"[bold]Reading git diff against {base}..."):
            summary = _load_diff(repo, base)
    console.print(Panel(_diff_table(summary), title="[bold]Diff", border_style="cyan"))
    if not summary.ui_files:
        console.print("[yellow]No UI files in this diff - the storyboard will lean on the page snapshot.[/]")
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
    out.mkdir(parents=True, exist_ok=True)
    with _friendly_errors():
        find_ffmpeg()
        page = resources.files("pullreel.assets").joinpath("demo.html")
        with resources.as_file(page) as html, _workdir(keep_frames) as work:
            with PipelineView(console, ["recording", "render", "gif"], "demo · bundled page, no LLM") as view:
                if not gif:
                    view.skip("gif", "--no-gif")
                rec = record_sync(html.resolve().as_uri(), demo_storyboard(), work, headed=headed, seed=seed,
                                  report=view.report)  # fmt: skip
                info = render(rec, out / "demo.mp4", gif=(out / "demo.gif") if gif else None, theme=StudioTheme(),
                              report=view.report)  # fmt: skip
    _summary(info, t0)
