"""Interactive prompts for `pullreel` (no arguments): pick the AI, the app URL and the scenarios to film.

The pure helpers (choice building, dev-server detection, base branch) are separated from the questionary prompts so
they can be tested without a terminal.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
import questionary
from questionary import Choice, Separator, Style

from pullreel.config import load_config, remember
from pullreel.diff_analyzer import GitError, git
from pullreel.models import DiffSummary
from pullreel.storyboard_agent import PROVIDERS, LLMConfig, LLMError, ScoutPick, match_focus, parse_model_spec, provider_key

STYLE = Style([
    ("qmark", "fg:#c678dd bold"), ("question", "bold"), ("pointer", "fg:#c678dd bold"),
    ("highlighted", "fg:#c678dd bold"), ("selected", "fg:#98c379"), ("answer", "fg:#c678dd bold"),
    ("instruction", "fg:#7f848e italic"), ("separator", "fg:#5c6370"),
])  # fmt: skip

WRITE_OWN = "__write_own__"
DEV_PORTS = (5173, 3000, 5174, 4173, 8080, 4200, 8000, 4321, 5000, 3001, 8888)
CLOUD = ("anthropic", "openai", "gemini", "mistral", "deepseek", "openrouter", "groq")


class Cancelled(Exception):
    """The user pressed Ctrl-C / Esc in a prompt."""


def _ask(question: questionary.Question) -> object:
    answer = question.ask()  # returns None on Ctrl-C
    if answer is None:
        raise Cancelled
    return answer


# ---------------------------------------------------------------------------------------------------- pure helpers
def detect_dev_servers(ports: Iterable[int] = DEV_PORTS, timeout: float = 0.25) -> list[str]:
    """Local dev servers answering on the usual ports."""
    found: list[str] = []
    for port in ports:
        for host in ("127.0.0.1", "::1"):
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                if sock.connect_ex((host, port)) == 0:
                    found.append(f"http://localhost:{port}/")
                    break
    return found


def detect_base_branch(repo: Path) -> str:
    for ref in ("main", "master", "origin/main", "origin/master", "develop"):
        try:
            git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
            return ref
        except GitError:
            continue
    return "main"


def is_git_repo(path: Path) -> bool:
    try:
        git(path, "rev-parse", "--is-inside-work-tree")
        return True
    except (GitError, OSError):
        return False


@dataclass(slots=True)
class ScenarioChoice:
    label: str
    pick: ScoutPick | None  # None = "write my own"
    checked: bool


def build_scenario_choices(recommended: list[ScoutPick], diff: DiffSummary) -> list[ScenarioChoice]:
    """Scout picks first (★, all pre-checked), then every other candidate, then "write my own"."""
    out: list[ScenarioChoice] = []
    taken: set[str] = set()
    for p in recommended:
        star = "★" if p.recommended else " "
        where = f"  {p.route}" if p.route else ""
        why = f"  — {p.why}" if p.why else ""
        out.append(ScenarioChoice(f"{star} {p.title}{where}{why}", p, checked=p.recommended))
        taken.add((p.route or p.title).lower())
    for f in diff.features:
        key = (f.route or f.title).lower()
        if key in taken:
            continue
        taken.add(key)
        hint = f.commits[0] if f.commits else f"+{f.added} lines"
        where = f"  {f.route}" if f.route else ""
        pick = ScoutPick(title=f.title, route=f.route, why=hint, labels=f.labels[:5], recommended=False)
        out.append(ScenarioChoice(f"  {f.title}{where}  — {hint}", pick, checked=False))
    out.append(ScenarioChoice("✍  Write my own scenario…", None, checked=False))
    return out


def custom_pick(diff: DiffSummary, text: str) -> ScoutPick:
    pick = match_focus(diff, text)
    pick.why = f"user request: {text}"
    return pick


# ---------------------------------------------------------------------------------------------------- prompts
def ask_ai(say: Callable[[str], None]) -> LLMConfig:
    """Local model or cloud API? Remembers the choice; keys are saved only with explicit consent."""
    last = load_config().get("model")
    if isinstance(last, str) and last:
        if _ask(questionary.confirm(f"Use your last AI setup ({last})?", default=True, style=STYLE)):
            try:
                return parse_model_spec(last)
            except LLMError as exc:
                say(f"[yellow]{exc}[/] - let's set it up again.")
    kind = _ask(questionary.select(
        "Which AI should direct the demos?",
        choices=[Choice("Local AI on this computer (Ollama - free, private, slower)", "local"),
                 Choice("Cloud AI with an API key (Claude, OpenAI, Gemini, Mistral…)", "cloud")],
        style=STYLE))  # fmt: skip
    return _ask_local(say) if kind == "local" else _ask_cloud(say)


def ask_director(cfg: LLMConfig) -> str:
    """Local models write slowly: offer the automatic composer (explores each feature and directs it, no AI writing)."""
    if cfg.provider != "ollama":
        return "auto"
    return str(_ask(questionary.select(
        "How should each story be written?",
        choices=[Choice("Automatic composer - explores each feature and directs the story itself (fast)", "composer"),
                 Choice("AI director, with the composer as safety net (best, ~5-8 min per scenario locally)", "auto")],
        style=STYLE)))  # fmt: skip


def _ollama_models() -> list[str] | None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    host = host if host.startswith("http") else f"http://{host}"
    try:
        tags = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return None
    return [m["name"] for m in tags.get("models", []) if "embed" not in m.get("name", "")]


def _ask_local(say: Callable[[str], None]) -> LLMConfig:
    while True:
        models = _ollama_models()
        if models is None:
            say("[yellow]Ollama is not running.[/] Start it in another terminal with [bold]ollama serve[/] "
                "(install: https://ollama.com).")  # fmt: skip
        elif not models:
            say("[yellow]Ollama has no model yet.[/] Install one, e.g. [bold]ollama pull qwen2.5-coder:7b[/].")
        else:
            break
        if not _ask(questionary.confirm("Try again?", default=True, style=STYLE)):
            raise Cancelled
    model = _ask(questionary.select("Which local model?", choices=models, style=STYLE,
                                    instruction="(7B-14B models are fastest on a laptop)"))  # fmt: skip
    spec = f"ollama/{model}"
    remember(spec)
    return parse_model_spec(spec)


def _ask_cloud(say: Callable[[str], None]) -> LLMConfig:
    choices = []
    for name in CLOUD:
        prov = PROVIDERS[name]
        _key, where = provider_key(prov)
        status = f"key found ({where})" if where else prov.note or ""
        choices.append(Choice(f"{name:<11} {status}", name))
    name = str(_ask(questionary.select("Which provider?", choices=choices, style=STYLE)))
    prov = PROVIDERS[name]
    key, where = provider_key(prov)
    if not (key and _ask(questionary.confirm(f"Use the key from {where}?", default=True, style=STYLE))):
        key = str(_ask(questionary.password(f"Paste your {name} API key (hidden):", style=STYLE,
                                            validate=lambda v: len(v.strip()) > 8 or "that does not look like a key")))
        key = key.strip()
        where = None
    model = str(_ask(questionary.text(
        "Model name:", default=prov.default_model or "", style=STYLE,
        instruction="" if prov.default_model else f"(as listed by {name}, e.g. the model id from its docs)",
        validate=lambda v: bool(v.strip()) or "a model name is required"))).strip()  # fmt: skip
    spec = f"{name}/{model}"
    save_key = False
    if where is None:  # typed now: offer to remember it
        save_key = bool(_ask(questionary.confirm(
            "Remember this key on this computer? (saved in your user settings, never in a repository)",
            default=False, style=STYLE)))  # fmt: skip
    path = remember(spec, provider=name, key=key if save_key else None)
    if save_key:
        say(f"[dim]Key saved to {path} - delete it there (or run `pullreel models`) to forget it.[/]")
    return parse_model_spec(spec, api_key=key)


def ask_url() -> str:
    servers = detect_dev_servers()
    if servers:
        choices = [Choice(u, u) for u in servers] + [Choice("Another URL…", "")]
        url = str(_ask(questionary.select("Which running app should I film?", choices=choices, style=STYLE)))
        if url:
            return url
    return str(_ask(questionary.text(
        "App URL (start your dev server first):", default="http://localhost:3000/", style=STYLE,
        validate=lambda v: v.startswith(("http://", "https://")) or "must start with http:// or https://"))).strip()


def ask_repo_and_base(cwd: Path) -> tuple[Path, str]:
    repo = cwd
    if not is_git_repo(repo):
        repo = Path(str(_ask(questionary.path(
            "Path of the project's git repository:", only_directories=True, style=STYLE,
            validate=lambda v: is_git_repo(Path(v)) or "not a git repository")))).expanduser()  # fmt: skip
    detected = detect_base_branch(repo)
    base = str(_ask(questionary.text("Compare the current branch against:", default=detected, style=STYLE))).strip()
    return repo, base or detected


def ask_scenarios(recommended: list[ScoutPick], diff: DiffSummary) -> list[ScoutPick]:
    entries = build_scenario_choices(recommended, diff)
    choices: list[Choice | Separator] = []
    for e in entries:
        if e.pick is None:
            choices.append(Separator("  ───────────"))
            choices.append(Choice(e.label, WRITE_OWN, checked=e.checked))
        else:
            choices.append(Choice(e.label, e.pick, checked=e.checked))
    selected = list(_ask(questionary.checkbox(
        "Which scenarios should I film?", choices=choices, style=STYLE,
        instruction="(space = select · a = all · enter = confirm)",
        validate=lambda s: bool(s) or "select at least one scenario")))  # fmt: skip
    picks: list[ScoutPick] = [p for p in selected if isinstance(p, ScoutPick)]
    if WRITE_OWN in selected:
        while True:
            text = str(_ask(questionary.text(
                "Describe what to show:", style=STYLE,
                instruction="(e.g. \"search a player, open his card, show his shooting stats\")",
                validate=lambda v: len(v.strip()) >= 4 or "a few words please"))).strip()  # fmt: skip
            picks.append(custom_pick(diff, text))
            if not _ask(questionary.confirm("Add another custom scenario?", default=False, style=STYLE)):
                break
    return picks


def confirm(question: str, default: bool = True) -> bool:
    return bool(_ask(questionary.confirm(question, default=default, style=STYLE)))
