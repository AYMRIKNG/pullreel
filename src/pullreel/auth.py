"""`pullreel login`: the user signs in by hand in a visible browser; we keep only the resulting session.

PullReel never sees, types or stores a password. The saved file is a Playwright storage state (cookies + localStorage
+ IndexedDB) that later browser contexts load, so `record` and `plan` start already signed in. It holds live session
tokens, so it is written to `.pullreel/` with a `.gitignore` that ignores the whole folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from pullreel.human_runner import VIEWPORT, launch_chromium

AUTH_DIR = ".pullreel"
AUTH_FILE = "auth.json"
_PASSWORD_VISIBLE_JS = "() => [...document.querySelectorAll('input[type=password]')].some(e => e.offsetParent !== null)"


@dataclass(slots=True)
class LoginResult:
    path: Path
    cookies: int
    origins: int
    still_on_login: bool  # the window was left on a page that still shows a password field


def default_auth_path(repo: Path) -> Path:
    return repo / AUTH_DIR / AUTH_FILE


def resolve_auth(repo: Path, explicit: Path | None) -> Path | None:
    """`--auth FILE` if given (must exist), else `<repo>/.pullreel/auth.json` when it exists, else None."""
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"saved session {explicit} not found: create it with `pullreel login --url ...`")
        return explicit
    candidate = default_auth_path(repo)
    return candidate if candidate.is_file() else None


def _write_private(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == AUTH_DIR:
        ignore = path.parent / ".gitignore"
        if not ignore.exists():
            ignore.write_text("# session tokens saved by `pullreel login` - never commit them\n*\n", encoding="utf-8")
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


async def login(url: str, path: Path, *, timeout_s: float = 900.0) -> LoginResult:
    """Open ``url`` in a visible browser and save the session once the user presses Enter or closes the window."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"login needs an http(s) URL, got {url!r}")
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()

    def wait_enter() -> None:  # daemon thread: never keeps the process alive
        with contextlib.suppress(EOFError, OSError):
            input()
        loop.call_soon_threadsafe(finished.set)

    async with async_playwright() as pw:
        browser = await launch_chromium(pw, headed=True)
        try:
            context = await browser.new_context(viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]})
            page = await context.new_page()
            page.on("close", lambda _p: loop.call_soon_threadsafe(finished.set))
            browser.on("disconnected", lambda _b: loop.call_soon_threadsafe(finished.set))
            await page.goto(url, wait_until="load")
            threading.Thread(target=wait_enter, daemon=True).start()

            latest: dict[str, Any] | None = None
            elapsed = 0.0
            while not finished.is_set() and elapsed < timeout_s:
                # Snapshot the session continuously: if the user simply closes the window we still have it.
                with contextlib.suppress(PlaywrightError):
                    latest = dict(await context.storage_state(indexed_db=True))
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(finished.wait(), timeout=1.5)
                elapsed += 1.5

            still_on_login = False
            if not page.is_closed():
                with contextlib.suppress(PlaywrightError):
                    latest = dict(await context.storage_state(indexed_db=True))
                    still_on_login = bool(await page.evaluate(_PASSWORD_VISIBLE_JS))
        finally:
            with contextlib.suppress(PlaywrightError):
                await browser.close()

    if latest is None:
        raise RuntimeError("the browser closed before any session could be saved - run `pullreel login` again")
    _write_private(path, latest)
    return LoginResult(path, len(latest.get("cookies", [])), len(latest.get("origins", [])), still_on_login)


def login_sync(url: str, path: Path) -> LoginResult:
    return asyncio.run(login(url, path))
