from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from pullreel.auth import _write_private, default_auth_path, resolve_auth
from pullreel.models import ActionStep, Camera, DiffSummary, Locator, Storyboard
from pullreel.storyboard_agent import (
    LLMConfig,
    LoginRequiredError,
    PageSnapshot,
    login_hint,
    login_wall,
    plan_storyboard,
    rehearse,
)

LOGIN_PAGE = """<!doctype html><title>Sign in</title>
<form><label>Email <input type="email" name="email"></label>
<label>Password <input type="password" name="password"></label>
<button type="submit">Sign in</button></form>"""


def test_login_wall_rules() -> None:
    gated = PageSnapshot(url="http://localhost:5173/statequipe", has_password_field=True)
    assert login_wall(gated, has_auth=False)  # no session: any password field is a wall
    assert not login_wall(gated, has_auth=True)  # signed in: e.g. a "change password" settings page is fine
    assert login_wall(PageSnapshot(url="http://x/login", has_password_field=True), has_auth=True)  # expired session
    assert not login_wall(PageSnapshot(url="http://x/login", has_password_field=False), has_auth=False)


def test_login_hint_points_to_the_login_command() -> None:
    assert "pullreel login --url http://localhost:5173" in login_hint("http://localhost:5173/statequipe", False)
    assert "expired" in login_hint("http://localhost:5173/login", True)


def test_resolve_auth(tmp_path: Path) -> None:
    assert resolve_auth(tmp_path, None) is None
    saved = default_auth_path(tmp_path)
    _write_private(saved, {"cookies": [], "origins": []})
    assert resolve_auth(tmp_path, None) == saved
    with pytest.raises(FileNotFoundError):
        resolve_auth(tmp_path, tmp_path / "missing.json")


def test_saved_session_folder_ignores_itself(tmp_path: Path) -> None:
    path = default_auth_path(tmp_path)
    _write_private(path, {"cookies": [{"name": "sb"}], "origins": []})
    assert json.loads(path.read_text())["cookies"][0]["name"] == "sb"
    assert (path.parent / ".gitignore").read_text().strip().endswith("*")


def test_login_screen_stops_before_any_llm_call(tmp_path: Path) -> None:
    page = tmp_path / "login.html"
    page.write_text(LOGIN_PAGE, encoding="utf-8")
    dead_llm = LLMConfig(model="none", base_url="http://127.0.0.1:9/v1", ollama_native=True)  # would fail if called
    diff = DiffSummary(base_ref="main", head_ref="feature")
    with pytest.raises(LoginRequiredError, match="pullreel login"):
        asyncio.run(plan_storyboard(page.as_uri(), diff, dead_llm))


def test_rehearsal_refuses_to_type_a_password(tmp_path: Path) -> None:
    page = tmp_path / "login.html"
    page.write_text(LOGIN_PAGE, encoding="utf-8")
    sb = Storyboard(
        feature_summary="bad",
        steps=[ActionStep(action="fill", locator=Locator(strategy="label", value="Password"),
                          input_value="password123", camera=Camera(mode="focus", zoom=1.6))],
    )  # fmt: skip

    async def run() -> str | None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                return (await rehearse(browser, page.as_uri(), sb)).error
            finally:
                await browser.close()

    error = asyncio.run(run())
    assert error is not None and "password field" in error


SPA_PAGE = """<!doctype html><title>Stats</title><body>
<script>
  // Renders late, like an SPA waiting for its auth/data calls.
  setTimeout(() => { document.body.innerHTML = `
    <nav><div class="nav" style="cursor:pointer" onclick="go('players')"><img alt="Joueurs" src="data:,"
         width="24" height="24"><span style="opacity:0">Joueurs tooltip</span></div></nav>
    <div class="tabs"><h5 style="cursor:pointer" onclick="tab(1)">Analyse</h5>
                      <h5 style="cursor:pointer" onclick="tab(2)">Feuille de stats</h5></div>
    <p>Analyse défensive détaillée</p>`; }, 1200);
</script></body>"""


def test_snapshot_waits_for_late_spa_and_finds_clickable_divs(tmp_path: Path) -> None:
    from pullreel.human_runner import find_locator, wait_for_content
    from pullreel.storyboard_agent import capture_snapshot

    page_file = tmp_path / "spa.html"
    page_file.write_text(SPA_PAGE, encoding="utf-8")

    async def run() -> tuple[list[str], str]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.goto(page_file.as_uri())
                assert await wait_for_content(page, timeout_s=8)
                snap = await capture_snapshot(page)
                exact = await find_locator(page, Locator(strategy="text", value="Analyse"))
                return [e.best_locator().describe() for e in snap.elements], await exact.evaluate("e => e.tagName")
            finally:
                await browser.close()

    locators, analyse_tag = asyncio.run(run())
    assert "role='img' name='Joueurs'" in locators  # icon named by its alt, not by the hidden tooltip
    assert "text='Analyse'" in locators and "text='Feuille de stats'" in locators
    assert not any("tooltip" in loc for loc in locators)
    assert analyse_tag == "H5"  # exact text wins over "Analyse défensive détaillée"
