"""Tests for Work Package 2: Concrete journey detection, realistic form filling, multi-page flows,
mutation safety (--dry-rehearsal, blocklist), sales ranking, and respecting user intent.

All browser tests run against local HTML files in tmp_path (zero network access).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import async_playwright

import pullreel.storyboard_agent as agent
from pullreel.composer import story_quality
from pullreel.journeys import (
    FormField,
    classify_workflows,
    detect_mutating_storyboard,
    explore_journeys,
    generate_field_value,
    is_destructive,
    is_sensitive_field,
    score_candidate_sales,
)
from pullreel.models import ActionStep, Camera, DiffSummary, FeatureCandidate, Locator, Storyboard
from pullreel.storyboard_agent import LLMConfig, LLMError, PlanError, ScoutPick, plan_storyboard, rehearse
from pullreel.wizard import build_scenario_choices, custom_pick

ROSTER_FORM_PAGE = """<!doctype html>
<title>Team Roster</title>
<style>
  body { font-family: sans-serif; padding: 20px; background: #111; color: #eee; }
  .modal { display: none; border: 1px solid #555; padding: 16px; margin-top: 12px; background: #1c1c24; }
  .modal.is-open { display: block; }
  table { margin-top: 16px; width: 620px; border-collapse: collapse; }
  th, td { border: 1px solid #333; padding: 6px 10px; text-align: left; }
</style>
<body>
  <main>
    <h1>Team Roster</h1>
    <div>
      <input type="search" aria-label="Search players" placeholder="Search players..." oninput="filterRows(this.value)" />
      <button type="button" onclick="openModal()">+ Add player</button>
      <button type="button" onclick="exportCsv()">Export CSV</button>
      <button type="button" onclick="deleteAll()">Delete all</button>
      <button type="button" onclick="logout()">Sign out</button>
    </div>

    <div id="modal" class="modal" role="dialog" aria-modal="true">
      <h2>New player</h2>
      <form id="player-form" onsubmit="submitPlayer(event)">
        <label for="p-name">Player name</label>
        <input id="p-name" name="player_name" type="text" placeholder="Full name" required /><br/>
        <label for="p-num">Jersey number</label>
        <input id="p-num" name="jersey" type="number" min="0" max="99" /><br/>
        <label for="p-email">Email</label>
        <input id="p-email" name="email" type="email" placeholder="player@example.com" /><br/>
        <label for="p-pos">Position</label>
        <select id="p-pos" name="position">
          <option value="">Select position...</option>
          <option value="PG">Point Guard</option>
          <option value="SF">Small Forward</option>
        </select><br/>
        <label for="p-secret">Account Password</label>
        <input id="p-secret" name="password" type="password" /><br/>
        <button type="submit">Save player</button>
      </form>
    </div>

    <p id="submit-count" data-count="0">Saved: 0</p>
    <table id="roster-table">
      <thead><tr><th>Player</th><th>#</th><th>Position</th><th>Email</th></tr></thead>
      <tbody id="roster-body">
        <tr><td>Marcus Hill</td><td>7</td><td>Point Guard</td><td>marcus@example.com</td></tr>
        <tr><td>Lukas Meyer</td><td>23</td><td>Small Forward</td><td>lukas@example.com</td></tr>
      </tbody>
    </table>
  </main>
  <script>
    let submits = 0;
    function openModal() { document.getElementById('modal').classList.add('is-open'); }
    function submitPlayer(ev) {
      ev.preventDefault();
      submits += 1;
      document.getElementById('submit-count').dataset.count = String(submits);
      document.getElementById('submit-count').textContent = 'Saved: ' + submits;
      const name = document.getElementById('p-name').value;
      const num = document.getElementById('p-num').value || '11';
      const email = document.getElementById('p-email').value;
      const pos = document.getElementById('p-pos').selectedOptions[0].text;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${name}</td><td>${num}</td><td>${pos}</td><td>${email}</td>`;
      document.getElementById('roster-body').appendChild(tr);
      document.getElementById('modal').classList.remove('is-open');
    }
    function filterRows(q) {
      const low = q.toLowerCase();
      for (const tr of document.querySelectorAll('#roster-body tr')) {
        tr.style.display = tr.innerText.toLowerCase().includes(low) ? '' : 'none';
      }
    }
    function exportCsv() {}
    function deleteAll() {}
    function logout() {}
  </script>
</body>"""


def test_realistic_form_data_and_safety_filters() -> None:
    assert generate_field_value(FormField(tag="input", input_type="email", label="Email")) == "alex.moreau@example.com"
    assert generate_field_value(FormField(tag="input", input_type="date", label="Due date")) == "2026-11-24"
    assert generate_field_value(FormField(tag="input", input_type="number", label="Jersey number", min_val=0, max_val=99)) == "11"
    assert generate_field_value(
        FormField(tag="select", input_type="select", label="Position", options=["Select position...", "Point Guard", "Center"])
    ) == "Point Guard"
    # Never fill passwords, credit cards, or secrets
    assert generate_field_value(FormField(tag="input", input_type="password", label="Password")) is None
    assert generate_field_value(FormField(tag="input", input_type="text", label="Credit card number")) is None
    assert generate_field_value(FormField(tag="input", input_type="text", name="api_key", label="API Key")) is None
    assert is_sensitive_field(label="CVV code")
    assert is_destructive("Delete account") and is_destructive("Sign out") and is_destructive("Revoke access")
    assert is_destructive("Nuke DB", extra_blocklist=["nuke"])
    assert not is_destructive("+ Add player")


def test_classify_workflows_assigns_business_titles_and_ignores_destructive() -> None:
    raw = {
        "heading": "Team Roster",
        "controls": [
            {"tag": "button", "role": "button", "text": "+ Add player", "label": "", "placeholder": "", "testid": ""},
            {"tag": "input", "role": "textbox", "text": "Search players", "label": "Search players", "placeholder": "Search...", "testid": "", "input_type": "search"},
            {"tag": "button", "role": "button", "text": "Export CSV", "label": "", "placeholder": "", "testid": ""},
            {"tag": "button", "role": "button", "text": "Delete all", "label": "", "placeholder": "", "testid": ""},
            {"tag": "button", "role": "button", "text": "Sign out", "label": "", "placeholder": "", "testid": ""},
        ],
    }
    wfs = classify_workflows(raw)
    kinds = [w.kind for w in wfs]
    assert kinds == ["create", "search", "export"]
    assert "Add a new player" in wfs[0].title and wfs[0].mutates is True
    assert wfs[1].mutates is False
    assert all("delete" not in w.trigger_text.lower() and "sign out" not in w.trigger_text.lower() for w in wfs)


def test_explore_journeys_fills_modal_form_and_shows_new_item(tmp_path: Path) -> None:
    page_file = tmp_path / "roster.html"
    page_file.write_text(ROSTER_FORM_PAGE, encoding="utf-8")

    async def run() -> list:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                return await explore_journeys(browser, page_file.as_uri(), dry_rehearsal=False)
            finally:
                await browser.close()

    plans = asyncio.run(run())
    assert plans
    create_plan = next(p for p in plans if p.workflow.kind == "create")
    sb = create_plan.storyboard
    assert sb.mutates is True
    assert "Add a new player" in sb.feature_summary
    # Password field must NEVER appear in steps; email must use @example.com
    filled = [(s.locator.describe(), s.input_value) for s in sb.steps if s.action == "fill" and s.locator]
    assert filled
    assert not any("password" in d.lower() for d, _ in filled)
    assert any(v and v.endswith("@example.com") for _, v in filled)
    # Must end with a money-shot show step on the newly created player + wide hold
    shows = [s for s in sb.steps if s.action == "show"]
    assert shows and shows[-1].locator is not None and shows[-1].locator.value == "Alex Moreau"
    assert detect_mutating_storyboard(sb) is True


def test_dry_rehearsal_verifies_submit_without_mutating_and_blocks_destructive(tmp_path: Path) -> None:
    page_file = tmp_path / "roster.html"
    page_file.write_text(ROSTER_FORM_PAGE, encoding="utf-8")

    sb_create = Storyboard(
        feature_summary="Add a new player",
        mutates=True,
        steps=[
            ActionStep(action="click", locator=Locator(strategy="role", value="button", name="+ Add player")),
            ActionStep(action="fill", locator=Locator(strategy="label", value="Player name"), input_value="Alex Moreau"),
            ActionStep(action="click", locator=Locator(strategy="role", value="button", name="Save player")),
            ActionStep(action="show", locator=Locator(strategy="text", value="Alex Moreau")),
        ],
    )
    sb_bad = Storyboard(
        feature_summary="Danger",
        steps=[ActionStep(action="click", locator=Locator(strategy="role", value="button", name="Delete all"))],
    )

    async def run() -> tuple:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                dry = await rehearse(browser, page_file.as_uri(), sb_create, dry_rehearsal=True)
                blocked = await rehearse(browser, page_file.as_uri(), sb_bad)
                return dry, blocked
            finally:
                await browser.close()

    dry, blocked = asyncio.run(run())
    assert dry.error is None, dry.error
    assert any("dry-rehearsal verified submit (not clicked)" in o for o in dry.observations)
    assert blocked.error is not None and "destructive" in blocked.error.lower()


def test_multipage_journey_flag_allows_crossing_pages(tmp_path: Path) -> None:
    detail = tmp_path / "detail.html"
    listing = tmp_path / "list.html"
    detail.write_text(
        f'<!doctype html><body><h1>Player Detail</h1><p>Stats for Marcus</p><a href="{listing.name}">Back to roster</a></body>',
        encoding="utf-8",
    )
    listing.write_text(
        f'<!doctype html><body><h1>Roster</h1><a href="{detail.name}">View Marcus</a></body>',
        encoding="utf-8",
    )
    sb = Storyboard(
        feature_summary="Inspect player detail and return",
        journey=True,
        steps=[
            ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=600),
            ActionStep(action="click", locator=Locator(strategy="role", value="link", name="View Marcus")),
            ActionStep(action="show", locator=Locator(strategy="text", value="Stats for Marcus")),
            ActionStep(action="click", locator=Locator(strategy="role", value="link", name="Back to roster")),
            ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1000),
        ],
    )

    async def run() -> Any:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                return await rehearse(browser, listing.as_uri(), sb)
            finally:
                await browser.close()

    res = asyncio.run(run())
    assert res.error is None, res.error
    assert res.left_page_at is None
    assert any("navigated:" in o for o in res.observations)
    assert not any("LEFT THE FEATURE PAGE" in o for o in res.observations)
    assert story_quality(res.storyboard, res.observations) >= 6.0


def test_scenario_sales_ranking_and_wizard_display() -> None:
    high = score_candidate_sales(
        visual_payoffs=3,
        effective_steps=4,
        workflow_kind="create",
        candidate=FeatureCandidate(title="Roster", route="/roster", added=420, new_files=2, labels=["+ Add player", "Save"]),
    )
    low = score_candidate_sales(
        visual_payoffs=0,
        effective_steps=1,
        candidate=FeatureCandidate(title="Footer", route="/about", added=10, new_files=0),
    )
    assert high > low + 35.0
    diff = DiffSummary(
        base_ref="main",
        head_ref="feat",
        features=[FeatureCandidate(title="Roster", route="/roster", added=200, new_files=1)],
    )
    rec = [ScoutPick(title="Add a new player", route="/roster", why="new modal", sales_score=high, mutates=True)]
    choices = build_scenario_choices(rec, diff)
    assert f"score {int(round(high))}" in choices[0].label
    assert "mutates" in choices[0].label


def test_user_requested_custom_scenario_is_never_replaced_by_composer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page_file = tmp_path / "roster.html"
    page_file.write_text(ROSTER_FORM_PAGE, encoding="utf-8")
    diff = DiffSummary(base_ref="main", head_ref="feat")
    pick = custom_pick(diff, "filter players by Marcus and show his row")
    assert pick.user_requested is True

    async def broken_chat(*_a: Any, **_k: Any) -> str:
        raise LLMError("model unreachable")

    monkeypatch.setattr(agent, "chat_json", broken_chat)
    cfg = LLMConfig(model="m", base_url="http://127.0.0.1:9/v1", provider="ollama", ollama_native=True)

    # Even in director="auto", a custom user request must fail clearly rather than letting the composer film something else.
    with pytest.raises(PlanError, match="custom scenario|not substitute"):
        asyncio.run(plan_storyboard(page_file.as_uri(), diff, cfg, feature=pick, director="auto"))

    # And director="composer" on a custom request must refuse clearly.
    with pytest.raises(PlanError, match="require an AI director"):
        asyncio.run(plan_storyboard(page_file.as_uri(), diff, cfg, feature=pick, director="composer"))
