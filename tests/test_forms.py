"""Creation journeys ("add a new player"): unwired labels, required fields, chips, the new item in the list."""

from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

from pullreel.models import ActionStep, Camera, Locator, Storyboard
from pullreel.storyboard_agent import RehearsalResult, capture_snapshot, rehearse

# Same shape as a real roster app: <label> siblings NOT wired to their inputs, required fields, position chips.
ROSTER = """<!doctype html><title>Roster</title><body>
<h2>Players</h2><ul id="list"><li>Anthony</li></ul>
<button onclick="document.getElementById('modal').hidden=false">Add a player</button>
<div id="modal" hidden><h3>New player</h3>
<form onsubmit="event.preventDefault(); const li=document.createElement('li');
  li.textContent=document.getElementById('f').value+' '+document.getElementById('l').value;
  document.getElementById('list').append(li); document.getElementById('modal').hidden=true;">
  <div><label>First name *</label><input id="f" required></div>
  <div><label>Last name *</label><input id="l" required></div>
  <div><label>Jersey *</label><input id="n" type="number" required></div>
  <div><button type="button">Guard</button><button type="button">Center</button></div>
  <button type="submit">Save player</button>
</form></div></body>"""


def _sb(*steps: ActionStep) -> Storyboard:
    return Storyboard(feature_summary="Add a player", start_path="/", steps=list(steps))


def _click(v: str) -> ActionStep:
    return ActionStep(action="click", locator=Locator(strategy="text", value=v), camera=Camera(mode="focus", zoom=1.5))


def _fill(loc: Locator, value: str) -> ActionStep:
    return ActionStep(action="fill", locator=loc, input_value=value, camera=Camera(mode="focus", zoom=1.8))


def _run(tmp_path: Path, sb: Storyboard | None) -> tuple[list[tuple[str, str]], RehearsalResult | None]:
    page_file = tmp_path / "roster.html"
    page_file.write_text(ROSTER, encoding="utf-8")

    async def go() -> tuple[list[tuple[str, str]], RehearsalResult | None]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.goto(page_file.as_uri())
                await page.click("text=Add a player")
                snap = await capture_snapshot(page)
                fields = [(e.role, e.label) for e in snap.elements if e.tag == "input"]
                res = await rehearse(browser, page_file.as_uri(), sb) if sb is not None else None
                return fields, res
            finally:
                await browser.close()

    return asyncio.run(go())


def test_unwired_labels_name_the_fields(tmp_path: Path) -> None:
    fields, _ = _run(tmp_path, None)
    assert fields == [("textbox", "First name"), ("textbox", "Last name"), ("spinbutton", "Jersey")]


def test_two_fills_into_the_same_field_fail_the_rehearsal(tmp_path: Path) -> None:
    anon = Locator(strategy="role", value="textbox")  # what a model writes when the fields have no accessible name
    _, res = _run(tmp_path, _sb(_click("Add a player"), _fill(anon, "Lucas"), _fill(anon, "Martin")))
    assert res is not None and res.error is not None and "SAME field" in res.error


def test_submitting_with_a_required_field_empty_fails_the_rehearsal(tmp_path: Path) -> None:
    _, res = _run(tmp_path, _sb(_click("Add a player"), _fill(Locator(strategy="label", value="First name"), "Lucas"),
                                _click("Center"), _click("Save player")))  # fmt: skip
    assert res is not None and res.error is not None
    assert "still invalid" in res.error and "Last name" in res.error
    assert res.failed_step == 3  # the submit, not the chip click before it: only a submit attempt counts


def test_a_complete_creation_journey_rehearses_and_shows_the_new_item(tmp_path: Path) -> None:
    by_label = lambda v: Locator(strategy="label", value=v)  # noqa: E731
    # role + the name the snapshot showed works too, although the browser itself gives these inputs no name
    jersey = Locator(strategy="role", value="spinbutton", name="Jersey")
    _, res = _run(tmp_path, _sb(
        _click("Add a player"),
        _fill(by_label("First name"), "Lucas"), _fill(by_label("Last name"), "Moreau"), _fill(jersey, "7"),
        _click("Center"), _click("Save player"),
    ))  # fmt: skip
    assert res is not None and res.error is None, res.error
    assert "Lucas Moreau" in res.observations[-1]  # the money shot exists: the new player is in the list


def test_the_created_item_becomes_the_money_shot(tmp_path: Path) -> None:
    from pullreel.storyboard_agent import with_money_shot

    by_label = lambda v: Locator(strategy="label", value=v)  # noqa: E731
    sb = _sb(_click("Add a player"), _fill(by_label("First name *"), "Lucas"), _fill(by_label("Last name"), "Moreau"),
             _fill(by_label("Jersey"), "7"), _click("Save player"),
             ActionStep(action="wait", camera=Camera(mode="wide")))  # fmt: skip
    _, res = _run(tmp_path, sb)
    assert res is not None and res.error is None, res.error
    assert res.created == "Lucas Moreau"  # found outside the (now closed) form, wherever it landed
    final, notes = with_money_shot(res.storyboard, res.created)
    assert [s.action for s in final.steps][-2:] == ["show", "wait"]
    assert final.steps[-2].locator == Locator(strategy="text", value="Lucas Moreau") and notes
