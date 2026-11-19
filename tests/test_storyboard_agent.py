from __future__ import annotations

import pytest

from diffreel.models import ActionStep, Camera, Locator, Storyboard
from diffreel.storyboard_agent import (
    InteractiveElement,
    PageSnapshot,
    extract_json,
    fuzzy_match,
    normalize_raw,
    parse_model_spec,
    polish,
)


def test_extract_json_tolerates_fences_think_and_prose() -> None:
    reply = '<think>{"nope": 1}</think>Sure! ```json\n{"a": {"b": "}"}, "c": [1, 2]}\n``` done'
    assert extract_json(reply) == {"a": {"b": "}"}, "c": [1, 2]}
    assert extract_json('noise {"x": "a \\" quote"} trailing {"y": 2}') == {"x": 'a " quote'}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_normalize_raw_fixes_small_model_slips() -> None:
    raw = {
        "feature_summary": "Add a task",
        "start_path": "",
        "steps": [
            {"action": "Pause", "locator": {"strategy": "role", "value": "button", "name": "x"}, "dwell_after_ms": "400"},
            {"action": "type", "locator": {"strategy": "placeholder", "value": "Add a task…", "name": "null"},
             "input_value": "Ship dark mode", "camera": {"mode": "medium", "zoom": 3}},
            {"action": "click", "locator": {"strategy": "role", "value": "Add task", "name": None},
             "camera": {"mode": "focus", "zoom": "1.6"}},
        ],
    }  # fmt: skip
    sb = Storyboard.model_validate(normalize_raw(raw))
    assert sb.start_path == "/"
    assert sb.steps[0].action == "wait" and sb.steps[0].locator is None and sb.steps[0].dwell_after_ms == 400
    assert sb.steps[1].action == "fill" and sb.steps[1].camera.mode == "focus" and sb.steps[1].camera.zoom == 2.0
    assert sb.steps[1].locator is not None and sb.steps[1].locator.name is None
    assert sb.steps[2].locator == Locator(strategy="role", value="button", name="Add task")


def test_polish_enforces_opening_closing_and_budget() -> None:
    sb = Storyboard(
        feature_summary="x",
        steps=[
            ActionStep(action="fill", locator=Locator(strategy="label", value="Name"), input_value="Ada Lovelace",
                       camera=Camera(mode="wide")),
            ActionStep(action="click", locator=Locator(strategy="role", value="button", name="Save"),
                       camera=Camera(mode="focus", zoom=2.0), dwell_after_ms=9000),
        ],
    )  # fmt: skip
    out, notes = polish(sb, max_seconds=8.0)
    assert out.steps[0].action == "wait" and out.steps[0].dwell_after_ms == 400
    assert out.steps[-1].action == "wait" and out.steps[-1].dwell_after_ms == 1000
    assert out.steps[1].camera.mode == "focus"  # fill forced into focus
    assert out.steps[2].camera.zoom == 1.9  # clamped
    assert out.steps[2].dwell_after_ms < 9000
    assert notes


def test_fuzzy_match_recovers_wording_drift() -> None:
    snap = PageSnapshot(
        url="http://x/",
        elements=[
            InteractiveElement(tag="button", role="button", name="Add task"),
            InteractiveElement(tag="input", role="textbox", name="New task", label="", placeholder="Add a task…"),
            InteractiveElement(tag="button", role="button", name="Done"),
        ],
    )
    hit = fuzzy_match(Locator(strategy="role", value="button", name="Add Task Button"), snap)
    assert hit is not None and hit.name == "Add task"
    hit = fuzzy_match(Locator(strategy="placeholder", value="Add a task"), snap)
    assert hit is not None and hit.best_locator() == Locator(strategy="role", value="textbox", name="New task")
    assert fuzzy_match(Locator(strategy="role", value="button", name="Delete workspace"), snap) is None


def test_parse_model_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OLLAMA_HOST", "DIFFREEL_BASE_URL", "DIFFREEL_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = parse_model_spec("ollama/qwen2.5-coder:7b")
    assert (cfg.model, cfg.base_url, cfg.api_key) == ("qwen2.5-coder:7b", "http://localhost:11434/v1", None)
    assert parse_model_spec("llama3.2").base_url == "http://localhost:11434/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = parse_model_spec("openai/gpt-4o-mini")
    assert (cfg.model, cfg.base_url, cfg.api_key) == ("gpt-4o-mini", "https://api.openai.com/v1", "sk-test")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9999")
    assert parse_model_spec("ollama/x").base_url == "http://127.0.0.1:9999/v1"
