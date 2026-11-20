from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import pullreel.storyboard_agent as agent
from pullreel.models import DiffSummary, FeatureCandidate
from pullreel.storyboard_agent import LLMError, ScoutResult, chat_json, match_focus, parse_model_spec, scout_features

KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PULLREEL_API_KEY",
        "PULLREEL_BASE_URL", "OLLAMA_HOST")  # fmt: skip


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    real = httpx.AsyncClient

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(agent.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(wrapped), **kw))  # fmt: skip
    return seen


def test_provider_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        parse_model_spec("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = parse_model_spec("anthropic")
    assert (cfg.provider, cfg.model, cfg.api_key) == ("anthropic", "claude-sonnet-5-5", "sk-ant-test")
    assert parse_model_spec("anthropic/claude-haiku-5-5").model == "claude-haiku-5-5"
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    cfg = parse_model_spec("gemini/gemini-flash")
    assert cfg.base_url.startswith("https://generativelanguage.googleapis.com") and cfg.api_key == "g-test"
    with pytest.raises(LLMError, match="name a model"):
        parse_model_spec("openai")
    assert parse_model_spec("ollama").model == "qwen2.5-coder:7b"  # local: no key needed


def test_anthropic_forces_structured_tool_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    picks = {"features": [{"title": "Adversaires", "route": "/adversaires", "why": "new page", "labels": []}]}
    seen = _mock_http(monkeypatch, lambda r: httpx.Response(200, json={
        "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "name": "feature_picks", "input": picks}]}))
    cfg = parse_model_spec("anthropic")
    out = asyncio.run(chat_json(cfg, [{"role": "system", "content": "SYS"}, {"role": "user", "content": "go"}],
                                schema=ScoutResult.model_json_schema(), schema_name="feature_picks"))  # fmt: skip
    assert json.loads(out) == picks
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path.endswith("/messages") and req.headers["x-api-key"] == "sk-ant-test"
    assert body["system"] == "SYS" and body["messages"] == [{"role": "user", "content": "go"}]
    assert body["tool_choice"] == {"type": "tool", "name": "feature_picks"}


def test_openai_compatible_falls_back_when_json_schema_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        fmt = json.loads(request.content).get("response_format", {}).get("type")
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": "response_format json_schema not supported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    seen = _mock_http(monkeypatch, handler)
    out = asyncio.run(chat_json(parse_model_spec("openai/some-model"), [{"role": "user", "content": "x"}]))
    assert out == '{"ok": true}'
    assert [json.loads(r.content)["response_format"]["type"] for r in seen] == ["json_schema", "json_object"]
    assert seen[0].headers["authorization"] == "Bearer sk-test"


DIFF = DiffSummary(
    base_ref="master",
    head_ref="feature",
    features=[
        FeatureCandidate(title="AdversairesPage", route="/adversaires", added=400, labels=["Leur jeu"], score=9),
        FeatureCandidate(title="JoueursPage", route="/joueurs", added=40, labels=["Ajouter un joueur"], score=5),
        FeatureCandidate(title="Button", route=None, added=3, score=1),
    ],
)


def test_scout_drops_invented_routes_and_tops_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(*_a: Any, **_k: Any) -> str:
        return json.dumps({"features": [
            {"title": "Scouting report", "route": "/rapport-secret", "why": "x", "labels": []},  # invented route
        ]})  # fmt: skip

    monkeypatch.setattr(agent, "chat_json", fake_chat)
    cfg = agent.LLMConfig(model="m", base_url="http://x", provider="ollama")
    picks = asyncio.run(scout_features(DIFF, cfg, 2))
    assert picks[0].title == "Scouting report" and picks[0].route is None
    assert picks[1].route == "/adversaires"  # topped up with the best-scored candidate
    assert len(picks) == 2


def test_scout_survives_a_broken_model(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(*_a: Any, **_k: Any) -> str:
        raise LLMError("down")

    monkeypatch.setattr(agent, "chat_json", broken)
    cfg = agent.LLMConfig(model="m", base_url="http://x", provider="ollama")
    picks = asyncio.run(scout_features(DIFF, cfg, 2))
    assert [p.route for p in picks] == ["/adversaires", "/joueurs"]


def test_match_focus_attaches_the_right_route() -> None:
    assert match_focus(DIFF, "leur jeu").route == "/adversaires"
    assert match_focus(DIFF, "ajouter un joueur").route == "/joueurs"
    assert match_focus(DIFF, "dark mode toggle").route is None


def test_transient_overload_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setattr(agent, "RETRY_DELAYS", (0.0, 0.0, 0.0))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503, json=[{"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}])
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    _mock_http(monkeypatch, handler)
    out = asyncio.run(chat_json(parse_model_spec("gemini/gemini-x"), [{"role": "user", "content": "x"}]))
    assert out == '{"ok": 1}' and calls["n"] == 3


def test_persistent_overload_gives_a_clean_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setattr(agent, "RETRY_DELAYS", (0.0, 0.0, 0.0))
    _mock_http(monkeypatch, lambda r: httpx.Response(
        503, json=[{"error": {"code": 503, "message": "This model is currently experiencing high demand."}}]))
    with pytest.raises(LLMError) as err:
        asyncio.run(chat_json(parse_model_spec("gemini/gemini-x"), [{"role": "user", "content": "x"}]))
    text = str(err.value)
    assert "experiencing high demand" in text and "overloaded" in text and "{" not in text
