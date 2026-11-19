"""Git diff + live accessibility snapshot -> grounded, cinematic Storyboard JSON (via a local or remote LLM).

Pipeline:
1. Open the page in headless Chromium (jumping to the route the diff touches when the URL is the site root) and
   capture an ARIA snapshot plus a flat list of visible interactive elements with their semantic locators.
2. Ask an OpenAI-compatible chat endpoint (Ollama by default) for a storyboard in strict JSON.
3. Normalise the JSON (synonyms, clamped zooms, role/name mix-ups), validate it with pydantic, and polish the pacing.
4. Rehearse the storyboard at full speed in a fresh browser context. Every locator must resolve *at the moment it is
   used* (elements created by earlier steps are fine). Misses are repaired deterministically by fuzzy-matching the live
   snapshot; anything still broken goes back to the LLM with the exact error and the page state at that step.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from playwright.async_api import Browser, Page
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, ValidationError

from pullreel.human_runner import SCALE, VIEWPORT, launch_chromium, resolve_locator
from pullreel.models import ActionStep, Camera, DiffSummary, Locator, Storyboard

ProgressFn = Callable[[str], None]

SYSTEM_PROMPT = """You are the Autonomous Demo Director inside PullReel.
Your job is to read a Git diff and the live Accessibility Snapshot of a web page, then design a crisp, high-impact 5-to-12-second UI demonstration storyboard in strict JSON format.

### INPUTS PROVIDED TO YOU
1. `GIT_DIFF_SUMMARY`: Changed files, routes, component names, labels, and state transitions introduced in the branch.
2. `PAGE_URL` & `ARIA_SNAPSHOT`: Interactive elements currently rendered on the page with their exact roles, accessible names, placeholders, and stable attributes.

### CINEMATIC DIRECTION RULES
1. **Show the Delta, Skip the Boilerplate:** Focus exclusively on demonstrating the new or modified UI behavior from the Git diff. Never waste time clicking unrelated navigation links unless strictly required to reach the feature.
2. **Zero Hallucinated Selectors:** NEVER guess brittle CSS classes or Tailwind hashes. Choose targets strictly present in `ARIA_SNAPSHOT` using semantic locators (`role` + `name`, `label`, `placeholder`, or `testid`). An element that only appears after one of your own steps (for example a row you just created) may be targeted by the name it will have, derived from the diff.
3. **Pacing for GitHub READMEs (5–12s total):**
   - Start with a brief `wait` (400ms) in `wide` shot so the viewer anchors the initial state.
   - Use `focus` camera mode (`zoom: 1.75`) on key micro-interactions (clicking a new button, typing into a new input, toggling a filter).
   - Always end with a 1000ms `wide` or `medium` hold (`zoom: 1.0` or `1.25`) so the final visual outcome is clearly readable before the GIF loops.
4. **Realistic Sample Data:** When typing into inputs, invent short, domain-authentic text (2 to 4 words max) rather than "test123" or "foo".

### STRICT JSON OUTPUT SCHEMA
Respond ONLY with a valid JSON object matching this exact structure (no markdown fences, no commentary):
{
  "feature_summary": "One-sentence description of what is being demonstrated",
  "start_path": "/optional-subroute-from-diff-or-root",
  "steps": [
    {
      "action": "wait | click | fill | press | hover | scroll",
      "locator": {
        "strategy": "role | label | placeholder | text | testid | css",
        "value": "button",
        "name": "Exact accessible name if strategy is role, else null"
      },
      "input_value": "Text to type for fill, key name for press (e.g. Enter), or null",
      "camera": {
        "mode": "wide | focus | pan",
        "zoom": 1.75
      },
      "dwell_after_ms": 450
    }
  ]
}
For `wait` and `press` steps, `locator` is null."""

ARIA_ROLES = {
    "button", "link", "textbox", "searchbox", "checkbox", "radio", "switch", "combobox", "listbox", "option",
    "tab", "tabpanel", "menuitem", "menuitemcheckbox", "menuitemradio", "slider", "spinbutton", "heading",
    "row", "cell", "gridcell", "dialog", "img", "listitem", "treeitem", "progressbar", "navigation",
}  # fmt: skip
ACTION_SYNONYMS = {"type": "fill", "input": "fill", "enter_text": "fill", "tap": "click", "mouseover": "hover",
                   "sleep": "wait", "pause": "wait", "key": "press", "keypress": "press"}  # fmt: skip
SNAPSHOT_URL_MAX_ARIA = 4500

# Collect visible interactive elements with the attributes a semantic locator can use.
_COLLECT_JS = r"""
() => {
  const sel = 'a[href],button,input,textarea,select,summary,[contenteditable="true"],[data-testid],[data-test],[data-cy],' +
    '[role=button],[role=link],[role=tab],[role=checkbox],[role=switch],[role=menuitem],[role=option],[role=radio],' +
    '[role=combobox],[role=textbox],[role=slider],[tabindex]:not([tabindex="-1"])';
  const implicit = (el) => {
    const t = el.tagName.toLowerCase();
    if (el.getAttribute('role')) return el.getAttribute('role');
    if (t === 'a') return 'link';
    if (t === 'button' || t === 'summary') return 'button';
    if (t === 'select') return el.multiple ? 'listbox' : 'combobox';
    if (t === 'textarea') return 'textbox';
    if (t === 'input') {
      const ty = (el.type || 'text').toLowerCase();
      return ({checkbox:'checkbox', radio:'radio', range:'slider', number:'spinbutton', search:'searchbox',
               button:'button', submit:'button', reset:'button', image:'button'})[ty] || 'textbox';
    }
    return '';
  };
  const text = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const labelOf = (el) => {
    if (el.labels && el.labels.length) return text(el.labels[0].innerText);
    const lb = el.getAttribute('aria-labelledby');
    if (lb) return text(lb.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' '));
    return '';
  };
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll(sel)) {
    if (seen.has(el)) continue; seen.add(el);
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    if (r.width < 2 || r.height < 2 || st.visibility === 'hidden' || st.display === 'none') continue;
    if (r.bottom < 0 || r.right < 0 || r.top > innerHeight * 2.5 || r.left > innerWidth) continue;
    const role = implicit(el);
    const label = labelOf(el);
    const aria = text(el.getAttribute('aria-label'));
    const placeholder = text(el.getAttribute('placeholder'));
    const inner = ['input','textarea','select'].includes(el.tagName.toLowerCase()) ? '' : text(el.innerText).slice(0, 80);
    const name = aria || label || inner || placeholder || text(el.getAttribute('title')) || text(el.getAttribute('alt'));
    out.push({
      tag: el.tagName.toLowerCase(), role, name: name.slice(0, 80), label, placeholder,
      testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-cy') || '',
      box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
    });
    if (out.length >= 90) break;
  }
  return out;
}
"""


class PlanError(RuntimeError):
    """The LLM could not produce a valid, grounded storyboard."""


class LLMError(RuntimeError):
    """The LLM endpoint failed (unreachable, unknown model, bad response)."""


# --------------------------------------------------------------------------------------------- LLM configuration
@dataclass(slots=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str | None = None
    temperature: float = 0.2
    timeout_s: float = 600.0
    ollama_native: bool = False  # use /api/chat: JSON-schema output, num_ctx, thinking off
    num_ctx: int = 16384

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.base_url}"


_PROVIDERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ollama": ("http://localhost:11434/v1", ()),
    "lmstudio": ("http://localhost:1234/v1", ()),
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "openrouter": ("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
}


def parse_model_spec(spec: str, *, base_url: str | None = None, api_key: str | None = None) -> LLMConfig:
    """``ollama/qwen2.5-coder:7b``, ``openai/gpt-4o-mini``, ``lmstudio/...`` or a bare model name (Ollama)."""
    provider, _, model = spec.partition("/")
    if not model or provider not in _PROVIDERS:
        provider, model = "ollama", spec
    default_url, key_envs = _PROVIDERS[provider]
    if provider == "ollama" and os.environ.get("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"]
        host = host if host.startswith("http") else f"http://{host}"
        default_url = host.rstrip("/") + "/v1"
    url = base_url or os.environ.get("PULLREEL_BASE_URL") or default_url
    key = api_key or os.environ.get("PULLREEL_API_KEY")
    for env in key_envs:
        key = key or os.environ.get(env)
    return LLMConfig(model=model, base_url=url.rstrip("/"), api_key=key, ollama_native=provider == "ollama")


async def _ollama_chat(cfg: LLMConfig, messages: list[dict[str, str]]) -> str:
    """Native Ollama chat: grammar-constrained to the Storyboard schema, big context, no hidden thinking."""
    root = cfg.base_url.removesuffix("/v1")
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "format": Storyboard.model_json_schema(),
        "think": False,
        "stream": False,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
    }
    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        try:
            resp = await client.post(f"{root}/api/chat", json=body)
            if resp.status_code == 400 and "think" in resp.text:  # model without a thinking switch
                body.pop("think")
                resp = await client.post(f"{root}/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise LLMError(f"cannot reach Ollama at {root}. Start it with `ollama serve`.") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"{cfg.label} timed out after {cfg.timeout_s:.0f}s (try a smaller model)") from exc
        if resp.status_code == 404:
            names: list[str] = []
            with contextlib.suppress(Exception):
                names = [m["name"] for m in (await client.get(f"{root}/api/tags")).json().get("models", [])]
            have = f" Installed: {', '.join(names)}." if names else ""
            raise LLMError(f"model {cfg.model!r} is not installed: run `ollama pull {cfg.model}`.{have}")
        if resp.status_code >= 400:
            raise LLMError(f"{cfg.label} returned HTTP {resp.status_code}: {resp.text[:400]}")
    content = resp.json().get("message", {}).get("content", "")
    if not content:
        raise LLMError(f"{cfg.label} returned an empty message")
    return str(content)


async def chat_json(cfg: LLMConfig, messages: list[dict[str, str]]) -> str:
    """One chat completion asking for a JSON object (Ollama native API, or any OpenAI-compatible endpoint)."""
    if cfg.ollama_native:
        return await _ollama_chat(cfg, messages)
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=cfg.timeout_s, headers=headers) as client:
        try:
            resp = await client.post(f"{cfg.base_url}/chat/completions", json=body)
            if resp.status_code == 400 and "response_format" in resp.text:
                body.pop("response_format")
                resp = await client.post(f"{cfg.base_url}/chat/completions", json=body)
        except httpx.ConnectError as exc:
            hint = " Is Ollama running? Start it with `ollama serve`." if "11434" in cfg.base_url else ""
            raise LLMError(f"cannot reach {cfg.base_url}.{hint}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"{cfg.label} timed out after {cfg.timeout_s:.0f}s") from exc
        if resp.status_code == 404 and "model" in resp.text.lower():
            raise LLMError(f"model {cfg.model!r} not found at {cfg.base_url}{await _available_models(client, cfg)}")
        if resp.status_code >= 400:
            raise LLMError(f"{cfg.label} returned HTTP {resp.status_code}: {resp.text[:400]}")
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"unexpected response from {cfg.label}: {resp.text[:400]}") from exc
    if not content:
        raise LLMError(f"{cfg.label} returned an empty message")
    return str(content)


async def _available_models(client: httpx.AsyncClient, cfg: LLMConfig) -> str:
    with contextlib.suppress(Exception):
        r = await client.get(f"{cfg.base_url}/models")
        names = [m.get("id", "") for m in r.json().get("data", [])]
        if names:
            pull = f" (or `ollama pull {cfg.model}`)" if "11434" in cfg.base_url else ""
            return f". Available: {', '.join(names[:12])}{pull}"
    return ""


# --------------------------------------------------------------------------------------------- page snapshot
class InteractiveElement(BaseModel):
    tag: str
    role: str = ""
    name: str = ""
    label: str = ""
    placeholder: str = ""
    testid: str = ""
    box: tuple[int, int, int, int] = (0, 0, 0, 0)

    def best_locator(self) -> Locator:
        if self.testid:
            return Locator(strategy="testid", value=self.testid)
        if self.role in ARIA_ROLES and self.name:
            return Locator(strategy="role", value=self.role, name=self.name)
        if self.label:
            return Locator(strategy="label", value=self.label)
        if self.placeholder:
            return Locator(strategy="placeholder", value=self.placeholder)
        return Locator(strategy="text", value=self.name or self.tag)

    def keys(self) -> list[str]:
        return [k for k in (self.name, self.label, self.placeholder, self.testid) if k]

    def line(self) -> str:
        parts = [f"- {self.role or self.tag}"]
        if self.name:
            parts.append(f'"{self.name}"')
        for k in ("label", "placeholder", "testid"):
            v = getattr(self, k)
            if v and v != self.name:
                parts.append(f"{k}={v!r}")
        return " ".join(parts)


class PageSnapshot(BaseModel):
    url: str
    title: str = ""
    aria: str = ""
    elements: list[InteractiveElement] = Field(default_factory=list)

    def to_prompt(self, max_aria: int = SNAPSHOT_URL_MAX_ARIA) -> str:
        aria = self.aria if len(self.aria) <= max_aria else self.aria[:max_aria] + "\n... (truncated)"
        listing = "\n".join(e.line() for e in self.elements[:70])
        return (
            f"PAGE_URL: {self.url}\nPAGE_TITLE: {self.title}\n\nARIA_SNAPSHOT:\n{aria}\n\n"
            f"INTERACTIVE_ELEMENTS (role \"accessible name\" + stable attributes):\n{listing}"
        )


async def capture_snapshot(page: Page) -> PageSnapshot:
    aria = ""
    with contextlib.suppress(PlaywrightError):
        aria = await page.locator("body").aria_snapshot(timeout=5000)
    raw = await page.evaluate(_COLLECT_JS)
    return PageSnapshot(
        url=page.url,
        title=await page.title(),
        aria=aria,
        elements=[InteractiveElement.model_validate(e) for e in raw],
    )


# --------------------------------------------------------------------------------------------- JSON handling
def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in an LLM reply (tolerates fences, prose and <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in the reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(text[start : i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("top-level JSON is not an object")
                return obj
    raise ValueError("unterminated JSON object in the reply")


def normalize_raw(data: dict[str, Any]) -> dict[str, Any]:
    """Fix the usual small-model slips before strict validation."""
    steps = data.get("steps")
    if not isinstance(steps, list):
        return data
    fixed: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        s = dict(step)
        action = str(s.get("action", "")).strip().lower()
        s["action"] = ACTION_SYNONYMS.get(action, action)
        loc = s.get("locator")
        if isinstance(loc, dict):
            loc = {k: v for k, v in loc.items() if v not in ("", "null", None) or k == "name"}
            strategy = str(loc.get("strategy", "")).lower().replace("test_id", "testid").replace("data-testid", "testid")
            loc["strategy"] = strategy
            if loc.get("name") in ("", "null"):
                loc["name"] = None
            if strategy == "role" and str(loc.get("value", "")).lower() not in ARIA_ROLES and not loc.get("name"):
                loc["name"], loc["value"] = loc.get("value"), "button"  # model put the name in value
            if strategy == "role" and loc.get("value"):
                loc["value"] = str(loc["value"]).lower()
            s["locator"] = loc if loc.get("value") else None
        elif loc in ("", "null"):
            s["locator"] = None
        if s["action"] in ("wait", "press") and not (s["action"] == "press" and s.get("locator")):
            s["locator"] = None
        cam = s.get("camera")
        if isinstance(cam, dict):
            cam = dict(cam)
            mode = str(cam.get("mode", "wide")).lower()
            cam["mode"] = {"medium": "focus", "close": "focus", "zoom": "focus", "follow": "pan"}.get(mode, mode)
            try:
                cam["zoom"] = min(2.0, max(1.0, float(cam.get("zoom", 1.0) or 1.0)))
            except (TypeError, ValueError):
                cam["zoom"] = 1.0
            if cam["mode"] not in ("wide", "focus", "pan"):
                cam["mode"] = "focus" if cam["zoom"] > 1.05 else "wide"
            s["camera"] = cam
        elif cam is None:
            s.pop("camera", None)
        if s.get("input_value") is not None and not isinstance(s["input_value"], str):
            s["input_value"] = str(s["input_value"])
        try:
            s["dwell_after_ms"] = int(min(10_000, max(0, float(s.get("dwell_after_ms", 450) or 0))))
        except (TypeError, ValueError):
            s["dwell_after_ms"] = 450
        fixed.append(s)
    data = dict(data)
    data["steps"] = fixed
    data.setdefault("feature_summary", "UI change demo")
    if not isinstance(data.get("start_path"), str) or not data["start_path"]:
        data["start_path"] = "/"
    return data


# --------------------------------------------------------------------------------------------- pacing polish
def _estimate_s(step: ActionStep) -> float:
    base = {"wait": 0.0, "click": 0.85, "hover": 0.7, "press": 0.1, "scroll": 0.35, "fill": 0.9}[step.action]
    if step.action == "fill" and step.input_value:
        base += 0.056 * len(step.input_value)
    return base + step.dwell_after_ms / 1000.0


def polish(sb: Storyboard, *, max_seconds: float = 12.0) -> tuple[Storyboard, list[str]]:
    """Enforce the README pacing rules the prompt asks for, whatever the model returned."""
    notes: list[str] = []
    steps = [s.model_copy(deep=True) for s in sb.steps]
    if steps[0].action != "wait":
        steps.insert(0, ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=400))
        notes.append("added a 400 ms wide opening shot")
    for s in steps:
        if s.action == "fill":
            if s.camera.mode != "focus":
                s.camera = Camera(mode="focus", zoom=1.75)
                notes.append("typing is unreadable wide: switched fill to focus 1.75x")
            if s.input_value and len(s.input_value) > 40:
                s.input_value = s.input_value[:40].rstrip()
        if s.camera.mode == "focus":
            s.camera.zoom = min(1.9, max(1.3, s.camera.zoom))
    last = steps[-1]
    if not (last.action == "wait" and last.camera.zoom <= 1.3 and last.dwell_after_ms >= 1000):
        steps.append(ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1000))
        notes.append("added a 1 s wide closing hold")
    total = sum(_estimate_s(s) for s in steps)
    if total > max_seconds:
        fixed = total - sum(s.dwell_after_ms for s in steps[1:-1]) / 1000.0
        budget = max(0.0, max_seconds - fixed)
        current = sum(s.dwell_after_ms for s in steps[1:-1]) / 1000.0
        ratio = budget / current if current > 0 else 1.0
        for s in steps[1:-1]:
            s.dwell_after_ms = max(150, int(s.dwell_after_ms * ratio))
        notes.append(f"tightened dwell times: ~{total:.1f}s -> ~{sum(_estimate_s(s) for s in steps):.1f}s")
    return Storyboard(feature_summary=sb.feature_summary, start_path=sb.start_path, steps=steps), notes


# --------------------------------------------------------------------------------------------- grounding
def fuzzy_match(loc: Locator, snapshot: PageSnapshot, *, cutoff: float = 0.6) -> InteractiveElement | None:
    """Closest live element for a locator that matched nothing (wrong role, case, small wording drift...)."""
    wanted = (loc.name or loc.value) if loc.strategy == "role" else loc.value
    wanted_l = wanted.lower().strip()
    best: tuple[float, InteractiveElement] | None = None
    for el in snapshot.elements:
        for key in el.keys():
            k = key.lower()
            score = difflib.SequenceMatcher(None, wanted_l, k).ratio()
            if wanted_l and (wanted_l in k or k in wanted_l):
                score = max(score, 0.85)
            if loc.strategy == "role" and el.role == loc.value:
                score += 0.05
            if best is None or score > best[0]:
                best = (score, el)
    return best[1] if best and best[0] >= cutoff else None


@dataclass(slots=True)
class RehearsalResult:
    storyboard: Storyboard
    repairs: list[str] = field(default_factory=list)
    error: str | None = None
    failed_step: int | None = None
    snapshot_at_failure: PageSnapshot | None = None


async def rehearse(browser: Browser, url: str, sb: Storyboard) -> RehearsalResult:
    """Play the storyboard fast; repair locators against the live page; stop at the first unfixable step."""
    context = await browser.new_context(
        viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, device_scale_factor=1, color_scheme="dark"
    )
    steps = [s.model_copy(deep=True) for s in sb.steps]
    result = RehearsalResult(storyboard=sb)
    try:
        page = await context.new_page()
        await page.goto(url, wait_until="load")
        with contextlib.suppress(PlaywrightError):
            await page.wait_for_load_state("networkidle", timeout=3000)
        for i, step in enumerate(steps):
            if step.locator is not None:
                target = resolve_locator(page, step.locator)
                try:
                    await target.wait_for(state="visible", timeout=2500)
                except PlaywrightError:
                    snap = await capture_snapshot(page)
                    match = fuzzy_match(step.locator, snap)
                    if match is None:
                        result.error = f"step {i} ({step.action}): no visible element for {step.locator.describe()}"
                        result.failed_step, result.snapshot_at_failure = i, snap
                        break
                    new = match.best_locator()
                    result.repairs.append(f"step {i}: {step.locator.describe()} -> {new.describe()}")
                    step.locator = new
                    target = resolve_locator(page, new)
                    await target.wait_for(state="visible", timeout=2500)
            try:
                await _quick_action(page, step)
            except PlaywrightError as exc:
                result.error = f"step {i} ({step.action}) failed: {str(exc).splitlines()[0]}"
                result.failed_step, result.snapshot_at_failure = i, await capture_snapshot(page)
                break
        result.storyboard = Storyboard(feature_summary=sb.feature_summary, start_path=sb.start_path, steps=steps)
    finally:
        await context.close()
    return result


async def _quick_action(page: Page, step: ActionStep) -> None:
    target = resolve_locator(page, step.locator) if step.locator else None
    match step.action:
        case "click":
            assert target is not None
            await target.click(timeout=3000)
        case "fill":
            assert target is not None and step.input_value is not None
            await target.click(timeout=3000)
            await target.fill(step.input_value, timeout=3000)
        case "hover":
            assert target is not None
            await target.hover(timeout=3000)
        case "press":
            assert step.input_value is not None
            await page.keyboard.press(step.input_value)
        case "scroll":
            if target is not None:
                await target.scroll_into_view_if_needed(timeout=3000)
            else:
                await page.mouse.wheel(0, float(step.input_value or 400))
        case "wait":
            pass
    await page.wait_for_timeout(120)


# --------------------------------------------------------------------------------------------- orchestration
@dataclass(slots=True)
class PlanResult:
    storyboard: Storyboard
    url: str  # page the storyboard starts on (start_path resolved)
    snapshot: PageSnapshot
    notes: list[str] = field(default_factory=list)
    attempts: int = 1
    raw_replies: list[str] = field(default_factory=list)


def _static_route(route: str) -> bool:
    return route.startswith("/") and not any(c in route for c in "[]:*{}")


def _user_message(diff: DiffSummary, snap: PageSnapshot) -> str:
    path = urlparse(snap.url).path or "/"
    return (
        f"GIT_DIFF_SUMMARY:\n{diff.to_prompt()}\n\n{snap.to_prompt()}\n\n"
        f"The page above is already open at start_path {path!r}. Design the storyboard now. JSON only."
    )


async def plan_storyboard(
    url: str,
    diff: DiffSummary,
    cfg: LLMConfig,
    *,
    max_attempts: int = 3,
    headed: bool = False,
    progress: ProgressFn | None = None,
) -> PlanResult:
    say = progress or (lambda _m: None)
    async with async_playwright() as pw:
        browser = await launch_chromium(pw, headed=headed, progress=say)
        try:
            context = await browser.new_context(
                viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, device_scale_factor=SCALE, color_scheme="dark"
            )
            page = await context.new_page()
            start_url = url
            say(f"opening {url}")
            await page.goto(url, wait_until="load")
            # Pointed at the site root but the diff touches a concrete page? Start there.
            if (urlparse(url).path in ("", "/")) and diff.routes:
                for route in diff.routes[:3]:
                    if _static_route(route) and route != "/":
                        candidate = urljoin(url, route)
                        with contextlib.suppress(PlaywrightError):
                            resp = await page.goto(candidate, wait_until="load")
                            if resp is None or resp.status < 400:
                                start_url = candidate
                                break
                if start_url == url:
                    await page.goto(url, wait_until="load")
            with contextlib.suppress(PlaywrightError):
                await page.wait_for_load_state("networkidle", timeout=4000)
            say("reading the accessibility tree")
            snapshot = await capture_snapshot(page)
            await context.close()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(diff, snapshot)},
            ]
            notes: list[str] = []
            replies: list[str] = []
            last_error = ""
            for attempt in range(1, max_attempts + 1):
                say(f"asking {cfg.model} for a storyboard (attempt {attempt}/{max_attempts})")
                reply = await chat_json(cfg, messages)
                replies.append(reply)
                messages.append({"role": "assistant", "content": reply})
                try:
                    sb = Storyboard.model_validate(normalize_raw(extract_json(reply)))
                except (ValueError, ValidationError) as exc:
                    last_error = f"invalid JSON storyboard: {_short_validation(exc)}"
                    messages.append({"role": "user", "content": f"{last_error}\nReturn the corrected JSON object only."})
                    continue
                parsed = urlparse(start_url)
                start_path = (parsed.path or "/") if parsed.scheme in ("http", "https") else "/"
                if sb.start_path != start_path:
                    notes.append(f"start_path {sb.start_path!r} -> {start_path!r} (page the snapshot came from)")
                    sb.start_path = start_path
                sb, polish_notes = polish(sb)
                notes.extend(polish_notes)
                say("rehearsing the storyboard on the live page")
                rehearsal = await rehearse(browser, start_url, sb)
                notes.extend(f"repaired {r}" for r in rehearsal.repairs)
                if rehearsal.error is None:
                    final, _ = polish(rehearsal.storyboard)
                    return PlanResult(final, start_url, snapshot, notes, attempt, replies)
                last_error = rehearsal.error
                feedback = f"Rehearsal failed: {rehearsal.error}."
                if rehearsal.snapshot_at_failure is not None:
                    feedback += (
                        "\nThis is the page state right before that step:\n"
                        + rehearsal.snapshot_at_failure.to_prompt(max_aria=2500)
                    )
                messages.append({"role": "user", "content": feedback + "\nReturn the corrected full JSON object only."})
            raise PlanError(f"no valid storyboard after {max_attempts} attempts; last error: {last_error}")
        finally:
            await browser.close()


def _short_validation(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
    return str(exc)


def plan_storyboard_sync(
    url: str,
    diff: DiffSummary,
    cfg: LLMConfig,
    *,
    max_attempts: int = 3,
    headed: bool = False,
    progress: ProgressFn | None = None,
) -> PlanResult:
    return asyncio.run(
        plan_storyboard(url, diff, cfg, max_attempts=max_attempts, headed=headed, progress=progress)
    )
