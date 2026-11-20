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
import dataclasses
import difflib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from playwright.async_api import Browser, Page
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, ValidationError

from pullreel.human_runner import SCALE, VIEWPORT, find_locator, launch_chromium, wait_for_content
from pullreel import net
from pullreel.config import saved_key
from pullreel.models import ActionStep, Camera, DiffSummary, Locator, Storyboard
from pullreel.progress import ProgressEvent, Reporter, say, silent


CUSTOM_REQUEST = """USER_SCENARIO (mandatory, written by the user): {request}
The storyboard MUST perform exactly this journey on the page, from start to finish. Do not film another feature.
- If it creates something (add / new / create / ajouter / nouveau / créer): click the button that opens the form or modal,
  `show` the empty form, fill EVERY visible field with realistic, coherent demo data for this domain (a believable
  FICTIONAL name - never a real celebrity -, a plausible number, a date, pick an option in each select / radio /
  chip group). Target each field with strategy "label" and the text written above it (e.g. "Nom"), one fill per
  field, each with a focus camera,
  then click the submit button, then `show` the created item where it now appears (list, table, card): that is the
  money shot, hold it ~2 s.
- If it edits, filters or searches: same pattern, use -> show the result -> refine -> show again.
- Never type credentials, emails other than name@example.com, or anything in a password field."""


SYSTEM_PROMPT = """You are the Demo Director inside PullReel. You write the storyboard of a short screen recording whose only job is to SELL one feature of this branch: a reviewer (or a future user) watching it must think "I want this".

### WHAT A SCENARIO IS
A scenario is NOT a tour of the interface and NOT a list of clicks. It is a tiny story about ONE feature, written backward from its "aha" moment (the result that proves the value):
1. ESTABLISH (wide, ~1 s): the page where the feature lives, untouched.
2. FIRST USE: the simplest use of the feature, immediately followed by a PAYOFF shot: a `show` step that frames the result that just appeared or changed (table, chart, number, list, card) and holds it 1.2-2 s so the viewer can read it.
3. ESCALATE: use the same feature again, each time richer (more items, another option, a harder or more interesting case), each followed by its own `show` payoff. 2 to 4 escalations: this is where the feature proves its depth.
4. MONEY SHOT: the most impressive state of the feature, framed tight, held ~2 s.
5. RESOLUTION: a wide `wait` of ~1.5 s on the final state.
Target 15-30 seconds and 8-18 steps.

### THE MINDSET, ON A REAL EXAMPLE (player-combination analyzer on a coaching page)
BAD (what a robot does): click player "Elias", click player "Brandon", click another tab. Two clicks, the result is never shown, and it leaves the feature.
GOOD (what a director does): wait wide -> click "Elias" (focus) -> click "Brandon" (focus) -> show the combination table (dwell 1600) -> click "Pascal", a 3-player lineup (focus) -> show the table again: the numbers changed (dwell 1400) -> click "Yohan", a 4-player lineup -> show the net-rating / "Écart" column (dwell 1800: the money shot) -> wait wide 1500.
The same pattern applies everywhere: filter -> show the filtered list -> refine the filter -> show again; add an item -> show it in the list -> add another -> show the updated totals; switch a mode -> show what changed.

### RULES
1. ONE concrete user journey. Stay on the feature's page UNLESS the journey itself spans 2-3 pages (e.g., create an entity -> view its detail page -> return to the list); in that case set `"journey": true`. Never click unrelated navigation.
2. Every click/fill must visibly change something, and that change must be shown right after with a `show` step (or a focus shot held >= 1200 ms that already contains it). A click with nothing shown after it is wasted.
3. Zero hallucinated selectors: take targets from INTERACTIVE_ELEMENTS and RESULT_AREAS (and from OBSERVATIONS when provided), with semantic locators (role+name, label, placeholder, text, testid). An element that only appears after one of your own steps may be targeted by the exact name you were told it has.
4. `show` = the camera frames a result area and the cursor points at it, without clicking. Give it a locator to a heading or a text inside the area (e.g. a column header): the camera automatically frames the whole container.
5. Camera rhythm: `focus` (zoom 1.5-1.9) on the control about to be used; `show` with zoom 1.2-1.5 on results; `wide` to establish and to conclude. Vary the rhythm.
6. Realistic, safe data: generate believable, domain-coherent form values (names, numbers, dates, select choices) from labels/placeholders. Always use `@example.com` for emails. NEVER fill passwords, credit cards, secrets, or click destructive actions (delete, remove, reset, sign out). If the journey creates or modifies data, set `"mutates": true`.
7. Add a crisp `"caption"` (3-8 words) on key interaction and payoff steps (null on wide waits) so viewers follow the story effortlessly.
8. `feature_summary` is a selling sentence about the outcome for the user, not a description of clicks.

### STRICT JSON OUTPUT (no markdown, no commentary)
{
  "feature_summary": "Compare any lineup in two clicks and see instantly which combination wins",
  "start_path": "/the-feature-page",
  "journey": false,
  "mutates": false,
  "steps": [
    {
      "action": "wait | click | fill | press | hover | scroll | show",
      "locator": {"strategy": "role | label | placeholder | text | testid | css", "value": "button", "name": "Exact accessible name if strategy is role, else null"},
      "input_value": "Text to type for fill, key name for press (e.g. Enter), or null",
      "camera": {"mode": "wide | focus | pan", "zoom": 1.75},
      "dwell_after_ms": 1200,
      "caption": "Short subtitle shown during this beat, or null"
    }
  ]
}
For `wait` and `press` steps, `locator` is null."""

REFINE_PROMPT = """OBSERVATIONS - your draft was rehearsed on the live app. This is what happened on screen after each step:
{observations}

Now rewrite the FULL storyboard so it tells the story of the feature:
- keep only steps that stay on the feature and change something visible; drop steps that left the page or changed nothing;
- after each meaningful interaction, add a `show` payoff on the result area that appeared or changed (use the exact names above);
- escalate: use the feature 2-4 times, each time richer than the last, then end on the money shot and a wide hold;
- aim for 15-30 seconds. JSON only."""

ARIA_ROLES = {
    "button", "link", "textbox", "searchbox", "checkbox", "radio", "switch", "combobox", "listbox", "option",
    "tab", "tabpanel", "menuitem", "menuitemcheckbox", "menuitemradio", "slider", "spinbutton", "heading",
    "row", "cell", "gridcell", "dialog", "img", "listitem", "treeitem", "progressbar", "navigation", "table",
}  # fmt: skip
ACTION_SYNONYMS = {"type": "fill", "input": "fill", "enter_text": "fill", "tap": "click", "mouseover": "hover",
                   "sleep": "wait", "pause": "wait", "key": "press", "keypress": "press", "highlight": "show",
                   "inspect": "show", "look": "show", "observe": "show", "reveal": "show", "zoom": "show",
                   "focus": "show", "present": "show", "display": "show", "showcase": "show"}  # fmt: skip
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
    if (!['input','textarea','select'].includes(el.tagName.toLowerCase())) return '';
    // unwired <label>/caption: the closest one before the field inside its small container
    for (let box = el.parentElement, d = 0; box && d < 3; box = box.parentElement, d++) {
      const cands = [...box.querySelectorAll('label, legend')].filter(
        (l) => l.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
      const own = cands.filter((l) => !l.control || l.control === el);
      if (own.length) return text(own[own.length - 1].innerText).replace(/[*:]+$/, '').trim();
    }
    return '';
  };
  // First chunk of text the user can actually see (skips tooltips/labels hidden with opacity, visibility or 0 size).
  const visibleText = (root) => {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      const t = text(n.nodeValue);
      const p = n.parentElement;
      if (!t || !p) continue;
      if (p.checkVisibility && !p.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
      const r = p.getBoundingClientRect();
      if (r.width >= 1 && r.height >= 1) return t;
    }
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
  // React/Vue apps often wire clicks on plain <div>/<span>/<li>: take the outermost element with a pointer
  // cursor that is not inside a real control, and name it by its first line of text.
  let scanned = 0;
  for (const el of document.body.getElementsByTagName('*')) {
    if (out.length >= 90 || ++scanned > 6000) break;
    if (seen.has(el) || el.closest(sel)) continue;
    const st = getComputedStyle(el);
    if (st.cursor !== 'pointer') continue;
    const parent = el.parentElement;
    if (parent && getComputedStyle(parent).cursor === 'pointer') continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || st.visibility === 'hidden') continue;
    if (r.bottom < 0 || r.right < 0 || r.top > innerHeight * 2.5 || r.left > innerWidth) continue;
    const first = visibleText(el);
    const aria = text(el.getAttribute('aria-label'));
    const alt = text(el.querySelector('img[alt]')?.getAttribute('alt'));
    const name = aria || first || alt;
    if (!name) continue;  // icon-only, no label: no semantic way to target it
    // Named only by an icon's alt text: target the <img> by role (the click bubbles to the handler).
    const role = !aria && !first && alt ? 'img' : '';
    seen.add(el);
    out.push({
      tag: el.tagName.toLowerCase(), role, name: name.slice(0, 60), label: aria, placeholder: '',
      testid: el.getAttribute('data-testid') || '', clickable: true,
      box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
    });
  }
  return out;
}
"""


_PASSWORD_JS = r"""
() => [...document.querySelectorAll('input[type=password]')].some((e) => {
  const r = e.getBoundingClientRect(), st = getComputedStyle(e);
  return r.width > 1 && r.height > 1 && st.visibility !== 'hidden' && st.display !== 'none';
})
"""
_LOGIN_PATH = re.compile(r"(log-?in|sign-?in|sign-?up|auth|connexion|connect|session|welcome|register)", re.I)


def login_wall(snapshot: PageSnapshot, has_auth: bool) -> bool:
    """True when the page is a login screen we must not script (a password field on a login-looking page,
    or any password field when no saved session was provided)."""
    if not snapshot.has_password_field:
        return False
    return (not has_auth) or bool(_LOGIN_PATH.search(urlparse(snapshot.url).path or ""))


def login_hint(url: str, has_auth: bool) -> str:
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}" if urlparse(url).netloc else url
    if has_auth:
        return (f"{url} still shows a login form although a saved session was loaded: it has probably expired. "
                f"Sign in again with `pullreel login --url {origin}`, then rerun.")
    return (f"{url} shows a login form. PullReel never types credentials. Sign in once yourself with "
            f"`pullreel login --url {origin}` (a browser opens; only the session cookies are saved to "
            f".pullreel/auth.json), then rerun the same command.")


# Result areas the camera can `show`: headings, tables (with their columns) and charts, as the viewer sees them.
_LANDMARKS_JS = r"""
() => {
  const t = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const vis = (e) => {
    const r = e.getBoundingClientRect(), st = getComputedStyle(e);
    return r.width > 4 && r.height > 4 && st.visibility !== 'hidden' && st.display !== 'none'
      && r.bottom > 0 && r.top < innerHeight * 2.5 && (!e.checkVisibility || e.checkVisibility({opacityProperty: true}));
  };
  const out = [];
  for (const h of document.querySelectorAll('h1,h2,h3,h4,h5,h6,[role=heading],caption,legend')) {
    const s = t(h.innerText);
    if (s && s.length <= 80 && vis(h)) out.push(`heading "${s}"`);
  }
  for (const tb of document.querySelectorAll('table,[role=table],[role=grid]')) {
    if (!vis(tb)) continue;
    const cols = [...tb.querySelectorAll('th,[role=columnheader]')].map((c) => t(c.innerText)).filter(Boolean).slice(0, 10);
    const rows = tb.querySelectorAll('tbody tr,[role=row]').length;
    out.push(`table "${cols[0] || ''}" - columns: ${cols.join(' | ')} (${rows} rows)`);
  }
  for (const c of document.querySelectorAll('canvas,svg')) {
    const r = c.getBoundingClientRect();
    if (r.width < 160 || r.height < 100 || !vis(c)) continue;
    let p = c.parentElement, title = '';
    for (let i = 0; i < 5 && p && !title; i++, p = p.parentElement) {
      const h = p.querySelector('h1,h2,h3,h4,h5,h6,[class*=title],[class*=Title]');
      if (h) title = t(h.innerText).slice(0, 60);
    }
    out.push(`chart${title ? ` "${title}"` : ''}`);
  }
  // list entries: a created item ("Lucas Moreau" in the roster) is the payoff of a creation journey
  let items = 0;
  for (const li of document.querySelectorAll('li,[role=listitem]')) {
    if (items >= 30) break;
    const s = t(li.innerText);
    if (s && s.length <= 60 && vis(li)) { out.push(`item "${s}"`); items++; }
  }
  return [...new Set(out)].slice(0, 70);
}
"""


class PlanError(RuntimeError):
    """The LLM could not produce a valid, grounded storyboard."""


class LoginRequiredError(PlanError):
    """The page sits behind a login form: PullReel never types credentials, the user must sign in once."""


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
    provider: str = "ollama"

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    base_url: str
    key_envs: tuple[str, ...] = ()
    default_model: str | None = None  # used when the spec is just "<provider>"
    local: bool = False
    note: str = ""


PROVIDERS: dict[str, Provider] = {
    p.name: p
    for p in (
        Provider("ollama", "http://localhost:11434/v1", (), "qwen2.5-coder:7b", True, "local, free, default"),
        Provider("lmstudio", "http://localhost:1234/v1", (), None, True, "local OpenAI-compatible server"),
        Provider("anthropic", "https://api.anthropic.com/v1", ("ANTHROPIC_API_KEY",), "claude-sonnet-5-5",
                 note="Claude, forced tool-use JSON"),
        Provider("openai", "https://api.openai.com/v1", ("OPENAI_API_KEY",), note="JSON-schema output"),
        Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                 ("GEMINI_API_KEY", "GOOGLE_API_KEY"), note="OpenAI-compatible endpoint"),
        Provider("mistral", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
        Provider("deepseek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",)),
        Provider("openrouter", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",), note="hundreds of models"),
        Provider("groq", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",), note="very fast"),
    )
}  # fmt: skip


def provider_key(p: Provider) -> tuple[str | None, str | None]:
    """(value, env var name) of the first API key found for ``p`` (generic PULLREEL_API_KEY last)."""
    for env in (*p.key_envs, "PULLREEL_API_KEY"):
        if os.environ.get(env):
            return os.environ[env], env
    stored = saved_key(p.name)
    if stored:
        return stored, "saved config"
    return None, None


def parse_model_spec(spec: str, *, base_url: str | None = None, api_key: str | None = None) -> LLMConfig:
    """``ollama/qwen2.5-coder:7b``, ``anthropic`` (default model), ``openai/<model>``... or a bare model name (Ollama)."""
    head, _, model = spec.partition("/")
    if head in PROVIDERS:
        prov = PROVIDERS[head]
        if not model:
            if prov.default_model is None:
                raise LLMError(f"name a model for {head}, e.g. `--model {head}/<model-name>`")
            model = prov.default_model
    else:
        prov, model = PROVIDERS["ollama"], spec
    default_url = prov.base_url
    if prov.name == "ollama" and os.environ.get("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"]
        host = host if host.startswith("http") else f"http://{host}"
        default_url = host.rstrip("/") + "/v1"
    url = base_url or os.environ.get("PULLREEL_BASE_URL") or default_url
    key = api_key or provider_key(prov)[0]
    if not prov.local and not key and not base_url:
        envs = " or ".join(prov.key_envs) or "PULLREEL_API_KEY"
        raise LLMError(f"{prov.name} needs an API key: set {envs} (or PULLREEL_API_KEY), or pass --api-key")
    return LLMConfig(model=model, base_url=url.rstrip("/"), api_key=key, ollama_native=prov.name == "ollama",
                     provider=prov.name, timeout_s=600.0 if prov.local else 180.0)  # fmt: skip


_NOT_CHAT = re.compile(r"(embed|tts|whisper|audio|image|imagen|dall|vision-only|moderation|transcri|realtime|veo|aqa|learnlm)", re.I)


def list_models(provider: str, api_key: str | None, base_url: str | None = None) -> list[str]:
    """Chat-capable model ids the provider says this key can use (empty list when it cannot tell)."""
    prov = PROVIDERS[provider]
    url = (base_url or prov.base_url).rstrip("/")
    try:
        if provider == "anthropic":
            resp = net.get(f"{url}/models", timeout=15, headers={"x-api-key": api_key or "",
                                                                  "anthropic-version": "2023-06-01"})  # fmt: skip
        else:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = net.get(f"{url}/models", timeout=15, headers=headers)
        if resp.status_code in (401, 403):
            raise LLMError(f"{provider} rejected the API key (HTTP {resp.status_code})")
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except httpx.HTTPError as exc:
        raise LLMError(f"cannot list {provider} models: {net.describe(exc)}") from exc
    except ValueError:
        return []
    ids = []
    for m in data:
        mid = str(m.get("id", "")).removeprefix("models/")
        if mid and not _NOT_CHAT.search(mid):
            ids.append(mid)
    return sorted(set(ids), key=lambda x: (x.count("-preview") + x.count("-exp"), x))


TRANSIENT = (429, 500, 502, 503, 504, 529)  # overloaded / rate limited: worth retrying
RETRY_DELAYS = (3.0, 8.0, 20.0)


def api_error(resp: httpx.Response) -> str:
    """The provider's own error sentence instead of a JSON dump."""
    try:
        data = resp.json()
    except ValueError:
        return resp.text.strip()[:300]
    if isinstance(data, list) and data:
        data = data[0]
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, dict):
        return str(err.get("message") or err.get("status") or err)[:300]
    return str(err)[:300]


async def _post_with_retry(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> httpx.Response:
    """POST, waiting and retrying when the provider is overloaded or rate-limiting (honours Retry-After)."""
    resp = await client.post(url, json=body)
    for delay in RETRY_DELAYS:
        if resp.status_code not in TRANSIENT:
            break
        if resp.status_code == 429 and "quota" in resp.text.lower():
            break  # out of quota: waiting will not help
        retry_after = resp.headers.get("retry-after", "")
        wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
        await asyncio.sleep(min(wait, 60.0))
        resp = await client.post(url, json=body)
    return resp


TokenFn = Callable[[int], None]  # called with the number of tokens generated so far


def _schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    return schema if schema is not None else Storyboard.model_json_schema()


async def _ollama_chat(
    cfg: LLMConfig, messages: list[dict[str, str]], on_tokens: TokenFn | None, schema: dict[str, Any] | None
) -> str:
    """Native Ollama chat: grammar-constrained to the JSON schema, big context, no hidden thinking, streamed."""
    root = cfg.base_url.removesuffix("/v1")
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "format": _schema(schema),
        "think": False,
        "stream": True,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
    }
    parts: list[str] = []
    async with net.async_client(timeout=cfg.timeout_s) as client:
        try:
            for _try in range(2):
                async with client.stream("POST", f"{root}/api/chat", json=body) as resp:
                    if resp.status_code >= 400:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        if resp.status_code == 400 and "think" in text and "think" in body:
                            body.pop("think")  # model without a thinking switch: retry once
                            continue
                        if resp.status_code == 404:
                            names: list[str] = []
                            with contextlib.suppress(Exception):
                                tags = (await client.get(f"{root}/api/tags")).json()
                                names = [m["name"] for m in tags.get("models", [])]
                            have = f" Installed: {', '.join(names)}." if names else ""
                            raise LLMError(f"model {cfg.model!r} is not installed: run `ollama pull {cfg.model}`.{have}")
                        raise LLMError(f"{cfg.label} returned HTTP {resp.status_code}: {text[:400]}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise LLMError(f"{cfg.label}: {chunk['error']}")
                        piece = chunk.get("message", {}).get("content", "")
                        if piece:
                            parts.append(piece)
                            if on_tokens:
                                on_tokens(len(parts))
                        if chunk.get("done"):
                            break
                    break
        except httpx.ConnectError as exc:
            raise LLMError(f"cannot reach Ollama at {root}. Start it with `ollama serve`.") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"{cfg.label} timed out after {cfg.timeout_s:.0f}s (try a smaller model)") from exc
    content = "".join(parts)
    if not content:
        raise LLMError(f"{cfg.label} returned an empty message")
    return content


async def _anthropic_chat(
    cfg: LLMConfig, messages: list[dict[str, str]], schema: dict[str, Any] | None, schema_name: str
) -> str:
    """Claude via the Messages API; the answer is forced through a tool whose input schema is ours."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    body: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": 4096,
        "temperature": cfg.temperature,
        "system": system,
        "messages": convo,
        "tools": [{"name": schema_name, "description": f"Return the {schema_name} as structured JSON.",
                   "input_schema": _schema(schema)}],
        "tool_choice": {"type": "tool", "name": schema_name},
    }  # fmt: skip
    headers = {"x-api-key": cfg.api_key or "", "anthropic-version": "2023-06-01", "content-type": "application/json"}
    async with net.async_client(timeout=cfg.timeout_s, headers=headers) as client:
        try:
            resp = await _post_with_retry(client, f"{cfg.base_url}/messages", body)
        except httpx.ConnectError as exc:
            raise LLMError(f"cannot reach {cfg.base_url}: {net.describe(exc)}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"{cfg.label} timed out after {cfg.timeout_s:.0f}s") from exc
    if resp.status_code >= 400:
        hint = " (check ANTHROPIC_API_KEY)" if resp.status_code in (401, 403) else ""
        raise LLMError(f"{cfg.label} returned HTTP {resp.status_code}{hint}: {api_error(resp)}")
    for block in resp.json().get("content", []):
        if block.get("type") == "tool_use":
            return json.dumps(block.get("input", {}))
    raise LLMError(f"{cfg.label} answered without the structured result: {resp.text[:400]}")


async def chat_json(
    cfg: LLMConfig,
    messages: list[dict[str, str]],
    on_tokens: TokenFn | None = None,
    schema: dict[str, Any] | None = None,
    schema_name: str = "storyboard",
) -> str:
    """One chat completion that must return a JSON object matching ``schema`` (default: Storyboard)."""
    if cfg.ollama_native:
        return await _ollama_chat(cfg, messages, on_tokens, schema)
    if cfg.provider == "anthropic":
        return await _anthropic_chat(cfg, messages, schema, schema_name)
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    formats: list[dict[str, Any] | None] = [
        {"type": "json_schema", "json_schema": {"name": schema_name, "schema": _schema(schema)}},
        {"type": "json_object"},
        None,
    ]
    body: dict[str, Any] = {"model": cfg.model, "messages": messages, "temperature": cfg.temperature, "stream": False}
    if cfg.provider == "gemini":
        body["reasoning_effort"] = "none"
    async with net.async_client(timeout=cfg.timeout_s, headers=headers) as client:
        try:
            for fmt in formats:  # strictest structured output the endpoint accepts
                if fmt is None:
                    body.pop("response_format", None)
                else:
                    body["response_format"] = fmt
                resp = await _post_with_retry(client, f"{cfg.base_url}/chat/completions", body)
                if resp.status_code == 400 and "reasoning" in resp.text and "reasoning_effort" in body:
                    body.pop("reasoning_effort")
                    resp = await _post_with_retry(client, f"{cfg.base_url}/chat/completions", body)
                unsupported = resp.status_code in (400, 422) and (
                    "response_format" in resp.text or "json_schema" in resp.text
                )
                if not unsupported:
                    break
        except httpx.ConnectError as exc:
            hint = " Is the server running?" if "localhost" in cfg.base_url else ""
            raise LLMError(f"cannot reach {cfg.base_url}: {net.describe(exc)}.{hint}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"{cfg.label} timed out after {cfg.timeout_s:.0f}s") from exc
        if resp.status_code == 404 and "model" in resp.text.lower():
            raise LLMError(f"{cfg.label}: {api_error(resp)}")
        if resp.status_code in (401, 403):
            raise LLMError(f"{cfg.label} rejected the API key (HTTP {resp.status_code})")
        if resp.status_code >= 400:
            busy = " (provider overloaded - retried 3 times; try again later or pick another model)" \
                if resp.status_code in TRANSIENT else ""
            raise LLMError(f"{cfg.label} returned HTTP {resp.status_code}: {api_error(resp)}{busy}")
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"unexpected response from {cfg.label}: {resp.text[:400]}") from exc
    if not content:
        raise LLMError(f"{cfg.label} returned an empty message")
    return str(content)


async def check_model(cfg: LLMConfig, timeout_s: float = 45.0) -> str | None:
    """None if the model answers a tiny prompt; otherwise the reason (retired, quota, overloaded, bad key...)."""
    probe = dataclasses.replace(cfg, timeout_s=timeout_s)
    try:
        await chat_json(probe, [{"role": "user", "content": 'Reply with {"ok": true}'}],
                        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}}, schema_name="ping")
    except LLMError as exc:
        return str(exc)
    return None


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
    clickable: bool = False  # a plain element with a click handler (pointer cursor), not a real control

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
        kind = self.role or (f"clickable {self.tag}" if self.clickable else self.tag)
        parts = [f"- {kind}"]
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
    has_password_field: bool = False  # a visible <input type=password>: almost always a login wall
    landmarks: list[str] = Field(default_factory=list)  # result areas: headings, tables, charts

    def to_prompt(self, max_aria: int = SNAPSHOT_URL_MAX_ARIA) -> str:
        aria = self.aria if len(self.aria) <= max_aria else self.aria[:max_aria] + "\n... (truncated)"
        listing = "\n".join(e.line() for e in self.elements[:70])
        return (
            f"PAGE_URL: {self.url}\nPAGE_TITLE: {self.title}\n\nARIA_SNAPSHOT:\n{aria}\n\n"
            f"INTERACTIVE_ELEMENTS (role \"accessible name\" + stable attributes):\n{listing}\n\n"
            "RESULT_AREAS (targets for `show`: use a heading or a text inside them as locator):\n"
            + ("\n".join(f"- {x}" for x in self.landmarks[:40]) or "- (none detected)")
        )


async def capture_snapshot(page: Page) -> PageSnapshot:
    aria = ""
    with contextlib.suppress(PlaywrightError):
        aria = await page.locator("body").aria_snapshot(timeout=5000)
    raw = await page.evaluate(_COLLECT_JS)
    has_password = bool(await page.evaluate(_PASSWORD_JS))
    landmarks: list[str] = []
    with contextlib.suppress(PlaywrightError):
        landmarks = list(await page.evaluate(_LANDMARKS_JS))
    return PageSnapshot(
        url=page.url,
        title=await page.title(),
        aria=aria,
        elements=[InteractiveElement.model_validate(e) for e in raw],
        has_password_field=has_password,
        landmarks=landmarks,
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
        if s["action"] == "show" and not s.get("locator"):
            s["action"] = "wait"
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
        cap = s.get("caption")
        if cap in ("", "null", None):
            s["caption"] = None
        else:
            s["caption"] = str(cap).strip()[:120] or None
        fixed.append(s)
    data = dict(data)
    data["steps"] = fixed
    data.setdefault("feature_summary", "UI change demo")
    data["journey"] = bool(data.get("journey", False))
    data["mutates"] = bool(data.get("mutates", False))
    if not isinstance(data.get("start_path"), str) or not data["start_path"]:
        data["start_path"] = "/"
    return data


# --------------------------------------------------------------------------------------------- pacing polish
def _estimate_s(step: ActionStep) -> float:
    base = {"wait": 0.0, "click": 0.85, "hover": 0.7, "press": 0.1, "scroll": 0.35, "fill": 0.9,
            "show": 0.6}[step.action]
    if step.action == "fill" and step.input_value:
        base += 0.056 * len(step.input_value)
    return base + step.dwell_after_ms / 1000.0


def polish(sb: Storyboard, *, max_seconds: float = 30.0) -> tuple[Storyboard, list[str]]:
    """Enforce the README pacing rules the prompt asks for, whatever the model returned."""
    notes: list[str] = []
    steps = [s.model_copy(deep=True) for s in sb.steps]
    if steps[0].action != "wait":
        steps.insert(0, ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=800))
        notes.append("added a wide establishing shot")
    for s in steps:
        if s.action == "fill":
            if s.camera.mode != "focus":
                s.camera = Camera(mode="focus", zoom=1.75)
                notes.append("typing is unreadable wide: switched fill to focus 1.75x")
            if s.input_value and len(s.input_value) > 40:
                s.input_value = s.input_value[:40].rstrip()
        if s.action == "show":
            if s.camera.mode != "focus" or s.camera.zoom < 1.15:
                s.camera = Camera(mode="focus", zoom=1.35)
            s.dwell_after_ms = max(s.dwell_after_ms, 1000)  # a payoff must be readable
        elif s.action in ("click", "fill", "press"):
            s.dwell_after_ms = max(s.dwell_after_ms, 500)  # let the UI react before the next beat
        if s.camera.mode == "focus":
            s.camera.zoom = min(1.9, max(1.15 if s.action == "show" else 1.3, s.camera.zoom))
    last = steps[-1]
    if not (last.action == "wait" and last.camera.zoom <= 1.3 and last.dwell_after_ms >= 1200):
        steps.append(ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1500))
        notes.append("added a wide closing hold")
    total = sum(_estimate_s(s) for s in steps)
    if total > max_seconds:
        fixed = total - sum(s.dwell_after_ms for s in steps[1:-1]) / 1000.0
        budget = max(0.0, max_seconds - fixed)
        current = sum(s.dwell_after_ms for s in steps[1:-1]) / 1000.0
        ratio = budget / current if current > 0 else 1.0
        for s in steps[1:-1]:
            s.dwell_after_ms = max(150, int(s.dwell_after_ms * ratio))
        notes.append(f"tightened dwell times: ~{total:.1f}s -> ~{sum(_estimate_s(s) for s in steps):.1f}s")
    return Storyboard(
        feature_summary=sb.feature_summary,
        start_path=sb.start_path,
        steps=steps,
        journey=sb.journey,
        mutates=sb.mutates,
    ), notes


# --------------------------------------------------------------------------------------------- grounding
EDITABLE_ROLES = {"textbox", "searchbox", "spinbutton", "combobox"}


def fuzzy_match(loc: Locator, snapshot: PageSnapshot, *, cutoff: float = 0.6,
                editable: bool = False) -> InteractiveElement | None:
    """Closest live element for a locator that matched nothing (wrong role, case, small wording drift...)."""
    wanted = (loc.name or loc.value) if loc.strategy == "role" else loc.value
    wanted_l = wanted.lower().strip()
    best: tuple[float, InteractiveElement] | None = None
    for el in snapshot.elements:
        if editable and el.role not in EDITABLE_ROLES:
            continue
        for key in el.keys():
            k = key.lower()
            score = difflib.SequenceMatcher(None, wanted_l, k).ratio()
            if len(k) < 3 and k != wanted_l:
                continue  # initials, badges, single letters: never a credible repair
            if wanted_l and (wanted_l in k or (k in wanted_l and len(k) * 2 >= len(wanted_l))):
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
    observations: list[str] = field(default_factory=list)  # what each step changed on screen
    left_page_at: int | None = None  # first step that navigated away from the feature page
    created: str | None = None  # text of the item a submit created ("Lucas Moreau" in the list)


def left_feature(start_path: str, now_path: str) -> bool:
    """True when navigation left the feature: sub-routes of the start page ("/stats" -> "/stats/sheet") stay in it."""
    root = (start_path or "/").rstrip("/")
    now = (now_path or "/").rstrip("/")
    if not root:
        return now != ""
    return not (now == root or now.startswith(root + "/"))


async def _screen_signature(page: Page) -> set[str]:
    """What a viewer can see and use: result areas + interactive elements, as comparable strings."""
    sig: set[str] = set()
    with contextlib.suppress(PlaywrightError):
        sig.update(await page.evaluate(_LANDMARKS_JS))
        for e in await page.evaluate(_COLLECT_JS):
            if e.get("name"):
                kind = e.get("role") or ("clickable" if e.get("clickable") else e.get("tag", ""))
                sig.add(f'{kind} "{e["name"]}"')
    return sig


# After a submit: where do the values the story typed now show up, outside the form? (the created item)
_CREATED_JS = r"""
(values) => {
  const t = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const inForm = (e) => e.closest('input,textarea,select,form,[role=dialog],[aria-modal=true]');
  const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const all = values.filter(Boolean);
  let best = null;  // the tightest text that names the item ("Lucas Moreau", not the whole list around it)
  for (const leaf of document.body.querySelectorAll('*')) {
    if (leaf.children.length > 2 || inForm(leaf) || !vis(leaf)) continue;
    const own = t(leaf.innerText);
    if (!own || own.length > 60 || !own.includes(all[0]) || (best && own.length >= best.length)) continue;
    let box = leaf;
    for (let d = 0; d < 4 && box; d++, box = box.parentElement) {
      const txt = box.innerText || '';
      if (all.every((v) => txt.includes(v))) { best = own; break; }
    }
  }
  return best;
}
"""

SUBMIT_WORDS = ("ajout", "add", "save", "enregistr", "créer", "creer", "create", "valider", "submit", "confirm",
                "envoyer", "send", "publier", "publish", "ok")


async def _created_item(page: Page, typed: list[str], timeout_s: float = 6.0) -> str | None:
    """Text of the element that now shows what was typed (e.g. the new player's name in the roster)."""
    values = [v for v in typed if len(v) >= 3 and not v.replace(".", "").isdigit()][:3]
    if not values:
        return None
    deadline = time.monotonic() + timeout_s
    while True:
        with contextlib.suppress(PlaywrightError):
            found = await page.evaluate(_CREATED_JS, values)
            if found:
                return str(found)
        if time.monotonic() > deadline:
            return None
        await page.wait_for_timeout(400)


def with_money_shot(sb: Storyboard, created: str | None) -> tuple[Storyboard, list[str]]:
    """A creation story ends on the created item: retarget (or add) the last `show` after the submit onto it."""
    if not created:
        return sb, []
    steps = [s.model_copy(deep=True) for s in sb.steps]
    if any(s.action == "show" and s.locator and created in (s.locator.name or s.locator.value) for s in steps):
        return sb, []
    last_click = max((i for i, s in enumerate(steps) if s.action == "click"), default=-1)
    shot = Locator(strategy="text", value=created)
    later_shows = [i for i, s in enumerate(steps) if s.action == "show" and i > last_click]
    if later_shows:
        steps[later_shows[-1]].locator = shot
        steps[later_shows[-1]].dwell_after_ms = max(steps[later_shows[-1]].dwell_after_ms, 2000)
    else:
        at = len(steps) - 1 if steps and steps[-1].action == "wait" else len(steps)
        steps.insert(at, ActionStep(action="show", locator=shot, camera=Camera(mode="focus", zoom=1.4),
                                    dwell_after_ms=2000))  # fmt: skip
    note = f"money shot: the created item {created!r}"
    return Storyboard(feature_summary=sb.feature_summary, start_path=sb.start_path, steps=steps), [note]


async def rehearse(
    browser: Browser,
    url: str,
    sb: Storyboard,
    report: Reporter = silent,
    auth: Path | None = None,
    *,
    dry_rehearsal: bool = False,
    extra_blocklist: tuple[str, ...] | list[str] = (),
) -> RehearsalResult:
    """Play the storyboard fast; repair locators against the live page; stop at the first unfixable step.

    Supports multi-page journeys (``sb.journey=True``, up to 3 pages), extended destructive-action blocking,
    and ``dry_rehearsal=True`` (verifies final form submit buttons without clicking them).
    """
    from pullreel.journeys import _SUBMIT_RE, detect_mutating_storyboard, is_destructive, is_sensitive_field

    context = await browser.new_context(
        viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]},
        device_scale_factor=1,
        color_scheme="dark",
        storage_state=str(auth) if auth else None,
    )
    await context.add_init_script(_INVALID_WATCH)
    steps = [s.model_copy(deep=True) for s in sb.steps]
    result = RehearsalResult(storyboard=sb)
    visited_paths: list[str] = [urlparse(url).path or "/"]
    filled_form = False
    skipped_submit = False
    try:
        page = await context.new_page()
        await page.goto(url, wait_until="load")
        await wait_for_content(page)
        typed: list[str] = []
        for i, step in enumerate(steps):
            what = f"{step.action} {step.locator.describe()}" if step.locator else step.action
            say(report, "rehearsal", detail=f"step {i + 1}/{len(steps)} · {what}", completed=i, total=len(steps))
            if skipped_submit and step.action == "show":
                # Post-submit show step cannot find an unsubmitted item during dry rehearsal.
                result.observations.append(f"step {i} {what}: dry-rehearsal skipped post-submit check")
                continue
            if step.locator is not None:
                loc_text = step.locator.name or step.locator.value
                if step.action == "click" and is_destructive(loc_text, extra_blocklist):
                    result.error = (
                        f"step {i} clicks a destructive/blocked control ({step.locator.describe()}): "
                        "PullReel never clicks delete/reset/logout actions"
                    )
                    result.failed_step, result.snapshot_at_failure = i, await capture_snapshot(page)
                    break
                target = await find_locator(page, step.locator)
                try:
                    await target.wait_for(state="visible", timeout=2500)
                except PlaywrightError:
                    snap = await capture_snapshot(page)
                    match = fuzzy_match(step.locator, snap, editable=step.action == "fill")
                    if match is None:
                        result.error = f"step {i} ({step.action}): no visible element for {step.locator.describe()}"
                        result.failed_step, result.snapshot_at_failure = i, snap
                        break
                    new = match.best_locator()
                    result.repairs.append(f"step {i}: {step.locator.describe()} -> {new.describe()}")
                    step.locator = new
                    target = await find_locator(page, new)
                    try:
                        await target.wait_for(state="visible", timeout=2500)
                    except PlaywrightError:
                        result.error = (f"step {i} ({step.action}): {new.describe()} exists but is not visible "
                                        "(hidden label or tooltip) - pick an element that is visible on screen")
                        result.failed_step, result.snapshot_at_failure = i, snap
                        break
            if step.action == "fill" and step.locator is not None:
                kind, fname = "", ""
                with contextlib.suppress(PlaywrightError):
                    el = await find_locator(page, step.locator)
                    kind = (await el.get_attribute("type", timeout=1000)) or ""
                    fname = (await el.get_attribute("name", timeout=1000)) or ""
                if is_sensitive_field(step.locator.name or step.locator.value, fname, "", kind):
                    result.error = (f"step {i} types into a sensitive/password field ({step.locator.describe()}): "
                                    "never fill credentials or payment details, show the feature itself")
                    result.failed_step, result.snapshot_at_failure = i, await capture_snapshot(page)
                    break
                filled_form = True
            has_later_fill = any(s.action == "fill" for s in steps[i + 1 :])
            if (
                dry_rehearsal
                and step.action == "click"
                and step.locator is not None
                and filled_form
                and not has_later_fill
                and _SUBMIT_RE.search(step.locator.name or step.locator.value)
            ):
                # Dry rehearsal: verify submit target exists & is enabled, without mutating app state.
                el = await find_locator(page, step.locator)
                if await el.is_disabled(timeout=1500):
                    result.error = f"step {i} ({step.action}): submit control {step.locator.describe()} is disabled"
                    result.failed_step, result.snapshot_at_failure = i, await capture_snapshot(page)
                    break
                skipped_submit = True
                result.observations.append(f"step {i} {what}: dry-rehearsal verified submit (not clicked)")
                continue
            if step.action == "fill" and step.locator is not None:
                owner = None
                with contextlib.suppress(PlaywrightError):
                    owner = await (await find_locator(page, step.locator)).evaluate(
                        "(e, i) => { const o = e.dataset.pullreelFilled; e.dataset.pullreelFilled = i; return o ?? null; }",
                        str(i))  # fmt: skip
                if owner is not None and owner != str(i):
                    result.error = (f"steps {owner} and {i} type into the SAME field ({step.locator.describe()}): "
                                    "the second erases the first. Target each field by its own label "
                                    "(strategy 'label' with the text written above the field)")
                    result.failed_step, result.snapshot_at_failure = i, await capture_snapshot(page)
                    break
            before = await _screen_signature(page) if step.action in ("click", "fill", "press") else None
            path_before = urlparse(page.url).path
            if step.action == "fill" and step.input_value:
                typed.append(step.input_value)
            try:
                await _quick_action(page, step)
            except PlaywrightError as exc:
                result.error = f"step {i} ({step.action}) failed: {str(exc).splitlines()[0]}"
                result.failed_step, result.snapshot_at_failure = i, await capture_snapshot(page)
                break
            if before is not None:
                await page.wait_for_timeout(450)
                await wait_for_content(page, timeout_s=3.0, quiet_s=0.5)
                after = await _screen_signature(page)
                path_after = urlparse(page.url).path
                line = f"step {i} {what}: "
                if path_after != path_before:
                    if path_after not in visited_paths:
                        visited_paths.append(path_after)
                    if sb.journey and len(visited_paths) <= 4:
                        line += f"navigated: {path_before} -> {path_after}. "
                    elif left_feature(urlparse(url).path, path_after):
                        line += f"LEFT THE FEATURE PAGE ({path_before} -> {path_after}). "
                        if result.left_page_at is None:
                            result.left_page_at = i
                invalid = await _invalid_fields(page) if step.action == "click" else []
                if invalid:
                    result.error = (f"step {i} ({what}) submitted a form that is still invalid - "
                                    + "; ".join(invalid) + ". Fill those fields (by label) before submitting")
                    result.failed_step, result.snapshot_at_failure = i, await capture_snapshot(page)
                    break
                appeared = sorted(after - before, key=lambda x: (not x.startswith(("table", "chart", "heading")), x))
                gone = before - after
                label = ((step.locator.name or step.locator.value) if step.locator else "").lower()
                if step.action == "click" and typed and result.created is None \
                        and any(w in label for w in SUBMIT_WORDS):
                    result.created = await _created_item(page, typed)
                    if result.created:
                        appeared.insert(0, f"CREATED ITEM: text {result.created!r} (show it: the money shot)")
                line += ("appeared: " + "; ".join(appeared[:12])) if appeared else "nothing new appeared"
                if gone:
                    line += f" ({len(gone)} things disappeared)"
                result.observations.append(line)
        final_sb = Storyboard(
            feature_summary=sb.feature_summary,
            start_path=sb.start_path,
            steps=steps,
            journey=sb.journey or len(visited_paths) > 1,
            mutates=sb.mutates,
        )
        final_sb.mutates = detect_mutating_storyboard(final_sb)
        result.storyboard = final_sb
    finally:
        await context.close()
    return result


# Browsers fire `invalid` on each bad field when a submit is attempted (and block it): record them.
_INVALID_WATCH = r"""
window.__pullreelInvalid = [];
document.addEventListener('invalid', (ev) => {
  const e = ev.target;
  const lab = (e.labels && e.labels[0]) || e.closest('div')?.querySelector('label');
  const name = (e.getAttribute('aria-label') || lab?.innerText || e.placeholder || e.name || 'field')
    .replace(/\s+/g, ' ').replace(/[*:]+\s*$/, '').trim();
  window.__pullreelInvalid.push(`field "${name}": ${e.validationMessage}`);
}, true);
"""


async def _invalid_fields(page: Page) -> list[str]:
    """Fields the browser refused when the last click tried to submit their form (then forget them)."""
    try:
        found = await page.evaluate("() => { const f = window.__pullreelInvalid || []; window.__pullreelInvalid = []; return f; }")
    except PlaywrightError:
        return []
    return list(dict.fromkeys(found))[:5]


async def _quick_action(page: Page, step: ActionStep) -> None:
    target = await find_locator(page, step.locator) if step.locator else None
    match step.action:
        case "click":
            assert target is not None
            await target.click(timeout=3000)
        case "fill":
            assert target is not None and step.input_value is not None
            tag = ""
            with contextlib.suppress(PlaywrightError):
                tag = str(await target.evaluate("e => e.tagName.toLowerCase()"))
            if tag == "select":
                await target.select_option(label=step.input_value, timeout=3000)
            else:
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
        case "wait" | "show":
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
    ai_down: str | None = None  # the model could not be reached (connection, key...): later scenarios skip it
    chapter: str | None = None  # sub-feature of the page this storyboard shows ("main view", a tab...)
    extras: list[tuple[str, Storyboard]] = field(default_factory=list)  # other strong chapters of the page


def _static_route(route: str) -> bool:
    return route.startswith("/") and not any(c in route for c in "[]:*{}")


# --------------------------------------------------------------------------------------------- feature scout
SCOUT_PROMPT = """You are the Feature Scout inside PullReel, a tool that films short UI demos of a git branch for its pull request.
Read the project description, the commit messages and the feature candidates (changed files grouped by the page where they show up), then pick the {n} most demo-worthy, user-visible changes.

### RULES
1. Prefer brand-new screens or interactions over small tweaks; the commits tell you what the author meant to ship.
2. Each pick must be visible and interactive in a 5-12 second clip of the running app. Skip refactors, styles-only, tests, build, data/sync plumbing without UI.
3. Picks must be distinct (different pages or clearly different flows). Strongly prefer candidates whose route has no parameter (no ":" or "["): the camera can open "/coaching" directly, but not "/match/:id".
4. `route` MUST be copied from a candidate's route (or null if the feature is not tied to one page). Never invent routes.
5. `why`: one sentence a reviewer would care about. `labels`: 2-5 on-screen texts (button names, tab names, headings) that the demo should click or show, taken from the candidate labels when possible.
6. Order picks best first. Respond ONLY with JSON: {{"features": [{{"title": "...", "route": "/path or null", "why": "...", "labels": ["..."]}}]}}"""


class ScoutPick(BaseModel):
    title: str
    route: str | None = None
    why: str = ""
    labels: list[str] = Field(default_factory=list)
    recommended: bool = True  # chosen by the scout model (False = topped up from the scored candidates)
    sales_score: float | None = None  # 0..100 estimated demo sales value
    mutates: bool = False  # True if the scenario creates/modifies data
    user_requested: bool = False  # True for --focus or "Write my own scenario..." (never replaced by composer)

    def to_prompt(self) -> str:
        if self.user_requested:
            return CUSTOM_REQUEST.format(request=self.title)
        lines = [f"FEATURE_TO_SHOW: {self.title}"]
        if self.why:
            lines.append(f"WHY IT MATTERS: {self.why}")
        if self.labels:
            lines.append("ON-SCREEN TEXTS TO USE: " + ", ".join(repr(x) for x in self.labels[:6]))
        lines.append("Build the storyboard around this feature only; ignore the other changes.")
        return "\n".join(lines)


class ScoutResult(BaseModel):
    features: list[ScoutPick] = Field(min_length=1)


def _pick_from_candidate(diff: DiffSummary, index: int = 0) -> ScoutPick:
    from pullreel.journeys import score_candidate_sales

    f = diff.features[index]
    why = f.commits[0] if f.commits else f"{f.added} lines changed"
    return ScoutPick(
        title=f.title,
        route=f.route,
        why=why,
        labels=f.labels[:5],
        sales_score=score_candidate_sales(candidate=f, title=f.title),
    )


def match_focus(diff: DiffSummary, focus: str) -> ScoutPick:
    """Turn a user's ``--focus "..."`` into a pick, attaching the route of the candidate it best matches."""
    wanted = focus.lower().strip()
    best: tuple[float, int] | None = None
    for i, f in enumerate(diff.features):
        keys = [f.title, f.route or "", *f.labels, *f.commits]
        score = max((difflib.SequenceMatcher(None, wanted, k.lower()).ratio() for k in keys if k), default=0.0)
        if any(wanted in k.lower() for k in keys if k):
            score = max(score, 0.9)
        if best is None or score > best[0]:
            best = (score, i)
    route = diff.features[best[1]].route if best and best[0] >= 0.5 else None
    return ScoutPick(title=focus, route=route, why="requested with --focus", labels=[], user_requested=True)


async def scout_features(diff: DiffSummary, cfg: LLMConfig, n: int, report: Reporter = silent) -> list[ScoutPick]:
    """Ask the model which ``n`` features to film (deterministic when there is nothing to choose from)."""
    if not diff.features:
        say(report, "scout", "done", "no feature candidates - filming the page given by --url")
        return [ScoutPick(title="Changes on this branch", route=None)]
    if n >= len(diff.features):  # "all": nothing to choose, film every page we can open directly
        picks = [_pick_from_candidate(diff, i) for i in range(len(diff.features))]
        openable = [p for p in picks if p.route and _static_route(p.route)]
        skipped = [p.title for p in picks if p not in openable]
        if openable:
            extra = f" · skipped (no direct URL): {', '.join(skipped)}" if skipped else ""
            say(report, "scout", "done", f"all {len(openable)} feature(s){extra}")
            return openable
    if len(diff.features) <= n and n == 1:
        pick = _pick_from_candidate(diff)
        say(report, "scout", "done", f"only one candidate: {pick.title}")
        return [pick]
    say(report, "scout", "start", f"{cfg.model} · choosing {n} feature(s) among {len(diff.features)} candidates")
    messages = [
        {"role": "system", "content": SCOUT_PROMPT.format(n=n)},
        {"role": "user", "content": f"GIT_DIFF_SUMMARY:\n{diff.to_prompt()}\n\nPick {n} feature(s). JSON only."},
    ]
    known = {f.route for f in diff.features if f.route}
    picks: list[ScoutPick] = []
    try:
        def on_tokens(k: int) -> None:
            say(report, "scout", detail=f"{cfg.model} · writing · {k} tokens")

        reply = await chat_json(cfg, messages, on_tokens, ScoutResult.model_json_schema(), "feature_picks")
        picks = ScoutResult.model_validate(extract_json(reply)).features
    except LLMError as exc:
        say(report, "scout", detail=f"AI unavailable ({exc}) - using the scored candidates")
    except (ValueError, ValidationError):
        say(report, "scout", detail="unusable AI answer - using the scored candidates")
    cleaned: list[ScoutPick] = []
    seen: set[str] = set()
    for p in picks:
        if p.route in ("", "null"):
            p.route = None
        if p.route is not None and p.route not in known:
            p.route = None  # never trust an invented route
        key = (p.route or p.title).lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(p)
    # Top up (or replace a failed answer) with the best-scored candidates.
    for i in range(len(diff.features)):
        if len(cleaned) >= n:
            break
        cand = _pick_from_candidate(diff, i)
        cand.recommended = False
        key = (cand.route or cand.title).lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(cand)
    cleaned = cleaned[:n]
    say(report, "scout", "done", " · ".join(f"{i}. {p.title}" for i, p in enumerate(cleaned, 1)))
    return cleaned


def _user_message(diff: DiffSummary, snap: PageSnapshot, feature: ScoutPick | None = None) -> str:
    path = urlparse(snap.url).path or "/"
    focus = f"{feature.to_prompt()}\n\n" if feature is not None else ""
    return (
        f"{focus}GIT_DIFF_SUMMARY:\n{diff.to_prompt()}\n\n{snap.to_prompt()}\n\n"
        f"The page above is already open at start_path {path!r}. Design the storyboard now. JSON only."
    )


async def plan_storyboard(
    url: str,
    diff: DiffSummary,
    cfg: LLMConfig,
    *,
    max_attempts: int = 4,
    headed: bool = False,
    report: Reporter = silent,
    auth: Path | None = None,
    feature: ScoutPick | None = None,
    refine: bool = True,
    director: str = "auto",
    chapters_per_page: int = 3,
    dry_rehearsal: bool = False,
    extra_blocklist: tuple[str, ...] | list[str] = (),
) -> PlanResult:
    user_requested = bool(feature is not None and feature.user_requested)
    if user_requested and director == "composer":
        raise PlanError(
            f"Cannot film custom scenario {feature.title!r} with --director composer: "  # type: ignore[union-attr]
            "custom requests (--focus / 'Write my own') require an AI director so PullReel follows your exact intent. "
            "Use --director ai or --director auto with a reachable model."
        )
    say(report, "snapshot", "start", "launching Chromium")
    async with async_playwright() as pw:
        browser = await launch_chromium(pw, headed=headed, progress=lambda m: say(report, "snapshot", detail=m))
        try:
            context = await browser.new_context(
                viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]},
                device_scale_factor=SCALE,
                color_scheme="dark",
                storage_state=str(auth) if auth else None,
            )
            page = await context.new_page()
            start_url = url
            say(report, "snapshot", detail=f"opening {url}")
            await page.goto(url, wait_until="load")
            http = urlparse(url).scheme in ("http", "https")
            if feature is not None and feature.route and _static_route(feature.route) and http:
                candidate = urljoin(url, feature.route)
                with contextlib.suppress(PlaywrightError):
                    resp = await page.goto(candidate, wait_until="load")
                    if resp is None or resp.status < 400:
                        start_url = candidate
            # Pointed at the site root but the diff touches a concrete page? Start there.
            elif (urlparse(url).path in ("", "/")) and diff.routes:
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
            say(report, "snapshot", detail="waiting for the page to finish rendering")
            await wait_for_content(page)
            say(report, "snapshot", detail="reading the accessibility tree")
            snapshot = await capture_snapshot(page)
            await context.close()
            shown = urlparse(start_url).path if urlparse(start_url).scheme in ("http", "https") else "page"
            if login_wall(snapshot, auth is not None):
                landed = urlparse(snapshot.url).path or shown or "/"
                say(report, "snapshot", "fail", f"{landed} is a login screen (asked for {shown or '/'})")
                raise LoginRequiredError(login_hint(snapshot.url, auth is not None))
            session = " · signed in (saved session)" if auth else ""
            say(report, "snapshot", "done", f"{shown or '/'} · {len(snapshot.elements)} interactive elements{session}")

            snapshot_path = (urlparse(start_url).path or "/") if urlparse(start_url).scheme in ("http", "https") else "/"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(diff, snapshot, feature)},
            ]
            notes: list[str] = []
            replies: list[str] = []

            async def ai_plan() -> tuple[Storyboard, list[str], int]:
                last_error = ""
                draft: Storyboard | None = None  # a storyboard that already rehearsed fine
                draft_obs: list[str] = []
                for attempt in range(1, max_attempts + 1):
                    tag = f"{cfg.model}" + (f" · attempt {attempt}/{max_attempts}" if attempt > 1 else "")
                    say(report, "storyboard", "start", f"{tag} · reading prompt")
                    t_ask = time.monotonic()
                    first: list[float] = []

                    def on_tokens(n: int, tag: str = tag, t_ask: float = t_ask, first: list[float] = first) -> None:
                        now = time.monotonic()
                        if not first:
                            first.append(now)
                        rate = n / max(now - first[0], 1e-3) if n > 1 else 0.0
                        speed = f" · {rate:.1f} tok/s" if rate else ""
                        say(report, "storyboard", detail=f"{tag} · writing · {n} tokens{speed}")

                    reply = await chat_json(cfg, messages, on_tokens)
                    replies.append(reply)
                    messages.append({"role": "assistant", "content": reply})
                    try:
                        sb = Storyboard.model_validate(normalize_raw(extract_json(reply)))
                    except (ValueError, ValidationError) as exc:
                        last_error = f"invalid JSON storyboard: {_short_validation(exc)}"
                        say(report, "storyboard", detail=f"{tag} · invalid JSON, asking again")
                        messages.append({"role": "user", "content": f"{last_error}\nReturn the corrected JSON object only."})
                        continue
                    say(report, "storyboard", "done",
                        f"{len(sb.steps)} steps · {sb.feature_summary[:60]} · {time.monotonic() - t_ask:.0f}s")  # fmt: skip
                    parsed = urlparse(start_url)
                    start_path = (parsed.path or "/") if parsed.scheme in ("http", "https") else "/"
                    if sb.start_path != start_path:
                        notes.append(f"start_path {sb.start_path!r} -> {start_path!r} (page the snapshot came from)")
                        sb.start_path = start_path
                    sb, polish_notes = polish(sb)
                    notes.extend(polish_notes)
                    say(report, "rehearsal", "start", "replaying at full speed", 0, len(sb.steps))
                    rehearsal = await rehearse(
                        browser, start_url, sb, report, auth,
                        dry_rehearsal=dry_rehearsal, extra_blocklist=extra_blocklist,
                    )
                    notes.extend(f"repaired {r}" for r in rehearsal.repairs)
                    if rehearsal.error is None:
                        fixed = f" · {len(rehearsal.repairs)} locator(s) repaired" if rehearsal.repairs else ""
                        final, _ = polish(rehearsal.storyboard)
                        final, shot_notes = with_money_shot(final, rehearsal.created)
                        notes.extend(shot_notes)
                        if refine and draft is None and attempt < max_attempts and rehearsal.observations:
                            # Draft works: now let the model see what its clicks produced, and direct for real.
                            draft, draft_obs = final, list(rehearsal.observations)
                            say(report, "rehearsal", "done",
                                f"draft OK · {len(rehearsal.observations)} result(s) observed -> refining the story{fixed}")  # fmt: skip
                            messages.append({"role": "user", "content": REFINE_PROMPT.format(
                                observations="\n".join(rehearsal.observations))})  # fmt: skip
                            continue
                        if rehearsal.left_page_at is not None and draft is not None:
                            notes.append("refined story left the feature page; kept it anyway (it rehearsed fine)")
                        stage = "story refined" if draft is not None else "all steps resolve on the live page"
                        say(report, "rehearsal", "done", f"{stage} · {len(final.steps)} steps{fixed}")
                        return final, rehearsal.observations, attempt
                    if draft is not None:  # the refined version broke: the draft is still a valid film
                        notes.append(f"refined storyboard failed rehearsal ({rehearsal.error}); kept the draft")
                        say(report, "rehearsal", "done", f"kept the working draft · {len(draft.steps)} steps")
                        return draft, draft_obs, attempt
                    last_error = rehearsal.error
                    say(report, "rehearsal", "fail", f"{rehearsal.error} -> back to the model")
                    feedback = f"Rehearsal failed: {rehearsal.error}."
                    if rehearsal.snapshot_at_failure is not None:
                        feedback += (
                            "\nThis is the page state right before that step:\n"
                            + rehearsal.snapshot_at_failure.to_prompt(max_aria=2500)
                        )
                    messages.append({"role": "user", "content": feedback + "\nReturn the corrected full JSON object only."})
                if draft is not None:
                    return draft, draft_obs, max_attempts
                raise PlanError(f"no valid storyboard after {max_attempts} attempts; last error: {last_error}")

            from pullreel.composer import GOOD_STORY, compose_chapter, explore_page, story_quality
            from pullreel.journeys import detect_mutating_storyboard, explore_journeys

            main_title: str | None = None
            best: tuple[Storyboard, float, int] | None = None
            ai_error: PlanError | None = None
            ai_down: str | None = None
            if director != "composer":
                try:
                    sb_ai, obs_ai, attempts = await ai_plan()
                    best = (sb_ai, story_quality(sb_ai, obs_ai), attempts)
                except PlanError as exc:
                    ai_error = exc
                    if user_requested:
                        raise PlanError(
                            f"Could not build the requested scenario {feature.title!r}: {exc} "  # type: ignore[union-attr]
                            "(PullReel will not replace your custom scenario with an unrelated auto-composed demo)."
                        ) from exc
                except LLMError as exc:
                    if director == "ai" or user_requested:
                        if user_requested:
                            raise PlanError(
                                f"AI failed while planning your custom scenario {feature.title!r}: {exc} "  # type: ignore[union-attr]
                                "(PullReel will not substitute an unrelated demo)."
                            ) from exc
                        raise
                    ai_error = PlanError(str(exc))
                    ai_down = str(exc)
                    notes.append(f"AI unavailable ({exc}) - story composed automatically")
                    say(report, "storyboard", "fail", f"AI unavailable: {exc}")

            extras: list[tuple[str, Storyboard, float]] = []
            # Never override or augment a custom user-requested scenario with automatic composer discoveries.
            if director != "ai" and not user_requested:
                say(report, "rehearsal", "start", "discovering concrete user journeys and interactive chapters...")
                composed: list[tuple[str, Storyboard, float]] = []
                # 1. Concrete workflows (create forms, search/filter, etc.)
                journeys = await explore_journeys(
                    browser, start_url, auth=auth, dry_rehearsal=dry_rehearsal,
                    diff=diff, extra_blocklist=extra_blocklist,
                )
                for jp in journeys:
                    sb_j, _ = polish(jp.storyboard)
                    trial = await rehearse(
                        browser, start_url, sb_j, report, auth,
                        dry_rehearsal=dry_rehearsal, extra_blocklist=extra_blocklist,
                    )
                    if trial.error is not None:
                        notes.append(f"journey {jp.workflow.title!r} failed rehearsal: {trial.error}")
                        continue
                    q = story_quality(trial.storyboard, trial.observations) + 2.5
                    composed.append((jp.workflow.title, polish(trial.storyboard)[0], q))

                # 2. Control-group chapters
                chapters = await explore_page(
                    browser, start_url, auth=auth, max_chapters=max(chapters_per_page, 1) + 1,
                    extra_blocklist=extra_blocklist,
                )
                summary = (feature.why or feature.title) if feature is not None else "See the feature in action"
                for chapter in chapters:
                    sb_c, _ = polish(compose_chapter(chapter, summary, snapshot_path))
                    say(report, "rehearsal", "start", f"chapter {chapter.title!r}: rehearsing", 0, len(sb_c.steps))
                    trial = await rehearse(
                        browser, start_url, sb_c, report, auth,
                        dry_rehearsal=dry_rehearsal, extra_blocklist=extra_blocklist,
                    )
                    if trial.error is not None:
                        notes.append(f"chapter {chapter.title!r} failed rehearsal: {trial.error}")
                        continue
                    q = story_quality(trial.storyboard, trial.observations)
                    composed.append((chapter.title, polish(trial.storyboard)[0], q))

                composed.sort(key=lambda c: -c[2])
                for title, _sb_c, q in composed:
                    notes.append(f"chapter {title!r}: auto-composed story, score {q:.0f}")
                if composed and (best is None or best[1] < GOOD_STORY):
                    title, sb_c, q = composed.pop(0)
                    if best is None or q > best[1]:
                        best = (sb_c, q, best[2] if best else 0)
                        main_title = title
                # the page's other strong chapters become their own scenarios ("Coaching > Situations de match")
                extras = [c for c in composed if c[2] >= GOOD_STORY * 0.8][: max(chapters_per_page - 1, 0)]
            if best is None:
                raise ai_error or PlanError("no story could be built for this feature")
            final_sb, score, attempts = best
            final_sb.mutates = detect_mutating_storyboard(final_sb)
            more = f" + {len(extras)} more chapter(s)" if extras else ""
            say(report, "rehearsal", "done", f"story ready · {len(final_sb.steps)} steps · score {score:.0f}{more}")
            res = PlanResult(final_sb, start_url, snapshot, notes, attempts, replies, ai_down=ai_down)
            res.chapter = main_title
            res.extras = [(title, sb_c) for title, sb_c, _q in extras]
            return res
        finally:
            await browser.close()


def _short_validation(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
    return str(exc)



async def plan_storyboards(
    url: str,
    diff: DiffSummary,
    cfg: LLMConfig,
    *,
    scenarios: int = 1,
    focus: str | None = None,
    max_attempts: int = 4,
    headed: bool = False,
    report: Reporter = silent,
    auth: Path | None = None,
    picks: list[ScoutPick] | None = None,
    refine: bool = True,
    director: str = "auto",
    chapters_per_page: int = 3,
    dry_rehearsal: bool = False,
    extra_blocklist: tuple[str, ...] | list[str] = (),
) -> list[tuple[ScoutPick, PlanResult]]:
    """Scout the branch, then plan one grounded storyboard per chosen feature (on that feature's own page).

    ``picks`` (e.g. chosen interactively) skips the scout.
    """
    if picks:
        say(report, "scout", "done", f"{len(picks)} scenario(s) selected")
    elif focus:
        picks = [match_focus(diff, focus)]
        where = f" on {picks[0].route}" if picks[0].route else ""
        say(report, "scout", "done", f"--focus {focus!r}{where}")
    else:
        picks = await scout_features(diff, cfg, max(1, scenarios), report)
    results: list[tuple[ScoutPick, PlanResult]] = []
    errors: list[str] = []
    for k, pick in enumerate(picks, 1):
        prefix = f"[{k}/{len(picks)}] " if len(picks) > 1 else ""

        def scoped(ev: ProgressEvent, prefix: str = prefix) -> None:
            report(ProgressEvent(ev.stage, ev.state, f"{prefix}{ev.detail}" if ev.detail else ev.detail,
                                 ev.completed, ev.total))  # fmt: skip

        try:
            res = await plan_storyboard(
                url, diff, cfg, max_attempts=max_attempts, headed=headed, report=scoped,
                auth=auth, feature=pick, refine=refine, director=director,
                chapters_per_page=chapters_per_page, dry_rehearsal=dry_rehearsal,
                extra_blocklist=extra_blocklist,
            )
            if res.ai_down and director == "auto" and not pick.user_requested:
                director = "composer"  # don't wait on a dead endpoint for every non-custom scenario
        except LoginRequiredError:
            raise  # same wall for every scenario
        except PlanError as exc:
            errors.append(f"{pick.title}: {exc}")
            continue
        main = pick.model_copy()
        main.mutates = res.storyboard.mutates
        if res.chapter and res.chapter != "main view":
            main.title = f"{pick.title} > {res.chapter}"
        results.append((main, res))
        for title, sb_extra in res.extras:  # deeper chapters of the same page, each its own video
            sub = pick.model_copy()
            sub.title = f"{pick.title} > {title}"
            sub.mutates = sb_extra.mutates
            results.append((sub, PlanResult(sb_extra, res.url, res.snapshot, [], res.attempts, [])))
    if not results:
        raise PlanError("no scenario could be planned: " + " | ".join(errors))
    for e in errors:
        results[0][1].notes.append(f"scenario skipped - {e}")
    return results


def plan_storyboards_sync(
    url: str,
    diff: DiffSummary,
    cfg: LLMConfig,
    *,
    scenarios: int = 1,
    focus: str | None = None,
    headed: bool = False,
    report: Reporter = silent,
    auth: Path | None = None,
    picks: list[ScoutPick] | None = None,
    refine: bool = True,
    director: str = "auto",
    chapters_per_page: int = 3,
    dry_rehearsal: bool = False,
    extra_blocklist: tuple[str, ...] | list[str] = (),
) -> list[tuple[ScoutPick, PlanResult]]:
    return asyncio.run(plan_storyboards(
        url, diff, cfg, scenarios=scenarios, focus=focus, headed=headed,
        report=report, auth=auth, picks=picks, refine=refine,
        director=director, chapters_per_page=chapters_per_page,
        dry_rehearsal=dry_rehearsal, extra_blocklist=extra_blocklist,
    ))


def plan_storyboard_sync(
    url: str,
    diff: DiffSummary,
    cfg: LLMConfig,
    *,
    max_attempts: int = 3,
    headed: bool = False,
    report: Reporter = silent,
    auth: Path | None = None,
) -> PlanResult:
    return asyncio.run(
        plan_storyboard(url, diff, cfg, max_attempts=max_attempts, headed=headed, report=report, auth=auth)
    )
