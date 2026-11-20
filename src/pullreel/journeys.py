"""Concrete user-journey detection, realistic form filling, mutation safety, and scenario sales ranking.

Moves PullReel beyond generic click tours:
* Detects real app workflows (create entity, edit, search/filter, compare, export/import, settings) and assigns
  concrete business titles ("Add a new player to the roster", not "Click Add").
* Inspects forms/modals and fills them with believable, domain-coherent data (always ``@example.com`` for emails,
  never passwords/cards/secrets), focusing the camera on each field and ending on the money shot (new item visible).
* Enforces an extended destructive-action blocklist and flags mutating storyboards for terminal warnings and
  ``--dry-rehearsal`` support.
* Ranks scenarios by sales value (visual payoff, interactive depth, diff novelty, workflow richness).
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from playwright.async_api import Browser, Page
from playwright.async_api import Error as PlaywrightError

from pullreel.human_runner import VIEWPORT, find_locator, wait_for_content
from pullreel.models import ActionStep, Camera, DiffSummary, FeatureCandidate, Locator, Storyboard

WorkflowKind = Literal["create", "edit", "search", "compare", "export", "import", "settings"]

# Extended destructive / auth / irreversible action blocklist (English + French).
DESTRUCTIVE_RE = re.compile(
    r"(supprim|delete|remove|destroy|\bdrop\b|purge|wipe|erase|clear\s+all|retirer|"
    r"reset|r[eé]initial|logout|log\s*out|sign\s*out|d[eé]connex|disconnect|"
    r"revoke|cancel\s+sub|close\s+account|deactivate|d[eé]sactiv|r[eé]silier|"
    r"vider|effacer|archive\s+all|unlink|ban\s+user|terminate)",
    re.I,
)

# Fields PullReel must NEVER fill (credentials, payment instruments, secrets).
SENSITIVE_FIELD_RE = re.compile(
    r"(password|passwd|\bpwd\b|mot\s*de\s*passe|passphrase|secret|token|api[_\s-]?key|"
    r"credit\s*card|card\s*number|\bcvv\b|\bcvc\b|\biban\b|\bswift\b|\bssn\b|"
    r"social\s*security|num[eé]ro\s*de\s*carte|\bpin\b|\botp\b|2fa|auth\s*code)",
    re.I,
)

_CREATE_RE = re.compile(
    r"^\s*\+?\s*(add|new|create|register|invite|compose|schedule|ajouter|nouveau|nouvelle|cr[eé]er|inviter)\b",
    re.I,
)
_EDIT_RE = re.compile(r"^\s*(edit|modify|rename|update|modifier|renommer)\b", re.I)
_SEARCH_RE = re.compile(r"(search|filter|find|lookup|recherch|filtr)", re.I)
_COMPARE_RE = re.compile(r"(compare|comparer|versus|\bvs\b)", re.I)
_EXPORT_RE = re.compile(r"(export|download|csv|pdf|report|exporter|t[eé]l[eé]charger)", re.I)
_IMPORT_RE = re.compile(r"(import|upload|importer|t[eé]l[eé]verser)", re.I)
_SETTINGS_RE = re.compile(r"(settings|preferences|param[eè]tres|r[eé]glages|config)", re.I)
_SUBMIT_RE = re.compile(
    r"(save|create|add|submit|confirm|apply|publish|send|invite|update|register|post|"
    r"enregistrer|cr[eé]er|ajouter|valider|confirmer|sauvegarder|envoyer|publier)",
    re.I,
)


def is_destructive(text: str, extra_blocklist: Sequence[str] = ()) -> bool:
    if not text:
        return False
    if DESTRUCTIVE_RE.search(text):
        return True
    low = text.lower()
    return any(term.lower() in low for term in extra_blocklist if term.strip())


def is_sensitive_field(label: str = "", name: str = "", placeholder: str = "", input_type: str = "") -> bool:
    if input_type.lower() == "password":
        return True
    blob = " ".join(x for x in (label, name, placeholder) if x)
    return bool(SENSITIVE_FIELD_RE.search(blob))


@dataclass(slots=True)
class FormField:
    tag: str
    input_type: str = "text"
    name: str = ""
    label: str = ""
    placeholder: str = ""
    role: str = ""
    testid: str = ""
    options: list[str] = field(default_factory=list)
    min_val: float | None = None
    max_val: float | None = None
    required: bool = False
    value: str = ""

    def locator(self) -> Locator:
        if self.testid:
            return Locator(strategy="testid", value=self.testid)
        if self.label:
            return Locator(strategy="label", value=self.label)
        if self.placeholder:
            return Locator(strategy="placeholder", value=self.placeholder)
        if self.role and self.name:
            return Locator(strategy="role", value=self.role, name=self.name)
        if self.name:
            return Locator(strategy="css", value=f'{self.tag}[name="{self.name}"]')
        return Locator(strategy="css", value=self.tag)

    def display_label(self) -> str:
        return self.label or self.placeholder or self.name or self.input_type or "field"


@dataclass(slots=True)
class DetectedWorkflow:
    kind: WorkflowKind
    title: str
    trigger: Locator
    trigger_text: str
    mutates: bool
    inline_form: bool = False
    context_heading: str = ""


@dataclass(slots=True)
class JourneyPlan:
    workflow: DetectedWorkflow
    storyboard: Storyboard
    appeared: list[str] = field(default_factory=list)
    sales_score: float = 0.0
    pages_visited: int = 1


_WORKFLOWS_JS = r"""
() => {
  const t = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const vis = (e) => {
    const r = e.getBoundingClientRect(), st = getComputedStyle(e);
    return r.width > 4 && r.height > 4 && st.visibility !== 'hidden' && st.display !== 'none'
      && r.right > 0 && r.left < innerWidth && r.bottom > 0 && r.top < innerHeight * 2.2;
  };
  const chrome = (e) => !!e.closest('nav,aside,footer,[role=navigation],[role=menubar]');
  const h1 = t(document.querySelector('h1,h2,[role=heading]')?.innerText || document.title || '');
  const labelOf = (el) => {
    if (el.labels && el.labels.length) return t(el.labels[0].innerText);
    const lb = el.getAttribute('aria-labelledby');
    if (lb) return t(lb.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' '));
    return t(el.getAttribute('aria-label'));
  };
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll('button,a[href],[role=button],input[type=search],input[type=text],input:not([type])')) {
    if (seen.has(el) || !vis(el) || chrome(el)) continue;
    seen.add(el);
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') || (tag === 'button' ? 'button' : tag === 'a' ? 'link' : 'textbox');
    const label = labelOf(el);
    const placeholder = t(el.getAttribute('placeholder'));
    const text = tag === 'input' ? (label || placeholder) : (t(el.innerText) || label || t(el.getAttribute('title')));
    const testid = el.getAttribute('data-testid') || el.getAttribute('data-test') || '';
    if (!text || text.length > 60) continue;
    const inForm = !!el.closest('form');
    out.push({
      tag, role, text, label, placeholder, testid,
      input_type: tag === 'input' ? (el.type || 'text').toLowerCase() : '',
      in_form: inForm,
      heading: h1,
    });
  }
  const hasVisibleForm = [...document.querySelectorAll('form,[role=dialog],dialog')].some((f) => vis(f));
  const sampleTexts = [...document.querySelectorAll('td,li,[role=listitem],[role=cell],.card h3,.task-title')]
    .filter((e) => vis(e) && !chrome(e))
    .map((e) => t(e.innerText))
    .filter((s) => s.length >= 3 && s.length <= 40 && /[a-zA-Z]/.test(s))
    .slice(0, 15);
  return {heading: h1, controls: out.slice(0, 80), has_visible_form: hasVisibleForm, samples: [...new Set(sampleTexts)]};
}
"""

_FORM_FIELDS_JS = r"""
() => {
  const t = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const vis = (e) => {
    const r = e.getBoundingClientRect(), st = getComputedStyle(e);
    return r.width > 4 && r.height > 4 && st.visibility !== 'hidden' && st.display !== 'none'
      && r.bottom > 0 && r.top < innerHeight * 2.5;
  };
  const labelOf = (el) => {
    if (el.labels && el.labels.length) return t(el.labels[0].innerText);
    const lb = el.getAttribute('aria-labelledby');
    if (lb) return t(lb.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' '));
    const wrap = el.closest('label');
    if (wrap) return t(wrap.innerText);
    return t(el.getAttribute('aria-label'));
  };
  // Prefer an open dialog/modal first, then the most prominent visible form.
  const dialog = [...document.querySelectorAll('dialog[open],[role=dialog],[aria-modal=true],.modal.open,.modal.is-open,.modal')].find(vis);
  const forms = [...document.querySelectorAll('form')].filter(vis);
  const root = dialog || forms[forms.length - 1] || document.body;
  const fields = [];
  for (const el of root.querySelectorAll('input,textarea,select')) {
    if (!vis(el) || el.disabled || el.readOnly) continue;
    const tag = el.tagName.toLowerCase();
    const ty = tag === 'input' ? (el.type || 'text').toLowerCase() : tag;
    if (['hidden', 'submit', 'button', 'reset', 'file', 'image'].includes(ty)) continue;
    const opts = tag === 'select'
      ? [...el.options].filter((o) => !o.disabled && t(o.text)).map((o) => t(o.text)).slice(0, 12)
      : [];
    const role = el.getAttribute('role') || (tag === 'select' ? 'combobox' : ty === 'checkbox' ? 'checkbox' : ty === 'radio' ? 'radio' : 'textbox');
    fields.push({
      tag,
      input_type: ty,
      name: el.getAttribute('name') || el.id || '',
      label: labelOf(el),
      placeholder: t(el.getAttribute('placeholder')),
      role,
      testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || '',
      options: opts,
      min_val: el.min !== '' && !isNaN(Number(el.min)) ? Number(el.min) : null,
      max_val: el.max !== '' && !isNaN(Number(el.max)) ? Number(el.max) : null,
      required: !!el.required,
      value: tag === 'select' ? '' : (el.value || ''),
    });
  }
  const submits = [];
  for (const b of root.querySelectorAll('button,input[type=submit],[role=button]')) {
    if (!vis(b) || b.disabled) continue;
    const text = t(b.innerText || b.value || b.getAttribute('aria-label'));
    if (!text || text.length > 50) continue;
    const isSubmit = (b.type || '').toLowerCase() === 'submit';
    submits.push({
      text,
      is_submit: isSubmit,
      testid: b.getAttribute('data-testid') || '',
      role: b.getAttribute('role') || 'button',
    });
  }
  return {in_dialog: !!dialog, fields: fields.slice(0, 12), submits: submits.slice(0, 12)};
}
"""


def _business_title(kind: WorkflowKind, trigger_text: str, heading: str) -> str:
    """Turn a raw UI control label into a clean, business-oriented scenario title."""
    clean = re.sub(r"^\s*\+\s*", "", trigger_text).strip()
    ctx = heading.strip()
    if kind == "create":
        m = re.match(r"^(?:add|new|create|ajouter|nouveau|nouvelle|cr[eé]er)\s+(?:a\s+|an\s+|un\s+|une\s+)?(.+)$", clean, re.I)
        entity = m.group(1).strip() if m else ""
        if entity and len(entity) >= 3:
            suffix = f" in {ctx}" if ctx and entity.lower() not in ctx.lower() and len(ctx) <= 28 else ""
            return f"Add a new {entity}{suffix}"
        return f"Create a new item in {ctx}" if ctx else "Create a new entry"
    if kind == "edit":
        return f"Edit an entry in {ctx}" if ctx else f"{clean} and save changes"
    if kind == "search":
        return f"Search and filter {ctx}" if ctx else "Search and filter results in real time"
    if kind == "compare":
        return f"Compare items in {ctx}" if ctx else "Compare entries side by side"
    if kind == "export":
        return f"Export {ctx} report" if ctx else f"Export data ({clean})"
    if kind == "import":
        return f"Import data into {ctx}" if ctx else "Import records into the workspace"
    if kind == "settings":
        return f"Update {ctx} preferences" if ctx else "Configure workspace settings"
    return clean


def _control_locator(ctrl: dict[str, object]) -> Locator:
    testid = str(ctrl.get("testid") or "")
    label = str(ctrl.get("label") or "")
    placeholder = str(ctrl.get("placeholder") or "")
    role = str(ctrl.get("role") or "")
    text = str(ctrl.get("text") or "")
    tag = str(ctrl.get("tag") or "")
    if testid:
        return Locator(strategy="testid", value=testid)
    if tag == "input":
        if label:
            return Locator(strategy="label", value=label)
        if placeholder:
            return Locator(strategy="placeholder", value=placeholder)
    if role in ("button", "link", "tab") and text:
        return Locator(strategy="role", value=role, name=text)
    return Locator(strategy="text", value=text or label or placeholder)


def classify_workflows(raw_scan: dict[str, object], extra_blocklist: Sequence[str] = ()) -> list[DetectedWorkflow]:
    """Pure classifier over ``_WORKFLOWS_JS`` output so it is easy to test deterministically."""
    heading = str(raw_scan.get("heading") or "")
    controls = raw_scan.get("controls") or []
    found: list[DetectedWorkflow] = []
    seen_keys: set[tuple[str, str]] = set()

    for ctrl in controls:  # type: ignore[union-attr]
        text = str(ctrl.get("text") or "")
        if not text or is_destructive(text, extra_blocklist):
            continue
        tag = str(ctrl.get("tag") or "")
        in_type = str(ctrl.get("input_type") or "")
        in_form = bool(ctrl.get("in_form"))

        kind: WorkflowKind | None = None
        mutates = False
        if tag == "input" and (in_type == "search" or _SEARCH_RE.search(text)):
            kind, mutates = "search", False
        elif tag != "input":
            if _CREATE_RE.search(text):
                kind, mutates = "create", True
            elif _EDIT_RE.search(text):
                kind, mutates = "edit", True
            elif _COMPARE_RE.search(text):
                kind, mutates = "compare", False
            elif _EXPORT_RE.search(text):
                kind, mutates = "export", False
            elif _IMPORT_RE.search(text):
                kind, mutates = "import", True
            elif _SETTINGS_RE.search(text) or _SETTINGS_RE.search(heading):
                if _SUBMIT_RE.search(text):
                    kind, mutates = "settings", True
        if kind is None:
            continue
        key = (kind, text.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        found.append(
            DetectedWorkflow(
                kind=kind,
                title=_business_title(kind, text, heading),
                trigger=_control_locator(ctrl),
                trigger_text=text,
                mutates=mutates,
                inline_form=in_form,
                context_heading=heading,
            )
        )

    priority = {"create": 0, "search": 1, "compare": 2, "edit": 3, "export": 4, "settings": 5, "import": 6}
    found.sort(key=lambda w: priority.get(w.kind, 9))
    return found


def generate_field_value(
    field: FormField,
    *,
    context_hint: str = "",
    sample_texts: Sequence[str] = (),
) -> str | None:
    """Generate believable, domain-coherent, non-sensitive form data from a field's metadata.

    Returns ``None`` for password/card/secret fields so PullReel never types sensitive data.
    Emails ALWAYS use ``@example.com``.
    """
    if is_sensitive_field(field.label, field.name, field.placeholder, field.input_type):
        return None

    ty = field.input_type.lower()
    hint = f"{field.label} {field.name} {field.placeholder} {context_hint}".lower()

    if field.tag == "select":
        for opt in field.options:
            if not re.match(r"^(select|choose|pick|choisir|s[eé]lectionner|--|\s*$)", opt, re.I):
                return opt
        return field.options[0] if field.options else None

    if ty in ("checkbox", "radio"):
        return "toggle"

    if ty == "email" or "email" in hint or "e-mail" in hint or "courriel" in hint:
        return "alex.moreau@example.com"
    if ty == "url" or "website" in hint or "url" in hint or "site web" in hint:
        return "https://example.com/demo"
    if ty == "tel" or "phone" in hint or "tél" in hint or "tel" in hint:
        return "+1 555 014 2890"
    if ty == "date" or "date" in hint or "deadline" in hint or "due" in hint:
        return "2026-11-24"
    if ty == "time":
        return "14:30"
    if ty in ("number", "range"):
        lo = field.min_val if field.min_val is not None else 1.0
        hi = field.max_val if field.max_val is not None else 999.0
        if any(k in hint for k in ("jersey", "maillot", "number", "numéro", "#")):
            val = 11.0
        elif any(k in hint for k in ("age", "âge")):
            val = 24.0
        elif any(k in hint for k in ("height", "taille", "cm")):
            val = 188.0
        elif any(k in hint for k in ("price", "amount", "cost", "budget", "tarif", "prix")):
            val = 149.0
        elif any(k in hint for k in ("qty", "quantity", "count", "quantit")):
            val = 3.0
        elif any(k in hint for k in ("rating", "score", "percent", "%")):
            val = 88.0
        else:
            val = 12.0
        clamped = min(hi, max(lo, val))
        return str(int(clamped) if clamped == int(clamped) else round(clamped, 1))

    if ty == "search" or _SEARCH_RE.search(f"{field.label} {field.placeholder} {field.name}"):
        for s in sample_texts:
            word = next((w for w in re.findall(r"[A-Za-zÀ-ÿ0-9]{3,16}", s)), "")
            if word:
                return word
        return "demo"

    field_hint = f"{field.label} {field.name} {field.placeholder}".lower()
    if field.tag == "textarea" or any(k in field_hint for k in ("description", "notes", "bio", "comment", "summary", "message")):
        return "Ready for review in the next sprint."
    if any(k in field_hint for k in ("first name", "prénom", "prenom", "firstname")):
        return "Alex"
    if any(k in field_hint for k in ("last name", "nom de famille", "lastname", "surname")):
        return "Moreau"
    if any(k in field_hint for k in ("player", "joueur", "member", "author", "user")):
        return "Alex Moreau"
    if any(k in field_hint for k in ("team", "équipe", "equipe", "club")):
        return "Lyon Falcons"
    if any(k in field_hint for k in ("project", "projet", "workspace", "company", "org")):
        return "Acme Studio"
    if any(k in field_hint for k in ("full name", "name", "nom")):
        if any(k in hint for k in ("team", "équipe", "equipe", "club")) and "player" not in hint:
            return "Lyon Falcons"
        if any(k in hint for k in ("project", "projet", "workspace", "company", "org")):
            return "Acme Studio"
        return "Alex Moreau"
    if any(k in hint for k in ("position", "poste", "role", "rôle")):
        return "Point Guard"
    if any(k in hint for k in ("city", "ville", "location", "lieu")):
        return "Lyon"
    if any(k in hint for k in ("tag", "label", "category", "catégorie")):
        return "Priority"
    if any(k in hint for k in ("title", "titre", "task", "tâche", "issue", "subject", "objet")):
        return "Ship Q4 analytics view"
    return "Alex Moreau"


def detect_mutating_storyboard(sb: Storyboard) -> bool:
    """True when the storyboard is flagged as mutating or submits a non-search form."""
    if sb.mutates:
        return True
    has_non_search_fill = False
    for step in sb.steps:
        if step.action == "fill" and step.locator is not None:
            desc = step.locator.describe()
            if not _SEARCH_RE.search(desc):
                has_non_search_fill = True
        if step.action == "click" and step.locator is not None:
            desc = step.locator.describe()
            if has_non_search_fill and _SUBMIT_RE.search(desc):
                return True
            if _CREATE_RE.search(step.locator.name or step.locator.value) and has_non_search_fill:
                return True
    return False


def score_candidate_sales(
    *,
    visual_payoffs: int = 0,
    effective_steps: int = 0,
    workflow_kind: WorkflowKind | None = None,
    candidate: FeatureCandidate | None = None,
    diff: DiffSummary | None = None,
    title: str = "",
) -> float:
    """Score a scenario's sales value on a 0..100 scale.

    Factors:
    * visual result richness (tables/charts/new items shown),
    * interactive depth (screen-changing steps),
    * diff novelty (new files, added UI lines, commit/label overlap),
    * workflow richness (end-to-end creation/search/compare journeys beat generic tours).
    """
    score = 20.0
    score += min(visual_payoffs, 4) * 9.0  # up to +36
    score += min(effective_steps, 5) * 5.0  # up to +25
    workflow_bonus = {
        "create": 14.0,
        "compare": 12.0,
        "search": 10.0,
        "edit": 9.0,
        "export": 8.0,
        "settings": 7.0,
        "import": 7.0,
    }
    if workflow_kind is not None:
        score += workflow_bonus.get(workflow_kind, 0.0)
    if candidate is not None:
        score += min(candidate.new_files * 5.0, 12.0)
        score += min(candidate.added / 40.0, 12.0)
        score += min(len(candidate.labels) * 1.2, 8.0)
    elif diff is not None and title:
        low = title.lower()
        for feat in diff.features:
            if feat.title.lower() in low or any(lbl.lower() in low for lbl in feat.labels):
                score += min(feat.new_files * 4.0 + feat.added / 50.0, 15.0)
                break
    return round(min(100.0, max(5.0, score)), 1)


async def detect_workflows(page: Page, extra_blocklist: Sequence[str] = ()) -> tuple[list[DetectedWorkflow], list[str]]:
    """Scan ``page`` for concrete user workflows and sample visible entity texts."""
    raw = await page.evaluate(_WORKFLOWS_JS)
    workflows = classify_workflows(raw, extra_blocklist=extra_blocklist)
    samples = [str(s) for s in (raw.get("samples") or [])]
    return workflows, samples


async def inspect_active_form(page: Page) -> tuple[bool, list[FormField], list[dict[str, object]]]:
    """Inspect the currently visible dialog or form on ``page``."""
    raw = await page.evaluate(_FORM_FIELDS_JS)
    fields = [FormField(**f) for f in (raw.get("fields") or [])]
    submits = list(raw.get("submits") or [])
    return bool(raw.get("in_dialog")), fields, submits


def _pick_submit_button(
    submits: list[dict[str, object]],
    extra_blocklist: Sequence[str] = (),
) -> tuple[Locator, str] | None:
    candidates = [s for s in submits if not is_destructive(str(s.get("text") or ""), extra_blocklist)]
    for s in candidates:
        if s.get("is_submit") and _SUBMIT_RE.search(str(s.get("text") or "")):
            return _submit_loc(s), str(s["text"])
    for s in candidates:
        if _SUBMIT_RE.search(str(s.get("text") or "")):
            return _submit_loc(s), str(s["text"])
    for s in candidates:
        if s.get("is_submit"):
            return _submit_loc(s), str(s["text"])
    return None


def _submit_loc(s: dict[str, object]) -> Locator:
    if s.get("testid"):
        return Locator(strategy="testid", value=str(s["testid"]))
    return Locator(strategy="role", value=str(s.get("role") or "button"), name=str(s["text"]))


async def explore_journeys(
    browser: Browser,
    url: str,
    *,
    auth: Path | None = None,
    dry_rehearsal: bool = False,
    diff: DiffSummary | None = None,
    extra_blocklist: Sequence[str] = (),
    max_journeys: int = 3,
) -> list[JourneyPlan]:
    """Discover and rehearse concrete end-to-end journeys (create forms, search/filter, settings) on ``url``."""
    from pullreel.storyboard_agent import _screen_signature

    start_path = urlparse(url).path or "/"
    plans: list[JourneyPlan] = []
    context = await browser.new_context(
        viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]},
        color_scheme="dark",
        storage_state=str(auth) if auth else None,
    )
    try:
        page = await context.new_page()
        await page.goto(url, wait_until="load")
        await wait_for_content(page)
        workflows, samples = await detect_workflows(page, extra_blocklist=extra_blocklist)

        for wf in workflows:
            if len(plans) >= max_journeys:
                break
            await page.goto(url, wait_until="load")
            await wait_for_content(page)

            if wf.kind == "search":
                query = generate_field_value(
                    FormField(tag="input", input_type="search", label=wf.trigger_text),
                    context_hint=wf.context_heading,
                    sample_texts=samples,
                )
                if not query:
                    continue
                before = await _screen_signature(page)
                try:
                    target = await find_locator(page, wf.trigger)
                    await target.click(timeout=2500)
                    await target.fill(query, timeout=2500)
                except PlaywrightError:
                    continue
                await page.wait_for_timeout(400)
                after = await _screen_signature(page)
                changed = sorted((after - before) | (before - after))
                if not changed and not samples:
                    continue
                show_text = next((s for s in samples if query.lower() in s.lower()), query)
                steps = [
                    ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=900),
                    ActionStep(
                        action="fill",
                        locator=wf.trigger,
                        input_value=query,
                        camera=Camera(mode="focus", zoom=1.75),
                        dwell_after_ms=700,
                        caption=f"Filter results by '{query}'",
                    ),
                    ActionStep(
                        action="show",
                        locator=Locator(strategy="text", value=show_text),
                        camera=Camera(mode="focus", zoom=1.4),
                        dwell_after_ms=1800,
                        caption="Matching results update immediately",
                    ),
                    ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1400),
                ]
                sb = Storyboard(feature_summary=wf.title, start_path=start_path, steps=steps, mutates=False)
                score = score_candidate_sales(
                    visual_payoffs=1, effective_steps=2, workflow_kind="search", diff=diff, title=wf.title
                )
                plans.append(JourneyPlan(workflow=wf, storyboard=sb, appeared=changed[:10], sales_score=score))

            elif wf.kind in ("create", "edit"):
                before_open = await _screen_signature(page)
                pages_seen = {urlparse(page.url).path or "/"}
                steps: list[ActionStep] = [ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=900)]
                opened_by_click = False
                if not wf.inline_form:
                    try:
                        trigger = await find_locator(page, wf.trigger)
                        await trigger.click(timeout=2500)
                        opened_by_click = True
                    except PlaywrightError:
                        continue
                    await page.wait_for_timeout(400)
                    await wait_for_content(page, timeout_s=3.0, quiet_s=0.4)
                    pages_seen.add(urlparse(page.url).path or "/")
                    steps.append(
                        ActionStep(
                            action="click",
                            locator=wf.trigger,
                            camera=Camera(mode="focus", zoom=1.65),
                            dwell_after_ms=650,
                            caption=f"Open {wf.trigger_text}",
                        )
                    )

                _in_dialog, fields, submits = await inspect_active_form(page)
                fillable = [f for f in fields if not is_sensitive_field(f.label, f.name, f.placeholder, f.input_type)]
                if not fillable:
                    continue

                primary_entered_value: str | None = None
                filled_count = 0
                for f in fillable[:5]:
                    val = generate_field_value(f, context_hint=wf.title, sample_texts=samples)
                    if val is None:
                        continue
                    floc = f.locator()
                    try:
                        el = await find_locator(page, floc)
                        await el.wait_for(state="visible", timeout=2000)
                        if f.tag == "select":
                            await el.select_option(label=val, timeout=2000)
                        elif f.input_type in ("checkbox", "radio"):
                            await el.click(timeout=2000)
                        else:
                            await el.click(timeout=2000)
                            await el.fill(val, timeout=2000)
                    except PlaywrightError:
                        continue
                    filled_count += 1
                    if primary_entered_value is None and f.tag != "select" and f.input_type not in ("checkbox", "radio"):
                        primary_entered_value = val
                    if f.input_type in ("checkbox", "radio"):
                        steps.append(
                            ActionStep(
                                action="click",
                                locator=floc,
                                camera=Camera(mode="focus", zoom=1.7),
                                dwell_after_ms=500,
                                caption=f"Select {f.display_label()}",
                            )
                        )
                    else:
                        steps.append(
                            ActionStep(
                                action="fill",
                                locator=floc,
                                input_value=val,
                                camera=Camera(mode="focus", zoom=1.75),
                                dwell_after_ms=500,
                                caption=f"Fill {f.display_label()}",
                            )
                        )

                if filled_count == 0:
                    continue

                submit_pick = _pick_submit_button(submits, extra_blocklist=extra_blocklist)
                if submit_pick is None:
                    continue
                submit_loc, submit_text = submit_pick
                try:
                    sub_el = await find_locator(page, submit_loc)
                    await sub_el.wait_for(state="visible", timeout=2000)
                except PlaywrightError:
                    continue

                steps.append(
                    ActionStep(
                        action="click",
                        locator=submit_loc,
                        camera=Camera(mode="focus", zoom=1.6),
                        dwell_after_ms=900,
                        caption=f"Submit ({submit_text})",
                    )
                )

                appeared: list[str] = []
                money_target: Locator | None = None
                if not dry_rehearsal:
                    try:
                        await sub_el.click(timeout=2500)
                    except PlaywrightError:
                        continue
                    await page.wait_for_timeout(500)
                    await wait_for_content(page, timeout_s=3.0, quiet_s=0.5)
                    pages_seen.add(urlparse(page.url).path or "/")
                    after_submit = await _screen_signature(page)
                    appeared = sorted(after_submit - before_open)
                    # Check if the newly created item is now visible on screen (the money shot!)
                    if primary_entered_value:
                        with contextlib.suppress(PlaywrightError):
                            cand = page.get_by_text(primary_entered_value, exact=False).first
                            if await cand.is_visible(timeout=1200):
                                money_target = Locator(strategy="text", value=primary_entered_value)
                elif primary_entered_value:
                    money_target = Locator(strategy="text", value=primary_entered_value)

                if money_target is not None:
                    steps.append(
                        ActionStep(
                            action="show",
                            locator=money_target,
                            camera=Camera(mode="focus", zoom=1.45),
                            dwell_after_ms=1900,
                            caption="New item appears immediately in the list",
                        )
                    )
                steps.append(ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1500))
                sb = Storyboard(
                    feature_summary=wf.title,
                    start_path=start_path,
                    steps=steps,
                    journey=len(pages_seen) > 1,
                    mutates=True,
                )
                score = score_candidate_sales(
                    visual_payoffs=2 if money_target else 1,
                    effective_steps=filled_count + (1 if opened_by_click else 0) + 1,
                    workflow_kind=wf.kind,
                    diff=diff,
                    title=wf.title,
                )
                plans.append(
                    JourneyPlan(
                        workflow=wf,
                        storyboard=sb,
                        appeared=appeared[:12],
                        sales_score=score,
                        pages_visited=len(pages_seen),
                    )
                )
    finally:
        await context.close()

    plans.sort(key=lambda p: -p.sales_score)
    return plans
