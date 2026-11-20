"""Automatic story composer: discover how a feature is used, then direct an escalating demo of it, without an LLM.

A language model can choose *which* feature to show, but small local models write weak stories ("click A, click B,
leave"). The composer gives every feature a real one by experiment:

1. EXPLORE - on the feature page, find groups of similar controls outside the app chrome (player chips, filters,
   options, segmented tabs...). For the most promising groups, click members one after another in a scratch
   browser and record what each click changed on screen (a table appeared, its row count changed, a chart...).
2. COMPOSE - turn the first group that drives a visible result into a story with step captions:
   establish -> use #1 (-> #2) -> show the result -> use #3 -> show it again -> use #4 -> money shot -> wide hold.
3. SCORE - `story_quality` grades any rehearsed storyboard (AI-written, workflow-driven, or composed) so the pipeline
   keeps the better film.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import Browser, Page
from playwright.async_api import Error as PlaywrightError

from pullreel.human_runner import VIEWPORT, find_locator, wait_for_content
from pullreel.journeys import DESTRUCTIVE_RE, is_destructive
from pullreel.models import ActionStep, Camera, Locator, Storyboard
from pullreel.storyboard_agent import ARIA_ROLES, _estimate_s, _screen_signature, left_feature

# Groups of look-alike controls (same parent, same tag, same classes once state classes are removed), outside nav,
# sidebars and headers. Each member is named by its longest visible text ("EM Elias" chip -> "Elias").
_GROUPS_JS = r"""
(opts) => {
  const minSize = (opts && opts.min) || 3, wholePage = !!(opts && opts.wholePage);
  const t = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const visible = (e) => {
    const r = e.getBoundingClientRect(), st = getComputedStyle(e);
    return r.width > 4 && r.height > 4 && st.visibility !== 'hidden' && st.display !== 'none'
      && r.right > 0 && r.left < innerWidth && (wholePage || (r.bottom > 0 && r.top < innerHeight));
  };
  const chrome = (e) => !!e.closest('nav,aside,header,footer,[role=navigation],[role=banner],[role=menubar]');
  const native = 'button,[role=button],[role=tab],[role=checkbox],[role=option],[role=switch],[role=radio],'
    + 'input[type=checkbox],input[type=radio],select,a[href]';
  const longestText = (root) => {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let best = '';
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      const s = t(n.nodeValue), p = n.parentElement;
      if (!s || !p || (p.checkVisibility && !p.checkVisibility({opacityProperty: true}))) continue;
      const r = p.getBoundingClientRect();
      if (r.width >= 1 && r.height >= 1 && s.length > best.length) best = s;
    }
    return best;
  };
  const all = [...document.body.getElementsByTagName('*')].slice(0, 8000);
  const clickable = all.filter((e) => visible(e) && !chrome(e)
    && (e.matches(native) || getComputedStyle(e).cursor === 'pointer'));
  const set = new Set(clickable);
  const outer = clickable.filter((e) => !set.has(e.parentElement));
  const stateClass = /(active|selected|checked|current|open|on|is-|has-)/i;
  const groups = new Map();
  for (const e of outer) {
    const cls = (x) => (typeof x.className === 'string' ? x.className.split(/\s+/).filter((c) => c && !stateClass.test(c)).sort().join('.') : '');
    const p = e.parentElement;
    const key = `${p ? p.tagName + '.' + cls(p) : ''}>${e.tagName}.${cls(e)}`;
    const role = e.getAttribute('role') || (e.tagName === 'BUTTON' ? 'button' : e.tagName === 'A' ? 'link' : '');
    const name = t(e.getAttribute('aria-label')) || longestText(e);
    if (!name || name.length > 40 || /^[\d\s.,:%+-]+$/.test(name)) continue;
    if (!groups.has(key)) groups.set(key, []);
    const r = e.getBoundingClientRect();
    groups.get(key).push({name, role, tab: role === 'tab', y: Math.round(r.y)});
  }
  return [...groups.values()]
    .map((g) => g.filter((m, i) => g.findIndex((x) => x.name === m.name) === i))
    .filter((g) => g.length >= minSize)
    .map((g) => g.slice(0, 12));
}
"""

_TABLE_RE = re.compile(r'^table "([^"]*)" - columns: (.*) \((\d+) rows\)$')
_NAMED_RE = re.compile(r'^(?:chart|heading) "([^"]+)"$')
_DESTRUCTIVE = DESTRUCTIVE_RE  # backward-compatible alias


@dataclass(slots=True)
class Member:
    name: str
    role: str = ""

    def locator(self) -> Locator:
        if self.role in ARIA_ROLES and self.role not in ("listitem", "row", "cell"):
            return Locator(strategy="role", value=self.role, name=self.name)
        return Locator(strategy="text", value=self.name)


@dataclass(slots=True)
class Probe:
    member: Member
    appeared: list[str] = field(default_factory=list)  # what this click made appear / change
    left_page: bool = False

    @property
    def effective(self) -> bool:
        return not self.left_page and any(x.startswith(("table", "chart", "heading")) for x in self.appeared)


@dataclass(slots=True)
class Exploration:
    group: list[Member]
    probes: list[Probe]

    @property
    def effective_count(self) -> int:
        return sum(p.effective for p in self.probes)


def area_label(appeared: list[str]) -> tuple[str, str] | None:
    """(locator text, kind) of the most useful result area in a diff: tables first, then charts, then headings."""
    for kind in ("table", "chart", "heading"):
        for item in appeared:
            if not item.startswith(kind):
                continue
            m = _TABLE_RE.match(item)
            if m:
                first = m.group(1).strip()
                cols = [c.strip() for c in m.group(2).split("|") if c.strip()]
                label = first or (cols[0] if cols else "")
                if label:
                    return label, "table"
            m = _NAMED_RE.match(item)
            if m:
                return m.group(1), kind
    return None


def money_column(appeared: list[str]) -> str | None:
    """A column worth ending on: a signed / rating-like header ("Écart", "+/-", "Diff", "Net", "Score"...)."""
    for item in appeared:
        m = _TABLE_RE.match(item)
        if not m:
            continue
        cols = [c.strip() for c in m.group(2).split("|") if c.strip()]
        for c in cols:
            if re.search(r"(écart|ecart|\+/-|diff|net|score|rating|total|gain|delta|%|ratio)", c, re.I):
                return c
    return None


def _rank_groups(
    groups: list[list[dict]], min_size: int = 3, extra_blocklist: Sequence[str] = ()
) -> list[list[Member]]:
    """Selection chips / options before tab bars; bigger groups first; never destructive controls."""
    ranked: list[tuple[float, list[Member]]] = []
    for g in groups:
        members = [
            Member(m["name"], m.get("role", ""))
            for m in g
            if not is_destructive(m["name"], extra_blocklist)
        ]
        if len(members) < min_size:
            continue
        tabs = all(m.get("tab") for m in g)
        score = len(members) - (6 if tabs else 0)
        ranked.append((score, members))
    ranked.sort(key=lambda x: -x[0])
    return [m for _s, m in ranked]


async def _probe_group(page: Page, members: list[Member], limit: int = 4) -> list[Probe]:
    probes: list[Probe] = []
    start_path = page.url
    for member in members[:limit]:
        before = await _screen_signature(page)
        try:
            target = await find_locator(page, member.locator())
            await target.click(timeout=2500)
        except PlaywrightError:
            break
        await page.wait_for_timeout(450)
        await wait_for_content(page, timeout_s=3.0, quiet_s=0.5)
        after = await _screen_signature(page)
        appeared = sorted(after - before, key=lambda x: (not x.startswith(("table", "chart", "heading")), x))
        left = left_feature(urlparse(start_path).path, urlparse(page.url).path)
        probes.append(Probe(member, appeared[:12], left))
        if left:
            break
    return probes


async def explore_feature(
    browser: Browser,
    url: str,
    *,
    auth: Path | None = None,
    max_groups: int = 3,
    extra_blocklist: Sequence[str] = (),
) -> Exploration | None:
    """Find the control group that drives a visible result on this page (or None)."""
    context = await browser.new_context(
        viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, color_scheme="dark",
        storage_state=str(auth) if auth else None,
    )  # fmt: skip
    try:
        page = await context.new_page()
        await page.goto(url, wait_until="load")
        await wait_for_content(page)
        groups: list[list[dict]] = []
        with contextlib.suppress(PlaywrightError):
            groups = await page.evaluate(_GROUPS_JS, {"min": 3, "wholePage": False})
        best: Exploration | None = None
        for members in _rank_groups(groups, extra_blocklist=extra_blocklist)[:max_groups]:
            await page.goto(url, wait_until="load")  # clean state for every experiment
            await wait_for_content(page)
            probes = await _probe_group(page, members)
            found = Exploration(members, probes)
            if best is None or found.effective_count > best.effective_count:
                best = found
            if found.effective_count >= 3:
                break
        return best if best is not None and best.effective_count >= 1 else None
    finally:
        await context.close()


def compose_story(exploration: Exploration, summary: str, start_path: str) -> Storyboard:
    """Establish -> use the feature with growing richness, each change shown -> money shot -> wide hold."""
    steps = [ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1000)]
    steps += escalation_steps(exploration)
    steps.append(ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1500))
    return Storyboard(feature_summary=summary, start_path=start_path, steps=steps)


def escalation_steps(exploration: Exploration) -> list[ActionStep]:
    """Use #1 (-> #2) -> show -> use #3 -> show ... -> money shot, from a probed control group."""
    steps: list[ActionStep] = []
    usable = [p for p in exploration.probes if not p.left_page]
    while usable and not usable[0].appeared:  # leading no-ops: the active tab / already-selected item
        usable.pop(0)
    cut, effective = len(usable), 0
    for i, probe in enumerate(usable):  # a demo, not an inventory: stop after MAX_BEATS visible results
        effective += probe.effective
        if effective >= MAX_BEATS:
            cut = i + 1
            break
    usable = usable[:cut]
    last_effective = next((p for p in reversed(usable) if p.effective), None)
    last_area: tuple[str, str] | None = None
    shown = 0
    for i, probe in enumerate(usable):
        steps.append(ActionStep(
            action="click",
            locator=probe.member.locator(),
            camera=Camera(mode="focus", zoom=1.7),
            dwell_after_ms=500,
            caption=f"Select {probe.member.name}",
        ))
        area = area_label(probe.appeared) if probe.effective else None
        if area is None:
            continue  # e.g. one player alone: keep building the selection, show once something appears
        last_area = area
        final = i == len(usable) - 1
        steps.append(ActionStep(
            action="show",
            locator=Locator(strategy="text", value=area[0]),
            camera=Camera(mode="focus", zoom=1.35),
            dwell_after_ms=1500 if not final else 1300,
            caption=f"Live {area[1]} updates: {area[0]}",
        ))
        shown += 1
    if last_area is not None:
        column = money_column(last_effective.appeared) if last_effective else None
        if column and column != last_area[0]:
            steps.append(ActionStep(
                action="show",
                locator=Locator(strategy="text", value=column),
                camera=Camera(mode="focus", zoom=1.55),
                dwell_after_ms=1900,
                caption=f"Key result highlighted: {column}",
            ))
        else:  # the last payoff becomes the money shot: tighter and longer
            last = steps[-1]
            last.camera, last.dwell_after_ms = Camera(mode="focus", zoom=1.5), 1900
    return steps


# --------------------------------------------------------------------------------------------- deep exploration
@dataclass(slots=True)
class Chapter:
    """A sub-feature of a page: what a section control (tab, segment...) opens, plus the best escalation inside it."""

    title: str
    path: list[Member]  # clicks from page load that open the chapter ([] = the page's default view)
    revealed: list[str] = field(default_factory=list)  # result areas the opening click made appear
    inner: Exploration | None = None  # control group probed inside the chapter

    @property
    def depth_score(self) -> float:
        areas = sum(1 for x in self.revealed if x.startswith(("table", "chart", "heading")))
        inner = self.inner.effective_count if self.inner else 0
        return min(areas, 6) * 0.5 + inner * 2.0 + (1.0 if self.path else 0.0)


def compose_chapter(chapter: Chapter, summary: str, start_path: str) -> Storyboard:
    """Establish -> open the chapter -> show what it reveals -> escalate inside it -> money shot -> wide hold."""
    steps = [ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1000)]
    for member in chapter.path:
        steps.append(ActionStep(
            action="click",
            locator=member.locator(),
            camera=Camera(mode="focus", zoom=1.6),
            dwell_after_ms=700,
            caption=f"Open {member.name}",
        ))
    opened = area_label(chapter.revealed) if chapter.path else None
    if opened is not None:
        steps.append(ActionStep(
            action="show",
            locator=Locator(strategy="text", value=opened[0]),
            camera=Camera(mode="focus", zoom=1.3),
            dwell_after_ms=1500,
            caption=f"Inspect {opened[0]}",
        ))
    if chapter.inner is not None and chapter.inner.effective_count:
        steps += escalation_steps(chapter.inner)
    elif opened is not None:  # nothing to play with inside: the reveal itself is the money shot
        steps[-1].camera, steps[-1].dwell_after_ms = Camera(mode="focus", zoom=1.4), 2000
    steps.append(ActionStep(action="wait", camera=Camera(mode="wide"), dwell_after_ms=1500))
    return Storyboard(feature_summary=summary, start_path=start_path, steps=steps)


async def _groups(page: Page, min_size: int, extra_blocklist: Sequence[str] = ()) -> list[list[Member]]:
    try:
        raw = await page.evaluate(_GROUPS_JS, {"min": min_size, "wholePage": True})
    except PlaywrightError:
        return []
    return _rank_groups(raw, min_size=min_size, extra_blocklist=extra_blocklist)


def _key(group: list[Member]) -> tuple[str, ...]:
    return tuple(m.name for m in group)


def _fresh(group: list[Member], seen: set[str]) -> bool:
    """At least half of the group's controls were not on screen before (else it is the same bar re-detected)."""
    names = [m.name for m in group]
    return sum(n not in seen for n in names) * 2 > len(names)


async def _open(page: Page, url: str, path: list[Member]) -> bool:
    """Fresh load, then replay the clicks that lead to a chapter."""
    await page.goto(url, wait_until="load")
    await wait_for_content(page)
    for member in path:
        try:
            await (await find_locator(page, member.locator())).click(timeout=2500)
        except PlaywrightError:
            return False
        await page.wait_for_timeout(400)
        await wait_for_content(page, timeout_s=3.0, quiet_s=0.5)
    return True


def _is_selection(probes: list[Probe]) -> bool:
    """Chips / filters keep updating the SAME result area; sections each open a different one."""
    labels = [area_label(p.appeared) for p in probes if p.effective]
    labels = [lab[0] for lab in labels if lab]
    return len(labels) >= 2 and len(set(labels)) <= max(1, len(labels) // 2)


async def explore_page(
    browser: Browser,
    url: str,
    *,
    auth: Path | None = None,
    max_chapters: int = 5,
    time_budget_s: float = 150.0,
    extra_blocklist: Sequence[str] = (),
) -> list[Chapter]:
    """Dive into a page: its default view, every section it can open, and the controls inside each section."""
    import time as _time

    deadline = _time.monotonic() + time_budget_s
    context = await browser.new_context(
        viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, color_scheme="dark",
        storage_state=str(auth) if auth else None,
    )  # fmt: skip
    start_path = urlparse(url).path
    chapters: list[Chapter] = []
    try:
        page = await context.new_page()
        await _open(page, url, [])
        top = await _groups(page, 3, extra_blocklist)
        top_keys = {_key(g) for g in top}
        default_inner: Exploration | None = None
        sections: list[tuple[Member, list[str], list[list[Member]]]] = []
        for group in top[:4]:
            if _time.monotonic() > deadline:
                break
            await _open(page, url, [])
            probes: list[Probe] = []
            opened: list[tuple[Member, list[str], list[list[Member]]]] = []
            for member in group[:7]:
                before_groups = await _groups(page, 2, extra_blocklist)
                before_keys = {_key(g) for g in before_groups}
                seen_names = {m.name for g in [*before_groups, *top] for m in g}
                before = await _screen_signature(page)
                try:
                    await (await find_locator(page, member.locator())).click(timeout=2500)
                except PlaywrightError:
                    continue
                await page.wait_for_timeout(450)
                await wait_for_content(page, timeout_s=3.0, quiet_s=0.5)
                after = await _screen_signature(page)
                appeared = sorted(after - before, key=lambda x: (not x.startswith(("table", "chart", "heading")), x))
                left = left_feature(start_path, urlparse(page.url).path)
                probes.append(Probe(member, appeared[:14], left))
                if left:
                    break
                new_groups = [g for g in await _groups(page, 2, extra_blocklist)
                              if _key(g) not in before_keys | top_keys and _fresh(g, seen_names)
                              and all(m.name != member.name for m in g)]  # the clicked bar itself is not new
                if appeared or new_groups:
                    opened.append((member, appeared[:14], new_groups))
            if _is_selection(probes):
                found = Exploration(group, probes)
                if default_inner is None or found.effective_count > default_inner.effective_count:
                    default_inner = found
            else:
                sections += opened
        if default_inner is not None and default_inner.effective_count:
            chapters.append(Chapter("main view", [], [], default_inner))
        # second level: inside each section, probe the controls it revealed (toggles, filters, rows...)
        for member, revealed, new_groups in sections:
            if len(chapters) >= max_chapters + 1 or _time.monotonic() > deadline:
                break
            chapter = Chapter(member.name, [member], revealed)
            for inner_group in new_groups[:2]:
                if _time.monotonic() > deadline or not await _open(page, url, [member]):
                    break
                probes = await _probe_group(page, inner_group)
                found = Exploration(inner_group, probes)
                if found.effective_count and (chapter.inner is None
                                              or found.effective_count > chapter.inner.effective_count):  # fmt: skip
                    chapter.inner = found
            if chapter.inner is not None or sum(x.startswith(("table", "chart")) for x in revealed) >= 1:
                chapters.append(chapter)
        # sections with nothing to play inside still tell a story together: open each, show what it reveals
        flat = [(m, rev) for m, rev, _g in sections
                if not any(c.path and c.path[0].name == m.name and c.inner for c in chapters)
                and area_label(rev) is not None]  # fmt: skip
        if len(flat) >= 2:
            flat = flat[:MAX_BEATS]
            tour = Exploration([m for m, _r in flat], [Probe(m, rev) for m, rev in flat])
            chapters.append(Chapter("sections tour", [], [], tour))
            chapters = [c for c in chapters if c.inner is not None or not c.path]  # the tour replaces flat ones
    finally:
        await context.close()
    chapters.sort(key=lambda c: -c.depth_score)
    return chapters[:max_chapters]


def story_quality(sb: Storyboard, observations: list[str]) -> float:
    """Higher is a better demo: actions that change something, shown payoffs, escalation, length, focus."""
    effective = sum(
        1 for o in observations
        if ("appeared:" in o or "dry-rehearsal verified" in o or "navigated:" in o)
        and "LEFT THE FEATURE PAGE" not in o
    )
    shows = sum(1 for s in sb.steps if s.action == "show")
    fills = sum(1 for s in sb.steps if s.action == "fill")
    left = (not sb.journey) and any("LEFT THE FEATURE PAGE" in o for o in observations)
    seconds = sum(_estimate_s(s) for s in sb.steps)
    score = (
        2.0 * min(effective, 4)
        + 1.5 * min(shows, 4)
        + 0.75 * min(fills, 3)
        + (2.0 if seconds >= 10 else seconds / 5)
    )
    return round(score - (6.0 if left else 0.0), 2)


MAX_BEATS = 4  # visible results per escalation: enough to prove depth, short enough to watch
GOOD_STORY = 10.0  # e.g. 3 effective actions + 2 payoff shots + >= 10 s
