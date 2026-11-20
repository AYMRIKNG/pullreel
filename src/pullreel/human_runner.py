"""Drive Chromium like a human and capture a Retina-sharp recording with 60 Hz telemetry.

Secrets implemented here:

* Retina 2x capture: logical viewport 1280x800 with ``device_scale_factor=2`` -> 2560x1600 frames, so a 2x camera
  zoom still maps 1 source pixel to 1 output pixel.
* Frames come from the CDP screencast (JPEG, timestamped) instead of Playwright's blurry WebM recorder.
* The mouse never teleports: cubic Bezier paths with controlled curvature, timed with the minimum-jerk law
  ``s(t) = 10t^3 - 15t^4 + 6t^5``. The path is stored analytically so telemetry is perfectly smooth even when
  the OS timer is coarse (Windows sleeps in ~15 ms steps).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import Browser, CDPSession, Locator as PWLocator, Page, Playwright
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from pullreel.models import (
    ActionStep,
    Box,
    CameraMode,
    ClickEvent,
    Locator,
    Recording,
    SourceFrame,
    Storyboard,
    TelemetryFrame,
)
from pullreel.progress import Reporter, say, silent

VIEWPORT = (1280, 800)
SCALE = 2.0
SAMPLE_HZ = 60.0

ProgressFn = Callable[[str], None]


class StepError(RuntimeError):
    """A storyboard step could not be executed (element missing, not visible, ...)."""


def minimum_jerk(t: float) -> float:
    """Biomechanical minimum-jerk position profile on [0, 1]."""
    t = min(1.0, max(0.0, t))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


@dataclass(slots=True)
class _Motion:
    """A cubic Bezier mouse path, parametrised in wall-clock time with minimum-jerk easing."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    p2: tuple[float, float]
    p3: tuple[float, float]
    t0: float
    duration: float

    @property
    def t1(self) -> float:
        return self.t0 + self.duration

    def at(self, t: float) -> tuple[float, float]:
        s = minimum_jerk((t - self.t0) / self.duration)
        u = 1.0 - s
        b0, b1, b2, b3 = u * u * u, 3 * u * u * s, 3 * u * s * s, s * s * s
        x = b0 * self.p0[0] + b1 * self.p1[0] + b2 * self.p2[0] + b3 * self.p3[0]
        y = b0 * self.p0[1] + b1 * self.p1[1] + b2 * self.p2[1] + b3 * self.p3[1]
        return x, y


@dataclass(slots=True)
class _State:
    mouse_x: float
    mouse_y: float
    is_clicking: bool = False
    target_zoom: float = 1.0
    focus_box: Box | None = None
    camera_mode: CameraMode = "wide"
    motion: _Motion | None = None

    def mouse_at(self, t: float) -> tuple[float, float]:
        m = self.motion
        if m is not None and m.t0 <= t < m.t1:
            return m.at(t)
        return self.mouse_x, self.mouse_y


@dataclass(slots=True)
class _Capture:
    work_dir: Path
    frames: list[SourceFrame] = field(default_factory=list)
    acks: set[asyncio.Task[Any]] = field(default_factory=set)
    first_frame: asyncio.Event = field(default_factory=asyncio.Event)


def resolve_locator(page: Page, loc: Locator) -> PWLocator:
    """Map a semantic storyboard locator to a Playwright locator (first match)."""
    match loc.strategy:
        case "role":
            if loc.name:
                target = page.get_by_role(loc.value, name=loc.name, exact=False)  # type: ignore[arg-type]
            else:
                target = page.get_by_role(loc.value)  # type: ignore[arg-type]
        case "label":
            target = page.get_by_label(loc.value)
        case "placeholder":
            target = page.get_by_placeholder(loc.value)
        case "text":
            target = page.get_by_text(loc.value)
        case "testid":
            target = page.get_by_test_id(loc.value)
        case "css":
            target = page.locator(loc.value)
    return target.first


async def find_locator(page: Page, loc: Locator) -> PWLocator:
    """Like :func:`resolve_locator`, but a ``text`` locator prefers an *exact* text match ("Analyse" must not land
    on "Analyse défensive détaillée" just because it comes first in the DOM)."""
    if loc.strategy == "text":
        exact = page.get_by_text(loc.value, exact=True)
        with contextlib.suppress(PlaywrightError):
            if await exact.count() > 0:
                return exact.first
    return resolve_locator(page, loc)


# Climb from a label (column header, title...) to the block that holds the result, so `show` frames the whole table /
# chart / card rather than a 20-px header cell. Stops before giant layout wrappers.
_REGION_JS = """
(el) => {
  const big = (r) => r.width >= 360 && r.height >= 150;
  const huge = (r) => r.width > innerWidth * 0.92 && r.height > innerHeight * 0.92;
  const preferred = (e) => /^(TABLE|SECTION|ARTICLE|FIGURE|FORM|UL|OL)$/.test(e.tagName)
    || ['table', 'grid', 'region', 'figure', 'list', 'dialog'].includes(e.getAttribute('role') || '');
  let e = el, best = el;
  for (let i = 0; i < 8 && e && e !== document.body; i++, e = e.parentElement) {
    const r = e.getBoundingClientRect();
    if (huge(r)) break;
    best = e;
    if (big(r) && (preferred(e) || i >= 2)) break;
  }
  const r = best.getBoundingClientRect();
  return [r.x, r.y, r.width, r.height];
}
"""


async def result_region(target: PWLocator, fallback: Box) -> Box:
    """Bounding box (CSS px, viewport) of the result area around ``target``, clamped to the viewport."""
    try:
        x, y, w, h = await target.evaluate(_REGION_JS)
    except PlaywrightError:
        return fallback
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(VIEWPORT[0]), x + w), min(float(VIEWPORT[1]), y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return fallback
    return (x0, y0, x1 - x0, y1 - y0)


# [visible text length, element count]: the page is "ready" once this stops changing.
_CONTENT_PROBE_JS = (
    "() => [document.body ? document.body.innerText.trim().length : 0,"
    " document.body ? document.body.getElementsByTagName('*').length : 0]"
)


async def wait_for_content(page: Page, *, timeout_s: float = 15.0, quiet_s: float = 1.2) -> bool:
    """Wait until the page shows text and its DOM has stopped changing for ``quiet_s``.

    SPAs often render nothing until auth/data calls resolve (or time out): `load` and `networkidle` fire on an empty
    shell. Returns False if the page never settled within ``timeout_s`` (we go on anyway).
    """
    with contextlib.suppress(PlaywrightError):
        await page.wait_for_load_state("networkidle", timeout=3000)
    deadline = time.monotonic() + timeout_s
    last: list[int] | None = None
    stable_since: float | None = None
    while time.monotonic() < deadline:
        try:
            sig = await page.evaluate(_CONTENT_PROBE_JS)
        except PlaywrightError:  # navigation in progress
            sig = None
        now = time.monotonic()
        if sig and sig[0] > 0 and sig == last:
            if stable_since is not None and now - stable_since >= quiet_s:
                return True
        else:
            stable_since = now
        last = sig
        await asyncio.sleep(0.3)
    return False


@contextlib.contextmanager
def _fine_timer() -> Iterator[None]:
    """Ask Windows for 1 ms timer resolution while recording (no-op elsewhere)."""
    winmm = None
    if sys.platform == "win32":
        try:
            import ctypes

            winmm = ctypes.WinDLL("winmm")
            winmm.timeBeginPeriod(1)
        except OSError:
            winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(1)


async def launch_chromium(pw: Playwright, *, headed: bool, progress: ProgressFn | None = None) -> Browser:
    """Launch Chromium, installing it once on the fly if this Playwright version has no browser yet."""
    try:
        return await pw.chromium.launch(headless=not headed)
    except PlaywrightError as exc:
        if "Executable doesn't exist" not in str(exc) and "playwright install" not in str(exc):
            raise
    if progress:
        progress("Chromium not found - installing it once (playwright install chromium)...")
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"playwright install chromium failed:\n{proc.stderr[-2000:]}")
    return await pw.chromium.launch(headless=not headed)


class HumanRunner:
    """Executes a storyboard on a page with human pacing while logging camera intent."""

    def __init__(self, page: Page, rng: random.Random) -> None:
        self.page = page
        self.rng = rng
        w, h = VIEWPORT
        self.state = _State(mouse_x=0.62 * w, mouse_y=0.78 * h)
        self.telemetry: list[TelemetryFrame] = []
        self.clicks: list[ClickEvent] = []
        self._sampling = False

    # ------------------------------------------------------------------ telemetry
    async def sample_forever(self) -> None:
        period = 1.0 / SAMPLE_HZ
        self._sampling = True
        next_tick = time.time()
        while self._sampling:
            now = time.time()
            s = self.state
            x, y = s.mouse_at(now)
            self.telemetry.append(
                TelemetryFrame(
                    t=now,
                    mouse_x=x,
                    mouse_y=y,
                    is_clicking=s.is_clicking,
                    target_zoom=s.target_zoom,
                    focus_box=s.focus_box,
                    camera_mode=s.camera_mode,
                )
            )
            next_tick += period
            delay = next_tick - time.time()
            if delay < -0.1:  # we fell far behind (GC pause...): resync instead of bursting
                next_tick = time.time()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))

    def stop_sampling(self) -> None:
        self._sampling = False

    # ------------------------------------------------------------------ locators
    async def _element_box(self, index: int, loc: Locator) -> tuple[PWLocator, Box]:
        target = await find_locator(self.page, loc)
        try:
            await target.wait_for(state="visible", timeout=6000)
            await target.scroll_into_view_if_needed(timeout=3000)
            bb = await target.bounding_box()
        except PlaywrightError as exc:
            raise StepError(f"step {index}: could not find a visible element for {loc.describe()}") from exc
        if bb is None:
            raise StepError(f"step {index}: element {loc.describe()} has no bounding box")
        return target, (bb["x"], bb["y"], bb["width"], bb["height"])

    # ------------------------------------------------------------------ camera intent
    def _set_camera(self, step: ActionStep, box: Box | None) -> None:
        s = self.state
        cam = step.camera
        s.camera_mode = cam.mode
        if cam.mode == "wide":
            s.target_zoom, s.focus_box = 1.0, None
        elif cam.mode == "pan":
            s.target_zoom, s.focus_box = cam.zoom, None
        else:  # focus: keep the previous box if this step has no element (e.g. a wait)
            s.target_zoom = cam.zoom
            if box is not None:
                s.focus_box = box

    # ------------------------------------------------------------------ human motion
    async def move_to(self, x: float, y: float, target_width: float = 24.0) -> None:
        s = self.state
        x0, y0 = s.mouse_x, s.mouse_y
        dx, dy = x - x0, y - y0
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return
        duration = 0.32 + 0.11 * math.log2(dist / max(target_width, 8.0) + 1.0)
        duration = min(1.1, max(0.35, duration))
        nx, ny = -dy / dist, dx / dist  # unit normal
        sign = self.rng.choice((-1.0, 1.0))
        o1 = dist * self.rng.uniform(0.06, 0.18) * sign
        o2 = o1 * self.rng.uniform(0.35, 0.8)
        p1 = (x0 + dx / 3 + nx * o1, y0 + dy / 3 + ny * o1)
        p2 = (x0 + 2 * dx / 3 + nx * o2, y0 + 2 * dy / 3 + ny * o2)
        motion = _Motion((x0, y0), p1, p2, (x, y), time.time(), duration)
        s.motion = motion
        period = 1.0 / SAMPLE_HZ
        while True:
            now = time.time()
            if now >= motion.t1:
                break
            mx, my = motion.at(now)
            await self.page.mouse.move(mx, my)
            await asyncio.sleep(max(0.0, period - (time.time() - now)))
        await self.page.mouse.move(x, y)
        s.mouse_x, s.mouse_y = x, y
        s.motion = None

    def _aim(self, box: Box) -> tuple[float, float]:
        bx, by, bw, bh = box
        jx = self.rng.gauss(0.0, 0.06 * bw)
        jy = self.rng.gauss(0.0, 0.06 * bh)
        x = min(bx + bw - 2, max(bx + 2, bx + bw / 2 + jx))
        y = min(by + bh - 2, max(by + 2, by + bh / 2 + jy))
        return x, y

    async def click_here(self) -> None:
        s = self.state
        await self.page.mouse.down()
        s.is_clicking = True
        self.clicks.append(ClickEvent(t=time.time(), x=s.mouse_x, y=s.mouse_y))
        await asyncio.sleep(self.rng.uniform(0.070, 0.110))
        await self.page.mouse.up()
        s.is_clicking = False

    async def type_human(self, text: str) -> None:
        for ch in text:
            await self.page.keyboard.type(ch)
            delay = self.rng.uniform(0.035, 0.075)
            if ch == " " and self.rng.random() < 0.08:
                delay += self.rng.uniform(0.060, 0.140)
            await asyncio.sleep(delay)

    async def smooth_scroll(self, delta: float) -> None:
        n = self.rng.randint(12, 18)
        done = 0.0
        for i in range(1, n + 1):
            target = delta * minimum_jerk(i / n)
            await self.page.mouse.wheel(0, target - done)
            done = target
            await asyncio.sleep(1.0 / SAMPLE_HZ)

    # ------------------------------------------------------------------ steps
    async def run_step(self, index: int, step: ActionStep) -> None:
        box: Box | None = None
        target: PWLocator | None = None
        if step.locator is not None:
            target, box = await self._element_box(index, step.locator)
        if step.action == "show" and target is not None and box is not None:
            region = await result_region(target, box)
            s = self.state
            s.camera_mode, s.focus_box = "focus", region
            s.target_zoom = step.camera.zoom if step.camera.zoom > 1.0 else 1.35
        else:
            self._set_camera(step, box)

        match step.action:
            case "wait":
                pass
            case "show":
                assert box is not None
                # Presenter gesture: glide to the label of the result area, slower than a click, never clicking.
                await self.move_to(*self._aim(box), target_width=max(box[2], 160.0))
            case "hover":
                assert box is not None
                await self.move_to(*self._aim(box), target_width=box[2])
            case "click":
                assert box is not None
                await self.move_to(*self._aim(box), target_width=box[2])
                await asyncio.sleep(self.rng.uniform(0.06, 0.12))  # tiny human hesitation
                await self.click_here()
            case "fill":
                assert box is not None and target is not None and step.input_value is not None
                await self.move_to(*self._aim(box), target_width=box[2])
                await asyncio.sleep(self.rng.uniform(0.05, 0.10))
                await self.click_here()
                with contextlib.suppress(PlaywrightError):
                    if await target.input_value(timeout=500):
                        await self.page.keyboard.press("ControlOrMeta+A")
                        await self.page.keyboard.press("Backspace")
                await asyncio.sleep(self.rng.uniform(0.12, 0.22))
                await self.type_human(step.input_value)
            case "press":
                assert step.input_value is not None
                await self.page.keyboard.press(step.input_value)
            case "scroll":
                delta = float(step.input_value) if step.input_value is not None else 400.0
                if box is None:
                    await self.smooth_scroll(delta)
        await asyncio.sleep(step.dwell_after_ms / 1000.0)

    async def final_hold(self, steps: list[ActionStep]) -> None:
        last = steps[-1]
        if last.action == "wait" and last.camera.mode == "wide" and last.dwell_after_ms >= 1000:
            return
        s = self.state
        s.camera_mode, s.target_zoom, s.focus_box = "wide", 1.0, None
        await asyncio.sleep(1.0)


def _start_capture(cdp: CDPSession, cap: _Capture) -> None:
    def on_frame(params: dict[str, Any]) -> None:
        n = len(cap.frames)
        path = cap.work_dir / f"frame_{n:05d}.jpg"
        path.write_bytes(base64.b64decode(params["data"]))
        ts = float(params.get("metadata", {}).get("timestamp") or time.time())
        cap.frames.append(SourceFrame(t=ts, path=path))
        cap.first_frame.set()
        task = asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]}))
        cap.acks.add(task)
        task.add_done_callback(cap.acks.discard)

    cdp.on("Page.screencastFrame", on_frame)


async def record(
    url: str,
    storyboard: Storyboard,
    work_dir: Path,
    *,
    headed: bool = False,
    seed: int | None = None,
    report: Reporter = silent,
    auth: Path | None = None,
) -> Recording:
    """Open ``url``, play ``storyboard`` like a human and return the captured recording.

    ``auth`` is a Playwright storage-state file (cookies + localStorage) saved by ``pullreel login``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    n_steps = len(storyboard.steps)
    say(report, "recording", "start", "launching Chromium (Retina 2x)", 0, n_steps)
    rng = random.Random(seed)
    with _fine_timer():
        async with async_playwright() as pw:
            browser = await launch_chromium(pw, headed=headed, progress=lambda m: say(report, "recording", detail=m))
            try:
                context = await browser.new_context(
                    viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]},
                    device_scale_factor=SCALE,
                    color_scheme="dark",
                    reduced_motion="no-preference",
                    storage_state=str(auth) if auth else None,
                )
                page = await context.new_page()
                say(report, "recording", detail=f"opening {url}")
                await page.goto(url, wait_until="load")
                say(report, "recording", detail="waiting for the page to finish rendering")
                await wait_for_content(page)
                await page.evaluate("document.fonts ? document.fonts.ready.then(() => true) : true")

                runner = HumanRunner(page, rng)
                await page.mouse.move(runner.state.mouse_x, runner.state.mouse_y)

                cap = _Capture(work_dir=work_dir)
                cdp = await context.new_cdp_session(page)
                _start_capture(cdp, cap)
                await cdp.send(
                    "Page.startScreencast",
                    {"format": "jpeg", "quality": 92, "maxWidth": 2560, "maxHeight": 1600, "everyNthFrame": 1},
                )
                try:
                    await asyncio.wait_for(cap.first_frame.wait(), timeout=2.0)
                except TimeoutError:
                    # Nothing repainted yet: nudge the compositor with an invisible change.
                    await page.evaluate(
                        "(() => { const d = document.createElement('div');"
                        " d.style.cssText='position:fixed;width:1px;height:1px;opacity:0.01;left:0;top:0';"
                        " document.body.appendChild(d); requestAnimationFrame(() => d.remove()); })()"
                    )
                    await asyncio.wait_for(cap.first_frame.wait(), timeout=3.0)

                t_start = time.time()
                sampler = asyncio.create_task(runner.sample_forever())
                try:
                    for i, step in enumerate(storyboard.steps):
                        what = f"{step.action} {step.locator.describe()}" if step.locator else step.action
                        if step.input_value and step.action == "fill":
                            what += f" ← {step.input_value!r}"
                        say(report, "recording", detail=f"step {i + 1}/{n_steps} · {what}", completed=i, total=n_steps)
                        await runner.run_step(i, step)
                    say(report, "recording", detail="closing hold", completed=n_steps, total=n_steps)
                    await runner.final_hold(storyboard.steps)
                finally:
                    runner.stop_sampling()
                    await sampler
                t_end = time.time()

                with contextlib.suppress(PlaywrightError):
                    await cdp.send("Page.stopScreencast")
                if cap.acks:
                    await asyncio.gather(*list(cap.acks), return_exceptions=True)
            finally:
                await browser.close()

    frames = sorted(cap.frames, key=lambda f: f.t)
    if not frames:
        raise RuntimeError("the screencast produced no frames")
    recording = Recording(
        viewport=VIEWPORT,
        scale=SCALE,
        frames=frames,
        telemetry=runner.telemetry,
        clicks=runner.clicks,
        t_start=t_start,
        t_end=t_end,
    )
    recording.save_json(work_dir / "recording.json")
    fps = len(frames) / max(recording.duration, 1e-6)
    say(report, "recording", "done", f"{recording.duration:.1f}s of footage · {len(frames)} captures ({fps:.0f}/s at 2560x1600)")
    return recording


def record_sync(
    url: str,
    storyboard: Storyboard,
    work_dir: Path,
    *,
    headed: bool = False,
    seed: int | None = None,
    report: Reporter = silent,
    auth: Path | None = None,
) -> Recording:
    return asyncio.run(record(url, storyboard, work_dir, headed=headed, seed=seed, report=report, auth=auth))
