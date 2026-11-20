# PullReel Pro Upgrade Notes (v0.2.0)

## Summary by Work Package

### Work Package 1: Studio Rendering Quality
1. **Polished studio chassis** (`src/pullreel/cinema_engine.py`, `src/pullreel/models.py`):
   - Configurable backgrounds: `studio`, `solid`, angled `gradient`, 7 presets (`studio`, `midnight`, `aurora`, `sunset`, `ocean`, `carbon`, `light`), and Gaussian-blurred `wallpaper` with subtle studio vignette.
   - Selectable window chrome (`macos`, `windows`, `none`), adjustable corner radius, soft drop shadow, safe padding (including `padding_px=0`), and optional browser URL bar pill (`--url-bar`).
2. **Camera upgrades** (`src/pullreel/cinema_engine.py`):
   - Pre-computed camera trajectory with Hanning-smoothed transitions between focus boxes and gentle zoom-dip on long pans.
   - Automatic anticipation via telemetry lookahead (`lookahead_s = 0.25`).
   - Generous focus padding (`focus_padding_px = 48.0`) so zoomed targets are never clipped at the viewport edge.
   - Speed-proportional temporal motion blur (2–3 sub-frame crop blend along camera velocity when moving fast; zero overhead when settled).
3. **High-resolution vector cursors** (`src/pullreel/cinema_engine.py`, `src/pullreel/human_runner.py`):
   - Three 8x-supersampled vector cursor sprites (`arrow`, `pointer` hand, `text` I-beam) automatically matched to the target element during recording (`TelemetryFrame.cursor_type`).
   - Subtle drop shadow, press squash (`click_press_scale = 0.90`), fast-movement cursor motion trail, and dual-ring accent/white click ripples.
4. **Animated text overlays & bundled font** (`src/pullreel/overlays.py`, `src/pullreel/assets/fonts/`):
   - Bundled **Inter** variable font (`Inter-Variable.ttf` + `OFL.txt`, SIL Open Font License 1.1) with optical-size & weight variation support plus custom font override.
   - Ease-out animated intro title card (`--intro`), branded outro CTA card (`--outro`), per-step captions (`ActionStep.caption`), and bottom-right keystroke HUD (`Recording.keys`) with cached layers and tight-ROI compositing.
5. **Formats, smart reframing & documentation assets** (`src/pullreel/cinema_engine.py`, `src/pullreel/exporter.py`):
   - `--aspect 16:9|1:1|9:16|4:5` with smart reframing (zero distortion/letterboxing; camera follows `focus_box` and cursor).
   - `--resolution native|1080p|1440p|4k`.
   - Exports: H.264 MP4 (`yuv420p`, `+faststart`), VP9 WebM (`--webm`), animated WebP (`--webp`), adaptive 2-pass GIF (`< 9.5 MB`), automatic money-shot best-frame poster PNG (`<stem>-poster.png`), and annotated key-screenshots contact sheet + individual step PNGs (`--screenshots`).
6. **Project branding (`pullreel.toml`)** (`src/pullreel/project_config.py`):
   - Auto-discovered at `<repo>/pullreel.toml` (or `--config FILE`), parsed via stdlib `tomllib`, validated with strict Pydantic models (`extra="forbid"`), with relative asset path resolution and CLI override precedence.

### Work Package 2: Scenario Intelligence
1. **Concrete workflow detection** (`src/pullreel/journeys.py`):
   - Detects real workflows (`create`, `edit`, `search`, `compare`, `export`, `import`, `settings`) beyond control groups and assigns business-oriented titles (e.g. `"Add a new player in Team Roster"`).
2. **Realistic, safe form filling** (`src/pullreel/journeys.py`, `src/pullreel/human_runner.py`, `src/pullreel/storyboard_agent.py`):
   - Inspects open dialogs/forms (`input`, `textarea`, `select`, `checkbox`, `radio`) and generates domain-coherent values respecting types, labels, placeholders, `<option>` lists, and `min`/`max`.
   - Always uses `@example.com` for emails; strictly skips passwords, credit cards, CVV, IBAN, SSN, API keys, and tokens.
   - Follows the complete story arc: open form → focus-fill each field → submit → `show` the newly created item in the list (money shot) → wide hold.
3. **Multi-step & multi-page journeys** (`src/pullreel/models.py`, `src/pullreel/storyboard_agent.py`, `src/pullreel/composer.py`):
   - Added `Storyboard.journey: bool` allowing scenarios to cross 2–3 pages without `LEFT THE FEATURE PAGE` errors or quality penalties.
4. **Mutation safety** (`src/pullreel/journeys.py`, `src/pullreel/storyboard_agent.py`, `src/pullreel/cli.py`):
   - Extended destructive-action blocklist (`DESTRUCTIVE_RE` + `safety.extra_blocklist`) enforced in both exploration and rehearsal.
   - Automatic `Storyboard.mutates` flagging + yellow terminal warning before recording.
   - `--dry-rehearsal` verifies the final submit button is visible and enabled without clicking it.
5. **Scenario sales ranking** (`src/pullreel/journeys.py`, `src/pullreel/wizard.py`, `src/pullreel/storyboard_agent.py`):
   - 0–100 sales-value score combining visual payoffs, screen-changing steps, workflow richness, and diff novelty, displayed alongside `[mutates]` tags in the interactive wizard.
6. **Strict user-intent respect** (`src/pullreel/storyboard_agent.py`, `src/pullreel/wizard.py`):
   - Custom scenarios (`--focus` or wizard "Write my own scenario…") are tagged `user_requested=True`, are never silently replaced by the automatic composer, and fail with a clear error message if the AI director fails or `--director composer` is chosen.

### Work Package 3: Production Quality
- **Typed configuration & friendly errors**: Pydantic-validated `pullreel.toml` (`ConfigError`), clean Rich error panels across all CLI commands (no raw user tracebacks).
- **Structured logging** (`src/pullreel/logging_setup.py`): `--verbose` / `-v` (stderr) and `--log-file PATH` (JSONL) with automatic API-key / bearer-token redaction.
- **Quality presets**: `--quality draft|standard|ultra` tuning fps, scale, motion blur, CRF, and x264 preset.
- **Render performance (measured before/after)** on a 180-frame (3 s @ 60 fps, 1392×912, 2x Retina source) synthetic benchmark, piping raw frames directly to FFmpeg with zero disk PNGs:
  - **Before (v0.1.0, serial, no blur)**: `render_only = 6.514 s (27.6 fps)`, `render + MP4 = 8.320 s (21.6 fps)`.
  - **After (v0.2.0, 4 worker threads + LRU source cache + tight ROI warp + fused edge strips, no blur)**: `render_only = 2.213 s (81.3 fps)` (**2.94× faster**, `-66%` frame render time), `render + MP4 = 5.617 s (32.0 fps)` (**1.48× faster** end-to-end).
  - **After (v0.2.0, 4 worker threads WITH 3-subframe temporal motion blur)**: `render_only = 4.669 s (38.6 fps)` (**1.40× faster** than the non-blurred baseline despite 3× sub-frame sampling).
- **Documentation & tests**: updated `README.md`, created `CHANGELOG.md` (`0.2.0`), bumped package version to `0.2.0`, and added 27 offline pytest tests covering studio rendering, workflows/forms/safety, and config/CLI/logging.

---

## Files Touched

### Modified
- `pyproject.toml` (version bumped to `0.2.0`)
- `README.md` (studio rendering, `pullreel.toml`, formats/resolutions/quality tables, journeys & safety)
- `src/pullreel/__init__.py` (`__version__ = "0.2.0"`)
- `src/pullreel/models.py` (captions, keystrokes, cursor types, aspect/resolution/quality/chrome/background/format types, `journey` & `mutates` flags, `StudioTheme.with_quality()`)
- `src/pullreel/cinema_engine.py` (chassis/chrome/URL bar, backgrounds/presets/wallpaper, smart reframing, camera smoothing/padding/motion blur, 3 vector cursors + trail + dual ripple, overlays integration, money-shot tracker, key screenshots, parallel ThreadPoolExecutor pipeline)
- `src/pullreel/exporter.py` (MP4 preset, VP9 WebM, animated WebP, annotated screenshot sheet export)
- `src/pullreel/human_runner.py` (target cursor-type detection, `<select>` filling, keystroke & caption capture, recording URL & summary metadata)
- `src/pullreel/composer.py` (shared extended destructive blocklist, step captions, multi-page `journey` scoring)
- `src/pullreel/storyboard_agent.py` (upgraded `SYSTEM_PROMPT` with captions/`journey`/`mutates`, multi-page & `--dry-rehearsal` rehearsal, destructive-click guard, journey integration, strict user-intent enforcement)
- `src/pullreel/wizard.py` (sales-score & `[mutates]` display, `user_requested=True` on custom picks)
- `src/pullreel/cli.py` (all new flags, `pullreel.toml` loading, mutation warnings, structured logging, clean error handling, multi-format render orchestration)
- `src/pullreel/progress.py`, `src/pullreel/progress_ui.py` (`exports` stage)
- `tests/test_cli.py` (updated version assertion to `0.2.0`)

### Added
- `src/pullreel/overlays.py` (Inter OFL font loader, cached intro/outro cards, tight-ROI captions & keystroke HUD, annotated screenshots & contact sheet)
- `src/pullreel/project_config.py` (`pullreel.toml` discovery, Pydantic validation, CLI override merging)
- `src/pullreel/journeys.py` (workflow detection, realistic non-sensitive form data generation, form journey explorer, destructive blocklist, mutation detector, sales-value ranking)
- `src/pullreel/logging_setup.py` (structured `--verbose` / `--log-file` JSONL logging with secret redaction)
- `src/pullreel/assets/fonts/Inter-Variable.ttf` & `src/pullreel/assets/fonts/OFL.txt`
- `tests/test_studio_render.py` (14 tests)
- `tests/test_journeys.py` (7 tests)
- `tests/test_config_and_pro.py` (6 tests)
- `CHANGELOG.md`
- `UPGRADE_NOTES.md`

*(Pre-existing working-tree edits on `upgrade-pro` prior to this mission, such as `src/pullreel/net.py`, `tests/test_composer.py`, `tests/test_providers_scout.py`, and `uv.lock`, were preserved without modifying `uv.lock`.)*

---

## New CLI Options

Available across `record`, `render`, `demo` (and where applicable `plan`, `start`, and root callback):

| Option | Values / Type | Purpose |
| --- | --- | --- |
| `--aspect` | `16:9`, `1:1`, `9:16`, `4:5` | Output aspect ratio with focus-following smart reframing |
| `--resolution` | `native`, `1080p`, `1440p`, `4k` | Target output resolution |
| `--quality` | `draft`, `standard`, `ultra` | Quality/speed preset (fps, scale, motion blur, CRF, x264 preset) |
| `--chrome` | `macos`, `windows`, `none` | Window frame style |
| `--url-bar` / `--no-url-bar` | flag | Show a browser URL pill in the window title bar |
| `--bg-preset` | `studio`, `midnight`, `aurora`, `sunset`, `ocean`, `carbon`, `light` | Studio backdrop preset |
| `--intro` / `--no-intro` | flag | Animated ease-out intro title card |
| `--outro` / `--no-outro` | flag | Animated branded outro CTA card |
| `--formats` | e.g. `mp4,gif,webp,webm` | Comma-separated export formats |
| `--webm` | flag | Also export VP9 WebM |
| `--webp` | flag | Also export animated WebP |
| `--screenshots` / `--no-screenshots` | flag | Export annotated key-screenshots contact sheet + individual PNGs |
| `--config` | `PATH` | Explicit `pullreel.toml` path (default: auto-discovered at repo root) |
| `--dry-rehearsal` | flag | Rehearse mutating form flows without clicking the final submit button (`plan`, `record`, `start`) |
| `--verbose` / `-v` | flag | Structured debug logging to stderr |
| `--log-file` | `PATH` | Structured JSONL log file (secrets redacted) |

---

## Test & Verification Results

- **Full pytest suite**: **104 passed, 0 failed** (77 pre-existing tests + 27 new offline tests; zero network required).
- **End-to-end smoke test**: Verified `pullreel demo --aspect 16:9 --quality ultra --url-bar --intro --outro --webp --screenshots` on Windows + headless Chromium, producing:
  - `demo.mp4` (`1440x810`, H.264 `yuv420p`, 60 fps, 627 frames, 1.85 MB)
  - `demo.gif` (`852px`, 30 fps, 9.13 MB, under 9.5 MB budget)
  - `demo.webp` (animated WebP, 2.31 MB)
  - `demo-poster.png` (money-shot best frame, 274 KB)
  - `demo-screenshots.png` (annotated contact sheet, 385 KB) + 3 individual annotated step PNGs.

### Environment Note (Windows)
- In this deep scratch workspace path, creating `.venv` inside the repo hits Windows' 260-character MAX_PATH limit when loading Playwright's `_greenlet` DLL. Running with a short venv path (`$env:UV_PROJECT_ENVIRONMENT="$env:LOCALAPPDATA\pullreel-pro-venv"`) resolves this cleanly; no code changes were needed for standard repo paths.

---

## Remaining / Not Verified

- **Live external LLM provider calls**: Tested via mocked HTTP/LLM fixtures (zero network/API keys in tests per constraints), not against live paid cloud endpoints or a live local Ollama daemon.
- **4K full-length video encode**: 4K canvas sizing (`3840×2160`) is unit-tested, while end-to-end FFmpeg encoding was verified at `1440×810` (`ultra`) and synthetic `draft` resolutions to keep test runtime practical.
- **Visual human inspection of rendered MP4/WebP/PNG**: Automated tests verify frame dimensions, codecs, non-empty overlays, and smart-reframing camera coordinates; a quick manual playback of the demo output is recommended to fine-tune aesthetic preferences (colors, intro duration, caption sizing).

---

## Full Demo Command

```bash
pullreel demo --aspect 16:9 --quality ultra
```

To showcase all Pro assets in a single run (intro/outro cards, URL bar, animated WebP, and annotated screenshots sheet):

```bash
pullreel demo --aspect 16:9 --quality ultra --url-bar --intro --outro --webp --screenshots
```
