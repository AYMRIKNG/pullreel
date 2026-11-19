<div align="center">

<h1>DiffReel</h1>

<p><strong>Turn your git diff into a cinematic, ScreenStudio-style demo GIF for your pull request — locally, in one command.</strong></p>

<p>Local-first · Ollama by default · no account · 60 fps MP4 + GitHub-ready GIF</p>

</div>

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/before.gif" alt="Raw headless capture" width="100%" />
      <br />
      <sub><strong>Raw headless capture</strong> — no cursor, no camera</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/after.gif" alt="DiffReel output" width="100%" />
      <br />
      <sub><strong>DiffReel</strong> — 2x capture, spring camera, cursor, studio frame</sub>
    </td>
  </tr>
</table>

<p align="center">Full-quality 60 fps capture: <a href="docs/demo.mp4"><code>docs/demo.mp4</code></a></p>

## Try it in one command

<!-- replace OWNER once the repo is public; uvx works once published to PyPI -->

```bash
uvx diffreel demo
```

Or run from source:

```bash
git clone https://github.com/OWNER/diffreel && cd diffreel
uv run diffreel demo
```

`demo` records a bundled sample page with no LLM and no configuration. The first run downloads Chromium once; after that, a full run takes about 20 s on a laptop and writes `demo.mp4`, `demo.gif`, and `demo-poster.png` to `diffreel-out/`.

## Use it on your branch

Start your dev server, then point DiffReel at it:

```bash
diffreel record --url http://localhost:3000 --base main --out pr-demo.gif
```

1. **Diff** — inspects `merge-base(base, HEAD)` against your working tree (including uncommitted and untracked files), ignoring lockfiles and build artifacts, to extract changed routes, components, labels, test IDs, and handlers.
2. **Storyboard** — opens the app, captures Playwright's ARIA snapshot and visible interactive elements, and asks the LLM for a grounded 5–12 s storyboard focused on the delta.
3. **Human-paced recording** — drives headless Chromium at 2x Retina with curved mouse paths, natural typing cadence, CDP screencast capture, and 60 Hz telemetry.
4. **Cinematic render** — applies a spring camera, synthetic cursor, click ripples, and studio chassis, then encodes a 60 fps H.264 MP4 and an adaptive GitHub-ready GIF (< 5 MB), alongside `<stem>-poster.png` and `<stem>.storyboard.json`.

Drag `pr-demo.gif` (or the sharper `pr-demo.mp4` — GitHub plays it inline) into your PR description.

Want to review or tweak the shots before recording?

```bash
diffreel plan --url http://localhost:3000 --base main -o storyboard.json
# inspect or edit steps, zoom, dwell_after_ms
diffreel record --url http://localhost:3000 -s storyboard.json --out pr-demo.gif
```

## Commands

| Command | Purpose |
| --- | --- |
| `diffreel record --url URL` | Full pipeline: diff, plan (or `-s FILE`), record, render MP4/GIF, poster, and storyboard JSON. |
| `diffreel plan --url URL -o FILE` | Dry run: analyze the branch, ground the storyboard against the live page, write JSON. |
| `diffreel render DIR_OR_JSON` | Re-render a saved `--keep-frames` recording without launching a browser. |
| `diffreel diff [--base main]` | Inspect what DiffReel extracted from the branch (`--prompt` prints the exact LLM input). |
| `diffreel demo [-o diffreel-out]` | Record and render the bundled demo page with no LLM or project setup. |

Passing `--out pr-demo.gif` writes both `pr-demo.mp4` and `pr-demo.gif`. Passing `--out pr-demo.mp4` writes only the MP4 unless `--gif` is set.

## Models

Set `--model` (or `DIFFREEL_MODEL`) using a provider prefix. The default is `ollama/qwen2.5-coder:7b`; a bare model name (`llama3.2`) means Ollama.

| Prefix | Default endpoint | Auth / notes |
| --- | --- | --- |
| `ollama/<model>` | `http://localhost:11434` (`OLLAMA_HOST`) | Default. Local, no key required. |
| `lmstudio/<model>` | `http://localhost:1234/v1` | Local OpenAI-compatible server. |
| `openai/<model>` | OpenAI API | `OPENAI_API_KEY` |
| `openrouter/<model>` | OpenRouter API | `OPENROUTER_API_KEY` |
| `groq/<model>` | Groq API | `GROQ_API_KEY` |

Override any endpoint or key with `--base-url` (`DIFFREEL_BASE_URL`) and `--api-key` (`DIFFREEL_API_KEY`). On Ollama, DiffReel calls the native `/api/chat` endpoint with the `Storyboard` JSON schema as grammar-constrained structured output, a 16k context window, and thinking disabled.

```bash
OPENAI_API_KEY=sk-... diffreel record --url http://localhost:3000 --model openai/gpt-4o-mini
DIFFREEL_MODEL=ollama/qwen2.5-coder:7b diffreel plan --url http://localhost:3000
```

## How it works

```text
git diff ──> diff_analyzer ──> DiffSummary ──┐
                                             ├──> storyboard_agent (LLM) ──> Storyboard JSON
app URL  ──> Playwright ARIA snapshot ───────┘              ^                       │
                                                            │ error + live state    v
                                                            └─── rehearsal & grounding loop (<=3x)
                                                                                    │
  MP4 + GIF <── exporter (ffmpeg) <── cinema_engine <── human_runner <──────────────┘
                H.264 + <5 MB GIF     spring camera     Chromium 2x, CDP screencast,
                                      cursor, frame     60 Hz telemetry
```

| Module | Responsibility |
| --- | --- |
| `src/diffreel/diff_analyzer.py` | Git diff extraction, route inference (Next.js, SvelteKit, Nuxt, Remix, Astro, Django, FastAPI/Flask), UI symbol/label digest (capped at 6000 chars). |
| `src/diffreel/storyboard_agent.py` | Live ARIA snapshot, LLM prompt & structured output, normalization, Pydantic validation, pacing rules, and live rehearsal/grounding. |
| `src/diffreel/human_runner.py` | 2x Retina Chromium driver, CDP `Page.startScreencast`, Bézier mouse & typing simulation, 60 Hz telemetry. |
| `src/diffreel/cinema_engine.py` | Spring camera solver, 8x supersampled cursor sprite, click squash/ripple, and dark studio compositing. |
| `src/diffreel/exporter.py` | FFmpeg H.264 MP4 (`crf 18`, 60 fps, `faststart`), two-stage palette GIF under 5 MB, and PNG poster export. |
| `src/diffreel/models.py` | Pydantic contracts (`DiffSummary`, `Storyboard`, `Recording`, `StudioTheme`, `RenderSettings`). |
| `src/diffreel/cli.py` | Typer CLI (`record`, `plan`, `render`, `diff`, `demo`) and Rich progress output. |

## The craft

### Retina 2x capture
Chromium renders a `1280×800` viewport at `device_scale_factor=2` (`2560×1600`) and streams frames via CDP `Page.startScreencast` (JPEG q92, ~28 captures/s) instead of Playwright's blurry WebM recorder. Up to 2x zoom only ever downsamples with an area filter, so UI text stays sharp.

### Minimum-jerk Bezier mouse
Pointer paths follow cubic Bézier curves with random perpendicular curvature, Fitts-like travel times (0.35–1.1 s), and the minimum-jerk law `s(t) = 10t³ − 15t⁴ + 6t⁵`. Real `mouse.move` events fire hover states, clicks hold 70–110 ms, typing runs 35–75 ms/char, and the trajectory is stored analytically at 60 Hz.

### Critically damped spring camera
Camera `(x, y, zoom)` integrates `a = ω²(target − x) − 2ζω·v` (`ω = 6.5` rad/s, `ζ = 1`, semi-implicit Euler with 4 substeps) while reading telemetry 250 ms ahead so zooms anticipate clicks. Focus shots blend 80% element centre + 20% cursor, pan mode uses a 30% deadzone, and every crop stays clamped inside `2560×1600`.

### Synthetic cursor, click ripple, and studio frame
Headless Chromium draws no pointer, so DiffReel composites an 8x supersampled arrow with white outline, soft shadow, sub-pixel placement, 0.88 press squash, and a 250 ms fading click ripple. The frame sits on a `#09090B` backdrop with faint centre light, drop shadow, 16 px rounded corners, 1 px `#27272A` border, and 56 px padding (`1392×912` output).

## Zero hallucinated selectors

DiffReel directs the LLM to target only semantic locators (`role` + `name`, `label`, `placeholder`, `text`, `testid`; `css` is discouraged) drawn from the live ARIA snapshot. After normalizing small-model slips, validating with Pydantic, and enforcing pacing guardrails (400 ms wide opening, `fill` always in focus, focus zoom 1.3–1.9x, 1 s closing hold, ≤ 12 s total), it rehearses the entire storyboard at full speed in a fresh browser. Every locator must resolve at the exact moment it runs — including elements created by earlier steps. Misses are fuzzy-repaired against live elements; unresolved errors and current page state are sent back to the model for up to 3 attempts.

## Storyboard format

```json
{
  "feature_summary": "Filter issues and export the filtered view to CSV",
  "start_path": "/issues",
  "steps": [
    {
      "action": "wait",
      "camera": { "mode": "wide", "zoom": 1.0 },
      "dwell_after_ms": 400
    },
    {
      "action": "fill",
      "locator": { "strategy": "placeholder", "value": "Search issues..." },
      "input_value": "billing export",
      "camera": { "mode": "focus", "zoom": 1.75 },
      "dwell_after_ms": 450
    },
    {
      "action": "click",
      "locator": { "strategy": "role", "value": "button", "name": "Export CSV" },
      "camera": { "mode": "focus", "zoom": 1.5 },
      "dwell_after_ms": 600
    },
    {
      "action": "wait",
      "camera": { "mode": "wide", "zoom": 1.0 },
      "dwell_after_ms": 1000
    }
  ]
}
```

## Requirements & troubleshooting

<details>
<summary>Environment, proxies, Ollama, GPU sizing, and timings</summary>

- **Runtime**: Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).
- **FFmpeg**: Optional. DiffReel uses `ffmpeg` from `PATH` if present, otherwise falls back to the static binary bundled by `imageio-ffmpeg`.
- **Chromium**: Installed automatically by Playwright on the first run.
- **Corporate TLS proxy** (`invalid peer certificate: UnknownIssuer`): pass `--system-certs` to `uv`, e.g. `uv sync --system-certs`.
- **Ollama not running**: Start the daemon with `ollama serve`.
- **Model not pulled**: Run `ollama pull qwen2.5-coder:7b` (or pull whichever model you pass to `--model`).
- **Small GPU (e.g. 4 GB VRAM)**: Stick to 7B models. On a 4 GB laptop GPU, local 26B models took 50–175 s just to plan; 7B models are substantially faster.
- **Typical timings & sizes**: `diffreel demo` runs in about 20 s on a laptop (~1 MB for a 6.7 s 60 fps MP4, ~4.7 MB GIF). GIF export starts at 960 px / 30 fps and adapts width (down to 720 px) then fps to stay under 5 MB.

</details>

## Roadmap

Planned next, not yet implemented:

- GitHub Action that comments the rendered demo GIF directly on pull requests
- Pre-recording auth / login steps for authenticated apps
- Additional studio themes (including a light studio backdrop)
- Multi-page navigation flows

## Contributing

```bash
uv sync
uv run pytest
```

39 tests cover the diff analyzer (including real temporary git repos), the storyboard agent and the CLI helpers.

To regenerate the README comparison GIFs:

```bash
uv run diffreel demo --out docs --keep-frames
uv run python scripts/make_readme_gifs.py <kept-frames-dir> docs
```

## License

MIT
