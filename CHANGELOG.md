# Changelog

All notable changes to PullReel are documented in this file.

## [0.2.0] - 2026-11-20

### Added — Studio Rendering Quality
- **Polished studio chassis**: configurable backgrounds (`studio`, `solid`, angled `gradient`, 7 presets including `midnight`, `aurora`, `sunset`, `ocean`, `carbon`, `light`, and Gaussian-blurred `wallpaper`), selectable window chrome (`macos`, `windows`, `none`), optional browser URL bar (`--url-bar`), and safe zero-padding/corner-radius rendering.
- **Upgraded spring camera**: speed-proportional temporal motion blur, Hanning-smoothed focus-box transitions with zoom-dip on long pans, lookahead anticipation, and configurable `focus_padding_px` so zoomed elements are never clipped.
- **Vector cursors**: 8x supersampled `arrow`, `pointer` (hand), and `text` (I-beam) sprites automatically matched to the target element during recording, plus `0.90` press squash, fast-motion trail, and dual-ring click ripples.
- **Animated overlays with bundled Inter font (SIL OFL 1.1)**: ease-out intro title cards (`--intro`), branded outro CTA cards (`--outro`), per-step captions (`ActionStep.caption`), and a tactile keystroke HUD during typing and shortcuts.
- **Smart reframing & multi-format export**: `--aspect 16:9|1:1|9:16|4:5` with focus-tracking reframing, `--resolution native|1080p|1440p|4k`, VP9 WebM (`--webm`), animated WebP (`--webp`), money-shot best-frame poster PNG, and annotated key-screenshots sheet (`--screenshots`).
- **Project branding (`pullreel.toml`)**: auto-discovered Pydantic-validated config for brand colors, logo, font, theme, intro/outro, render defaults, and safety rules.

### Added — Scenario Intelligence
- **Concrete workflow detection** (`pullreel.journeys`): detects entity creation (`+ Add…`, `New…`, `Create…`), editing, search/filter, comparison, export/import, and settings flows with business-oriented titles.
- **Realistic, safe form filling**: generates domain-coherent data from labels/placeholders/types/options (`@example.com` for emails; strictly skips passwords, cards, and secrets) and ends on a money-shot `show` step of the created result.
- **Multi-page journeys**: `Storyboard.journey` allows flows spanning 2–3 pages without triggering navigation errors or quality penalties.
- **Mutation safety**: extended destructive-action blocklist enforced in both exploration and rehearsal, `Storyboard.mutates` flag with pre-recording terminal warning, and `--dry-rehearsal` to rehearse forms without clicking the final submit button.
- **Sales-value scenario ranking**: scores candidates (0–100) by visual payoff, interactive depth, diff novelty, and workflow value, displayed in the interactive wizard.
- **Strict user-intent respect**: `--focus` and interactive "Write my own scenario…" picks are never silently replaced by the automatic composer; clear errors are reported if the AI fails.

### Added — Production Quality
- **Quality presets**: `--quality draft|standard|ultra` tuning fps, scale, motion blur, CRF, and x264 preset.
- **Parallel rendering pipeline**: multi-threaded frame rendering (`ThreadPoolExecutor`) with thread-safe LRU source cache, tight ROI affine warping, and fused edge-strip compositing piped directly to FFmpeg without writing PNGs to disk.
- **Structured logging & clean errors**: `--verbose` / `-v` and `--log-file PATH` (JSONL) with automatic API-key/token redaction; friendly Rich error panels with no raw user-facing tracebacks.

## [0.1.0]

- Initial open-source release: git diff analyzer, multi-provider LLM storyboard agent, automatic control-group composer, 2x Retina Playwright human runner, spring camera cinema engine, MP4 + adaptive GIF export, interactive wizard, and login session support.
