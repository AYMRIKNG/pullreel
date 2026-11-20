"""Tests for Work Package 3: pullreel.toml validation, quality presets, structured logging, error handling, cp1252 safety."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pullreel.cli import app
from pullreel.logging_setup import configure_logging, redact_secrets
from pullreel.models import StudioTheme
from pullreel.project_config import ConfigError, build_render_profile, load_project_config


def test_pullreel_toml_loads_validates_and_resolves_relative_assets(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    logo = tmp_path / "assets" / "logo.png"
    logo.write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "pullreel.toml").write_text(
        """
[brand]
name = "Acme Pro"
accent_color = "#10B981"
logo = "assets/logo.png"

[theme]
background_preset = "midnight"
window_chrome = "windows"
show_url_bar = true
radius_px = 20
padding_px = 64

[intro]
enabled = true
title = "Acme Analytics"
subtitle = "Instant cohort insights"
duration_s = 1.5

[outro]
enabled = true
cta = "acme.dev"

[render]
aspect = "1:1"
resolution = "1080p"
quality = "ultra"
formats = ["mp4", "webp"]
screenshots = true

[safety]
dry_rehearsal = true
extra_blocklist = ["nuke", "wipe_db"]
""",
        encoding="utf-8",
    )
    cfg = load_project_config(tmp_path)
    assert cfg.source_path == tmp_path / "pullreel.toml"
    assert cfg.brand.name == "Acme Pro"
    assert cfg.safety.dry_rehearsal is True
    assert "nuke" in cfg.safety.extra_blocklist

    theme, settings = build_render_profile(cfg)
    assert theme.accent == "#10B981"
    assert theme.logo_path == str(logo.resolve())
    assert theme.background_style == "gradient"
    assert theme.window_chrome == "windows"
    assert theme.show_url_bar is True
    assert theme.aspect == "1:1"
    assert theme.resolution == "1080p"
    assert theme.quality == "ultra"
    assert theme.crf == 15
    assert settings.motion_blur == 0.50
    assert theme.screenshots is True

    # CLI overrides must win over pullreel.toml
    theme_ov, settings_ov = build_render_profile(cfg, aspect="9:16", quality="draft", chrome="none")
    assert theme_ov.aspect == "9:16"
    assert theme_ov.quality == "draft"
    assert theme_ov.output_fps == 30
    assert settings_ov.motion_blur == 0.0
    assert theme_ov.window_chrome == "none"


def test_invalid_pullreel_toml_raises_clean_config_error(tmp_path: Path) -> None:
    (tmp_path / "pullreel.toml").write_text("[render]\nunknown_key = 123\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown_key"):
        load_project_config(tmp_path)

    (tmp_path / "pullreel.toml").write_text("[broken toml ===\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid pullreel.toml"):
        load_project_config(tmp_path)


def test_quality_presets_adjust_fps_crf_and_motion_blur() -> None:
    base = StudioTheme()
    d_th, d_st = base.with_quality("draft")
    s_th, s_st = base.with_quality("standard")
    u_th, u_st = base.with_quality("ultra")
    assert d_th.output_fps == 30 and d_th.crf == 24 and d_st.motion_blur == 0.0
    assert s_th.output_fps == 60 and s_th.crf == 18 and s_st.motion_blur > 0.0
    assert u_th.output_fps == 60 and u_th.crf == 15 and u_st.motion_blur >= s_st.motion_blur


def test_structured_logging_writes_jsonl_and_redacts_secrets(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "pullreel.jsonl"
    logger = configure_logging(verbose=False, log_file=log_path)
    logger.info("Connecting with Bearer sk-ant-secretvalue1234567890")
    logger.debug("Finished step %s", "render")
    for h in logger.handlers:
        h.flush()
    lines = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "[REDACTED]" in lines[0]["message"]
    assert "sk-ant-secretvalue" not in lines[0]["message"]
    assert redact_secrets("x-api-key=abcd12345678") == "[REDACTED]"
    # Reset logger handlers cleanly
    configure_logging(verbose=False, log_file=None)


def test_cli_validates_new_flags_cleanly_without_traceback() -> None:
    res = CliRunner().invoke(app, ["demo", "--aspect", "21:9"])
    assert res.exit_code == 1
    assert "PullReel failed" in res.output
    assert "Traceback" not in res.output

    res2 = CliRunner().invoke(app, ["demo", "--quality", "hyper"])
    assert res2.exit_code == 1
    assert "PullReel failed" in res2.output
    assert "Traceback" not in res2.output


def test_cp1252_console_never_crashes_on_unicode_output() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="replace")
    from rich.console import Console

    c = Console(file=stream, force_terminal=False)
    c.print("★ Scenario 1 — Add player · score 92 · ✍ Write my own…")
    stream.flush()
    decoded = raw.getvalue().decode("cp1252")
    assert "Scenario 1" in decoded and "score 92" in decoded
