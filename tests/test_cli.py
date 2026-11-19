from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pullreel.cli import _outputs, _start_url, app, demo_storyboard
from pullreel.models import Storyboard


def test_outputs_from_extension(tmp_path: Path) -> None:
    mp4, gif = _outputs(tmp_path / "pr-demo.gif", None)
    assert (mp4.name, gif and gif.name) == ("pr-demo.mp4", "pr-demo.gif")
    mp4, gif = _outputs(tmp_path / "clip.mp4", None)
    assert (mp4.name, gif) == ("clip.mp4", None)
    mp4, gif = _outputs(tmp_path / "clip.mp4", True)
    assert gif is not None and gif.name == "clip.gif"
    mp4, gif = _outputs(tmp_path / "pr-demo.gif", False)
    assert gif is None
    mp4, gif = _outputs(tmp_path / "videos", None)  # a folder
    assert mp4 == tmp_path / "videos" / "demo.mp4" and gif == tmp_path / "videos" / "demo.gif"


def test_start_url() -> None:
    sb = Storyboard(feature_summary="x", start_path="/settings/billing", steps=demo_storyboard().steps)
    assert _start_url("http://localhost:3000", sb) == "http://localhost:3000/settings/billing"
    assert _start_url("http://localhost:3000/app/", sb) == "http://localhost:3000/settings/billing"
    assert _start_url("file:///C:/x/demo.html", sb) == "file:///C:/x/demo.html"
    sb.start_path = "/"
    assert _start_url("http://localhost:3000/todos", sb) == "http://localhost:3000/todos"


def test_help_lists_all_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("record", "plan", "render", "diff", "demo"):
        assert cmd in result.output


def test_version_works_without_a_command() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "pullreel 0.1.0" in result.output


def test_record_requires_url() -> None:
    result = CliRunner().invoke(app, ["record"])
    assert result.exit_code != 0


def test_diff_reports_git_errors_cleanly(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["diff", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "PullReel failed" in result.output
