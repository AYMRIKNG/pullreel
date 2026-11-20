from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pullreel.cli import _slug
from pullreel.config import config_path, load_config, remember, saved_key
from pullreel.models import DiffSummary, FeatureCandidate
from pullreel.storyboard_agent import PROVIDERS, ScoutPick, parse_model_spec, provider_key
from pullreel.wizard import build_scenario_choices, custom_pick, detect_base_branch

DIFF = DiffSummary(
    base_ref="master",
    head_ref="feature",
    features=[
        FeatureCandidate(title="CoachingPage", route="/coaching", added=1020, labels=["Comparer deux groupes"],
                         commits=["Coaching : onglet Défense"], score=23),
        FeatureCandidate(title="StatEquipePage", route="/statequipe", added=1900, score=29),
        FeatureCandidate(title="SettingPage", route="/settings", added=800, labels=["Email"], score=20),
        FeatureCandidate(title="StatJoueursPage", route="/statjoueurs", added=1000, labels=["Ajouter un joueur"],
                         score=22),
    ],
)  # fmt: skip


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PULLREEL_CONFIG", str(tmp_path / "cfg" / "config.json"))
    for k in ("ANTHROPIC_API_KEY", "PULLREEL_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_scenario_choices_star_and_precheck_scout_picks_then_list_everything() -> None:
    recommended = [
        ScoutPick(title="Coaching analytics", route="/coaching", why="new tactical tabs"),
        ScoutPick(title="Team defense", route="/statequipe", why="new Analyse tab"),
        ScoutPick(title="Settings", route="/settings", why="profile", recommended=False),  # topped up, not chosen
    ]
    choices = build_scenario_choices(recommended, DIFF)
    labels = [c.label for c in choices]
    assert labels[0].startswith("★ Coaching analytics  /coaching")
    assert [c.checked for c in choices[:3]] == [True, True, False]
    assert not labels[2].startswith("★")
    # every other candidate is still offered, once, unchecked
    assert sum("/statjoueurs" in label for label in labels) == 1
    assert sum("/coaching" in label for label in labels) == 1
    assert choices[-1].pick is None and "Write my own" in labels[-1]


def test_custom_scenario_keeps_the_user_text_and_finds_the_page() -> None:
    pick = custom_pick(DIFF, "ajouter un joueur puis ouvrir sa fiche")
    assert pick.title == "ajouter un joueur puis ouvrir sa fiche"
    assert pick.why.startswith("user request")
    assert pick.route == "/statjoueurs"
    assert "FEATURE_TO_SHOW: ajouter un joueur" in pick.to_prompt()


def test_remembered_setup_and_consented_key_live_outside_the_repo(tmp_path: Path) -> None:
    remember("ollama/gemma4:26b")
    assert load_config() == {"model": "ollama/gemma4:26b"}
    assert saved_key("anthropic") is None
    remember("anthropic/claude-sonnet-5-5", provider="anthropic", key="sk-ant-saved")
    assert json.loads(config_path().read_text())["keys"] == {"anthropic": "sk-ant-saved"}
    assert config_path().is_relative_to(tmp_path / "cfg")
    assert provider_key(PROVIDERS["anthropic"]) == ("sk-ant-saved", "saved config")
    assert parse_model_spec("anthropic").api_key == "sk-ant-saved"


def test_environment_key_wins_over_saved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    remember("anthropic/claude-sonnet-5-5", provider="anthropic", key="sk-ant-saved")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert provider_key(PROVIDERS["anthropic"]) == ("sk-ant-env", "ANTHROPIC_API_KEY")


def test_output_file_names_are_readable_slugs() -> None:
    assert _slug("Coaching Analytics Dashboard") == "coaching-analytics-dashboard"
    assert _slug("Équipe : défense & zones chaudes!") == "equipe-defense-zones-chaudes"
    assert _slug("???") == "scenario"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_detect_base_branch_prefers_main_then_master(tmp_path: Path) -> None:
    def run(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    run("init", "-b", "master")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "T")
    (tmp_path / "a.txt").write_text("a")
    run("add", "-A")
    run("commit", "-m", "init")
    assert detect_base_branch(tmp_path) == "master"
    run("branch", "main")
    assert detect_base_branch(tmp_path) == "main"
