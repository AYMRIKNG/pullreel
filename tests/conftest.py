from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _never_read_the_real_user_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not see (or write) the developer's saved model / API keys."""
    cfg: Path = tmp_path_factory.mktemp("pullreel-config") / "config.json"
    monkeypatch.setenv("PULLREEL_CONFIG", str(cfg))
