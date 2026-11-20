"""Per-user settings remembered between runs (never stored inside a repository).

    Windows: %APPDATA%\\pullreel\\config.json      macOS/Linux: $XDG_CONFIG_HOME/pullreel/config.json (~/.config)

Holds the last AI choice ("model": "anthropic/claude-sonnet-5-5") and, only when the user explicitly agreed, API keys
("keys": {"anthropic": "sk-ant-..."}). Environment variables always win over saved keys.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def config_path() -> Path:
    override = os.environ.get("PULLREEL_CONFIG")
    if override:
        return Path(override)
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        base = Path(os.environ["APPDATA"])
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "pullreel" / "config.json"


def load_config() -> dict[str, Any]:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(data: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        path.chmod(0o600)  # owner-only where the OS supports it (keys may live here)
    return path


def saved_key(provider: str) -> str | None:
    keys = load_config().get("keys")
    value = keys.get(provider) if isinstance(keys, dict) else None
    return value if isinstance(value, str) and value else None


def remember(model: str, *, provider: str | None = None, key: str | None = None) -> Path:
    data = load_config()
    data["model"] = model
    if provider and key:
        keys = data.get("keys") if isinstance(data.get("keys"), dict) else {}
        keys[provider] = key
        data["keys"] = keys
    return save_config(data)


def forget_key(provider: str) -> None:
    data = load_config()
    if isinstance(data.get("keys"), dict) and data["keys"].pop(provider, None) is not None:
        save_config(data)
