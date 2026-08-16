# ─── Configuration (config.toml) ─────────────────────────────────────────────
import os
import sys
import tomllib

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.toml")


def load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        sys.exit(f"ERROR: config file not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)
