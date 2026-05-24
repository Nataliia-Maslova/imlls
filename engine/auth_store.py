"""
engine/auth_store.py — helpers for streamlit-authenticator YAML store.

The users.yaml lives at data/users.yaml and has this structure:
  credentials:
    usernames:
      user@email.com:
        email: user@email.com
        first_name: Natalia
        last_name: ''
        password: $2b$12$...   (bcrypt hash, auto-hashed by streamlit-authenticator)
        native_lang: Ukrainian  # custom field
        target_lang: English    # custom field
        logged_in: false
  cookie:
    expiry_days: 30
    key: <secret>
    name: imlls_auth
"""
from __future__ import annotations

import secrets
from pathlib import Path

import yaml

USERS_YAML = Path(__file__).parent.parent / "data" / "users.yaml"

_DEFAULT_CONFIG: dict = {
    "credentials": {
        "usernames": {}
    },
    "cookie": {
        "expiry_days": 30,
        "key": secrets.token_hex(32),   # unique per install
        "name": "imlls_auth",
    },
}


# ─── load / save ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load users.yaml, creating it with defaults if it doesn't exist."""
    if not USERS_YAML.exists():
        save_config(_DEFAULT_CONFIG)
        return _DEFAULT_CONFIG.copy()
    with open(USERS_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Ensure required keys exist
    cfg.setdefault("credentials", {}).setdefault("usernames", {})
    cfg.setdefault("cookie", _DEFAULT_CONFIG["cookie"])
    return cfg


def save_config(config: dict) -> None:
    """Write config dict back to users.yaml."""
    USERS_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_YAML, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)


# ─── user metadata ────────────────────────────────────────────────────────────

def get_user_meta(username: str) -> dict:
    """
    Return custom metadata for a registered user.
    username is the email used as the YAML key.
    Returns dict with native_lang, target_lang, display_name.
    """
    try:
        cfg  = load_config()
        user = cfg["credentials"]["usernames"].get(username, {})
        return {
            "native_lang":   user.get("native_lang", "Ukrainian"),
            "target_lang":   user.get("target_lang", "English"),
            "display_name":  user.get("first_name", username.split("@")[0]),
        }
    except Exception:
        return {"native_lang": "Ukrainian", "target_lang": "English",
                "display_name": username.split("@")[0]}


def save_user_meta(username: str, native_lang: str, target_lang: str) -> None:
    """Persist native/target language selection for a registered user."""
    try:
        cfg = load_config()
        users = cfg["credentials"]["usernames"]
        if username in users:
            users[username]["native_lang"] = native_lang
            users[username]["target_lang"] = target_lang
            save_config(cfg)
    except Exception as e:
        print(f"[auth_store] save_user_meta failed: {e}")


def list_users() -> list[dict]:
    """Return list of {username, display_name, native_lang, target_lang}."""
    try:
        cfg   = load_config()
        users = cfg["credentials"]["usernames"]
        return [
            {
                "username":     uname,
                "display_name": info.get("first_name", uname),
                "native_lang":  info.get("native_lang", "Ukrainian"),
                "target_lang":  info.get("target_lang", "English"),
            }
            for uname, info in users.items()
        ]
    except Exception:
        return []
