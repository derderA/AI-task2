from __future__ import annotations

import json
import os
from typing import Optional

from .config import Settings


class DeepSeekConfigService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def read_file_config(self) -> dict:
        path = self.settings.deepseek_config_path
        if path is None or not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return {
            "api_key": str(payload.get("api_key", "")).strip(),
            "base_url": str(payload.get("base_url", self.settings.deepseek_base_url)).strip() or self.settings.deepseek_base_url,
            "model": str(payload.get("model", self.settings.deepseek_model)).strip() or self.settings.deepseek_model,
        }

    def write_file_config(self, api_key: str, base_url: str, model: str) -> None:
        path = self.settings.deepseek_config_path
        if path is None:
            raise RuntimeError("当前未配置 DeepSeek 配置文件路径。")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "api_key": api_key.strip(),
            "base_url": base_url.strip() or self.settings.deepseek_base_url,
            "model": model.strip() or self.settings.deepseek_model,
        }
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def resolve_credentials(self, session_values: Optional[dict] = None) -> dict:
        session_values = session_values or {}
        session_key = str(session_values.get("api_key", "")).strip()
        if session_key:
            return {
                "source": "session",
                "api_key": session_key,
                "base_url": str(session_values.get("base_url", "")).strip() or self.settings.deepseek_base_url,
                "model": str(session_values.get("model", "")).strip() or self.settings.deepseek_model,
            }

        file_config = self.read_file_config()
        if file_config.get("api_key"):
            return {
                "source": "file",
                "api_key": file_config["api_key"],
                "base_url": file_config.get("base_url", self.settings.deepseek_base_url),
                "model": file_config.get("model", self.settings.deepseek_model),
            }

        env_key = os.getenv(self.settings.deepseek_api_key_env, "").strip()
        if env_key:
            return {
                "source": "env",
                "api_key": env_key,
                "base_url": os.getenv("DEEPSEEK_BASE_URL", self.settings.deepseek_base_url).strip() or self.settings.deepseek_base_url,
                "model": os.getenv("DEEPSEEK_MODEL", self.settings.deepseek_model).strip() or self.settings.deepseek_model,
            }

        return {
            "source": "none",
            "api_key": "",
            "base_url": self.settings.deepseek_base_url,
            "model": self.settings.deepseek_model,
        }

    def build_view_state(self, session_values: Optional[dict] = None) -> dict:
        resolved = self.resolve_credentials(session_values)
        file_config = self.read_file_config()
        return {
            "source": resolved["source"],
            "has_api_key": bool(resolved["api_key"]),
            "masked_api_key": self.mask_key(resolved["api_key"]),
            "base_url": resolved["base_url"],
            "model": resolved["model"],
            "config_file_path": str(self.settings.deepseek_config_path) if self.settings.deepseek_config_path else "",
            "config_file_exists": bool(file_config),
            "config_file_masked_key": self.mask_key(file_config.get("api_key", "")),
        }

    def mask_key(self, value: str) -> str:
        value = value.strip()
        if not value:
            return "未配置"
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"
