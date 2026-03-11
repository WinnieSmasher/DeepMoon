"""YAML 配置加载与覆盖工具。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigNode(dict):
    """支持点号访问的轻量配置对象。"""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def copy(self) -> "ConfigNode":
        return ConfigNode({key: _to_config_node(value) for key, value in self.items()})

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


def _to_config_node(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode({key: _to_config_node(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    return value


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return {key: _to_plain_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:

    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def apply_overrides(config: ConfigNode, overrides: dict[str, Any] | None = None) -> ConfigNode:

    if not overrides:
        return config

    plain = config.to_dict()
    for dotted_key, value in overrides.items():
        current = plain
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return _to_config_node(plain)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ConfigNode:

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    config = _to_config_node(content)
    return apply_overrides(config, overrides)


def save_config(config: ConfigNode, path: str | Path) -> None:

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)


def parse_cli_overrides(pairs: Iterable[str]) -> dict[str, Any]:

    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"配置覆盖项格式错误: {pair}")
        key, raw_value = pair.split("=", 1)
        value = yaml.safe_load(raw_value)
        overrides[key] = value
    return overrides
