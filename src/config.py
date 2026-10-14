"""Config loading.

One YAML file fully describes an experiment. We keep it as nested dicts with
small typed accessors rather than a deep dataclass tree, because you will be
adding fields constantly in the first few weeks and a rigid schema just gets in
the way.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.path.join("configs", "default.yaml")


class Config:
    """Thin wrapper over a nested dict with dotted-path lookup."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up 'runtime.n_ctx' style paths."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {dotted}")
        return value

    def override(self, dotted: str, value: Any) -> None:
        """Set a value in place, creating intermediate dicts as needed."""
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def model_path(self, role: str) -> str:
        """Absolute-ish path to a registered model's GGUF file."""
        spec = self.require(f"models.{role}")
        models_dir = self.get("models_dir", "models")
        return os.path.join(models_dir, spec["filename"])

    def model_label(self, role: str) -> str:
        return self.require(f"models.{role}")["label"]
