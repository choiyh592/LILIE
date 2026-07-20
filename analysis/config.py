"""Config loading + path resolution for the analysis pipeline.

A single ``config.yaml`` holds every path and parameter. Each module reads the
same config so the ordered steps stay consistent and reproducible. Paths in the
config are resolved relative to the config file's own directory unless
absolute, so the config is portable across machines.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


def _resolve(base_dir: str, path: str) -> str:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


@dataclass
class Config:
    raw: Dict[str, Any]
    config_dir: str

    # convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, *keys: str) -> str:
        """Resolve a (possibly nested) path value against the config dir."""
        node: Any = self.raw
        for k in keys:
            node = node[k]
        return _resolve(self.config_dir, node)

    @property
    def output_dir(self) -> str:
        out = _resolve(self.config_dir, self.raw["paths"]["output_dir"])
        os.makedirs(out, exist_ok=True)
        return out

    def out(self, name: str) -> str:
        """Absolute path for an output artifact by filename."""
        return os.path.join(self.output_dir, name)


def load_config(config_path: str) -> Config:
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, config_dir=os.path.dirname(config_path))


def add_arg(parser):
    """Attach the standard --config argument to an argparse parser."""
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "analysis", "config.yaml")
    parser.add_argument("--config", type=str, default=default,
                        help="Path to config.yaml")
    return parser
