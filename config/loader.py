from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from . import WORKSPACE_DIR

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    server_name: str
    endpoint: Dict
    allowed_tools: List[str]
    sensitive_tools: List[str]


class ConfigLoaderError(Exception):
    pass


def load_server_configs(directory: Path | str = Path(__file__).parent / "servers") -> List[ServerConfig]:
    """Load server configurations from JSON files in ``directory``.

    Parameters
    ----------
    directory : Path or str
        Directory containing ``*.json`` server configuration files.

    Returns
    -------
    List[ServerConfig]
        Parsed server configurations.
    """
    dir_path = Path(directory)
    configs: List[ServerConfig] = []
    if not dir_path.exists():
        raise ConfigLoaderError(f"Config directory not found: {dir_path}")

    for json_file in dir_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # JSONDecodeError or others
            raise ConfigLoaderError(f"Failed to parse {json_file}: {e}")

        missing = [k for k in ["server_name", "endpoint", "allowed_tools", "sensitive_tools"] if k not in data]
        if missing:
            raise ConfigLoaderError(f"{json_file}: missing keys: {', '.join(missing)}")
        if not isinstance(data["allowed_tools"], list) or not isinstance(data["sensitive_tools"], list):
            raise ConfigLoaderError(f"{json_file}: 'allowed_tools' and 'sensitive_tools' must be lists")

        endpoint = data["endpoint"]
        # Replace workspace placeholder if present in args
        args = endpoint.get("args", [])
        endpoint["args"] = [str(a).replace("${WORKSPACE_DIR}", str(WORKSPACE_DIR)) for a in args]

        configs.append(
            ServerConfig(
                server_name=data["server_name"],
                endpoint=endpoint,
                allowed_tools=data["allowed_tools"],
                sensitive_tools=data["sensitive_tools"],
            )
        )

    return configs
