import os
from pathlib import Path
from typing import Any

import yaml
from agent import Sts2Agent
from llm import LLM
from network import Proxy
from game_env import game_env_instance
from tools import all_tools


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("config.yaml root must be a mapping")
    return payload


def _build_proxy(config: dict[str, Any]) -> Proxy:
    network_cfg = config.get("network")
    if not isinstance(network_cfg, dict):
        raise ValueError("config.yaml must contain 'network' mapping")

    base_url = network_cfg.get("base_url")
    port = network_cfg.get("port")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("config.network.base_url must be a non-empty string")
    if not isinstance(port, int):
        raise ValueError("config.network.port must be an integer")

    return Proxy(base_url=base_url, port=port)


def _build_llm(config: dict[str, Any]) -> LLM:
    llm_cfg = config.get("llm")
    if not isinstance(llm_cfg, dict):
        raise ValueError("config.yaml must contain 'llm' mapping")

    base_url = llm_cfg.get("base_url")
    model = llm_cfg.get("model")
    env_key_name = llm_cfg.get("env_key_name", "LLM_KEY")

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("config.llm.base_url must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("config.llm.model must be a non-empty string")
    if not isinstance(env_key_name, str) or not env_key_name.strip():
        raise ValueError("config.llm.env_key_name must be a non-empty string")

    key = os.getenv(env_key_name, "")
    if not key:
        raise ValueError(
            f"LLM key is missing. Please set '{env_key_name}' in environment or .env"
        )

    return LLM(base_url=base_url, key=key, model=model)

def main():
    _load_dotenv(ENV_PATH)
    config = _load_config(CONFIG_PATH)

    proxy = _build_proxy(config)
    # 注入proxy
    game_env_instance.insert_proxy(proxy)
    llm = _build_llm(config)

    agent = Sts2Agent.build(
        longterm_memories=[],
        llm=llm,
        all_available_tools=all_tools
    )
    agent.play()


if __name__ == "__main__":
    main()