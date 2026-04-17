import os
from pathlib import Path
from config import Config, load_config
from agent import Sts2Agent
from llm import LLM
from network import Proxy
from game_env import game_env_instance
from tools import build_all_tools


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


def _build_proxy(config: Config) -> Proxy:
    return Proxy(base_url=config.network.base_url, port=config.network.port)


def _build_llm(config: Config) -> LLM:
    env_key_name = config.llm.env_key_name
    key = os.getenv(env_key_name, "")
    if not key:
        raise ValueError(
            f"LLM key is missing. Please set '{env_key_name}' in environment or .env"
        )

    return LLM(
        base_url=config.llm.base_url,
        key=key,
        model=config.llm.model,
        max_tokens=config.llm.max_tokens,
        max_token_retries=config.llm.max_token_retries,
        temperature=config.llm.sampling.temperature,
        top_p=config.llm.sampling.top_p,
    )

def main():
    _load_dotenv(ENV_PATH)
    config = load_config(CONFIG_PATH)

    proxy = _build_proxy(config)
    # 注入proxy
    game_env_instance.insert_proxy(proxy)
    llm = _build_llm(config)
    tools = build_all_tools(
        enable_query_cards_info_tool=config.tools.enable_query_cards_info_tool
    )

    agent = Sts2Agent.build(
        longterm_memories=[],
        llm=llm,
        all_available_tools=tools,
        config=config,
    )
    agent.play()


if __name__ == "__main__":
    main()