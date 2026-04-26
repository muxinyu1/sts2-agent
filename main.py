import argparse
import os
from pathlib import Path
import random
import string
import time
from config import Config, load_config
from agent import Sts2Agent
from llm import LLM
from network import Proxy
from game_env import game_env_instance
from tools import build_all_tools


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sts2-agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file (default: config.yaml in project root)",
    )
    return parser.parse_args()


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


def _start_singleplayer_run(proxy: Proxy, seed: str) -> None:
    run_seed = seed.strip()
    if not run_seed:
        raise ValueError("run.seed in config must be a non-empty string")

    rsp = proxy.post({"action": "start_singleplayer_run", "seed": run_seed})
    if not rsp.is_ok():
        raise RuntimeError(f"start_singleplayer_run failed: {rsp.error}")

def _death_to_main_menu(proxy: Proxy):
    rsp = proxy.post({"action": "death_to_main_menu"})
    if not rsp.is_ok():
        raise RuntimeError(f"death_to_main_menu failed: {rsp.error}")
    
def _generate_random_seed():
    chars = string.ascii_uppercase + string.digits
    seed = "".join(random.choices(chars, k=10))
    return seed

def main():
    args = _parse_args()
    config_path = args.config.expanduser().resolve()

    _load_dotenv(ENV_PATH)
    config = load_config(config_path)

    for _ in range(config.run.total_run_times):
        proxy = _build_proxy(config)
        # 注入proxy
        game_env_instance.insert_proxy(proxy)
        run_seed = (
            _generate_random_seed() if config.run.use_random_seed else config.run.seed
        )
        config.run.seed = run_seed
        _start_singleplayer_run(proxy, run_seed)
        # 等待加载
        time.sleep(10)

        llm = _build_llm(config)
        tools = build_all_tools(enable_query_cards_info_tool=False)

        agent = Sts2Agent.build(
            longterm_memories=[],
            llm=llm,
            all_available_tools=tools,
            config=config,
        )
        agent.play()
    
        # 等待“Continue”出现
        time.sleep(5)
        _death_to_main_menu(proxy)


if __name__ == "__main__":
    main()