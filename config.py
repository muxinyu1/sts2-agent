from pathlib import Path

import yaml
from pydantic import AliasChoices, BaseModel, Field


class NetworkConfig(BaseModel):
	base_url: str = "http://localhost"
	port: int = 15526


class LLMSamplingConfig(BaseModel):
	temperature: float = 0.45
	top_p: float = 0.90
	use_default_sampling: bool = False
	temperature_battle: float = 0.25
	top_p_battle: float = 0.85
	temperature_decision: float = 0.70
	top_p_decision: float = 0.95
	high_diversity_state_types: list[str] = Field(
		default_factory=lambda: [
			"card_reward",
			"card_select",
			"hand_select",
			"event",
			"map",
			"rest_site",
			"rewards",
			"relic_select",
			"bundle_select",
			"shop",
			"fake_merchant",
			"treasure",
		]
	)


class LLMConfig(BaseModel):
	base_url: str = "https://llmapi.paratera.com/v1"
	model: str = "DeepSeek-R1-0528"
	env_key_name: str = "LLM_KEY"
	max_tokens: int = 2048
	max_token_retries: int = 1
	sampling: LLMSamplingConfig = Field(default_factory=LLMSamplingConfig)


class AgentConfig(BaseModel):
	debug: bool = True
	state_settle_seconds: float = 2.0
	enable_tool_optimization: bool = True
	enable_battle_replay: bool = Field(
		default=True,
		validation_alias=AliasChoices("enable_battle_replay", "enable_battle_judge"),
	)
	battle_replay_limit_per_floor_battle: int = 0


class ToolsConfig(BaseModel):
	enable_query_cards_info_tool: bool = False


class RunConfig(BaseModel):
	seed: str = "unknown"
	total_run_times: int = 5


class Config(BaseModel):
	network: NetworkConfig = Field(default_factory=NetworkConfig)
	llm: LLMConfig = Field(default_factory=LLMConfig)
	agent: AgentConfig = Field(default_factory=AgentConfig)
	tools: ToolsConfig = Field(default_factory=ToolsConfig)
	run: RunConfig = Field(default_factory=RunConfig)


def load_config(path: Path) -> Config:
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")

	payload = yaml.safe_load(path.read_text(encoding="utf-8"))
	if payload is None:
		payload = {}
	if not isinstance(payload, dict):
		raise ValueError("config.yaml root must be a mapping")

	return Config.model_validate(payload)
