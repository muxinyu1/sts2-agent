from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class BattleRun:
	run_id: str
	floor_id: str
	battle_id: str
	trajectory_file: Path
	reward_file: Path
	hp_loss: float
	battle_session_key: str


@dataclass(frozen=True)
class BattleWeightInfo:
	weight: float
	advantage: float
	mapped_advantage: float
	hp_loss: float
	mean_hp_loss: float
	group_size: int
	battle_group_key: str
	battle_session_key: str
	battle_id: str
	run_id: str
	floor_id: str


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"构建 SFT 训练数据：战斗样本按同战斗多对局 hp_loss 计算 soft 权重，"
			"非战斗样本权重固定 1.0。"
		)
	)
	parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="日志根目录，默认: logs")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("distill") / "sft_weighted.jsonl",
		help="输出 JSONL 路径，默认: distill/sft_weighted.jsonl",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1.0,
		help="优势映射温度 t，映射公式 exp(advantage / t)，默认 1.0",
	)
	parser.add_argument("--encoding", type=str, default="utf-8", help="文件编码，默认 utf-8")
	return parser


def parse_last_json_obj(jsonl_file: Path, encoding: str) -> Optional[dict]:
	last_obj: Optional[dict] = None
	with jsonl_file.open("r", encoding=encoding) as f:
		for raw in f:
			line = raw.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			if isinstance(obj, dict):
				last_obj = obj
	return last_obj


def rel_posix(root: Path, p: Path) -> str:
	return p.relative_to(root).as_posix()


def discover_battle_runs(logs_dir: Path, encoding: str) -> Tuple[List[BattleRun], int]:
	battles: List[BattleRun] = []
	missing_or_invalid = 0

	for reward_file in sorted(logs_dir.rglob("battle_replay_rewards.jsonl")):
		battle_dir = reward_file.parent
		trajectory_file = battle_dir / "trajectory.jsonl"
		if not trajectory_file.exists():
			missing_or_invalid += 1
			continue

		rel_parts = battle_dir.relative_to(logs_dir).parts
		if len(rel_parts) < 3:
			missing_or_invalid += 1
			continue

		run_id = rel_parts[0]
		floor_id = rel_parts[1]
		battle_id = "/".join(rel_parts)

		reward_obj = parse_last_json_obj(reward_file, encoding=encoding)
		if not reward_obj:
			missing_or_invalid += 1
			continue

		hp_loss_raw = reward_obj.get("hp_loss")
		if not isinstance(hp_loss_raw, (int, float, str)):
			missing_or_invalid += 1
			continue
		try:
			hp_loss = float(hp_loss_raw)
		except ValueError:
			missing_or_invalid += 1
			continue

		battle_session_key_raw = reward_obj.get("battle_session_key")
		if isinstance(battle_session_key_raw, str) and battle_session_key_raw:
			battle_session_key = battle_session_key_raw
		else:
			# 兜底：缺失时按 run+floor 归并。
			battle_session_key = f"{run_id}::{floor_id}"

		battles.append(
			BattleRun(
				run_id=run_id,
				floor_id=floor_id,
				battle_id=battle_id,
				trajectory_file=trajectory_file,
				reward_file=reward_file,
				hp_loss=hp_loss,
				battle_session_key=battle_session_key,
			)
		)

	return battles, missing_or_invalid


def softmax(xs: List[float]) -> List[float]:
	if not xs:
		return []
	mx = max(xs)
	exps = [math.exp(x - mx) for x in xs]
	s = sum(exps)
	if s == 0:
		uniform = 1.0 / len(xs)
		return [uniform for _ in xs]
	return [v / s for v in exps]


def compute_battle_weights(
	battles: List[BattleRun],
	temperature: float,
) -> Tuple[Dict[Path, BattleWeightInfo], int]:
	groups: Dict[Tuple[str, str, str], List[BattleRun]] = defaultdict(list)
	for b in battles:
		groups[(b.run_id, b.floor_id, b.battle_session_key)].append(b)

	weight_map: Dict[Path, BattleWeightInfo] = {}
	for (run_id, floor_id, session_key), items in groups.items():
		mean_hp_loss = sum(x.hp_loss for x in items) / len(items)
		advantages = [mean_hp_loss - x.hp_loss for x in items]
		mapped = [math.exp(adv / temperature) for adv in advantages]
		weights = softmax(mapped)

		group_key = f"{run_id}::{floor_id}::{session_key}"
		for b, adv, mapped_adv, w in zip(items, advantages, mapped, weights):
			weight_map[b.trajectory_file.resolve()] = BattleWeightInfo(
				weight=float(w),
				advantage=float(adv),
				mapped_advantage=float(mapped_adv),
				hp_loss=float(b.hp_loss),
				mean_hp_loss=float(mean_hp_loss),
				group_size=len(items),
				battle_group_key=group_key,
				battle_session_key=session_key,
				battle_id=b.battle_id,
				run_id=b.run_id,
				floor_id=b.floor_id,
			)

	return weight_map, len(groups)


def iter_trajectory_samples(trajectory_file: Path, encoding: str) -> Iterable[Tuple[int, dict, str, str, str]]:
	with trajectory_file.open("r", encoding=encoding) as f:
		for line_no, raw in enumerate(f, start=1):
			line = raw.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			if not isinstance(obj, dict):
				continue

			system_prompt = obj.get("system_prompt")
			user_prompt = obj.get("user_prompt")
			response = obj.get("llm_response")
			if not isinstance(system_prompt, str):
				continue
			if not isinstance(user_prompt, str):
				continue
			if not isinstance(response, str):
				continue

			yield line_no, obj, system_prompt, user_prompt, response


def is_battle_trajectory_file(traj_file: Path) -> bool:
	return traj_file.parent.name.startswith("battle_")


def is_error_tool_call_sample(execution_has_error: object) -> bool:
	if isinstance(execution_has_error, bool):
		return execution_has_error
	if isinstance(execution_has_error, (int, float)):
		return execution_has_error != 0
	if isinstance(execution_has_error, str):
		return execution_has_error.strip().lower() in {"1", "true", "yes", "y"}
	return False


def main() -> None:
	args = build_arg_parser().parse_args()

	logs_dir: Path = args.logs_dir
	output_path: Path = args.output
	encoding: str = args.encoding
	temperature: float = float(args.temperature)

	if not logs_dir.exists() or not logs_dir.is_dir():
		raise FileNotFoundError(f"logs 目录不存在: {logs_dir}")
	if temperature <= 0:
		raise ValueError("temperature 必须 > 0")

	battles, invalid_battle_count = discover_battle_runs(logs_dir=logs_dir, encoding=encoding)
	battle_weight_map, battle_group_count = compute_battle_weights(battles=battles, temperature=temperature)

	all_traj_files = sorted(logs_dir.rglob("trajectory.jsonl"), key=lambda p: rel_posix(logs_dir, p))

	output_path.parent.mkdir(parents=True, exist_ok=True)
	total_records = 0
	battle_records = 0
	non_battle_records = 0
	battle_missing_reward_records = 0
	filtered_error_tool_call_records = 0

	with output_path.open("w", encoding=encoding) as out:
		for traj_file in all_traj_files:
			traj_resolved = traj_file.resolve()
			weight_info = battle_weight_map.get(traj_resolved)

			if weight_info is not None:
				source_type = "battle"
				weight = weight_info.weight
			elif is_battle_trajectory_file(traj_file):
				source_type = "battle_missing_reward"
				weight = 1.0
			else:
				source_type = "non_battle"
				weight = 1.0

			for line_no, obj, system_prompt, user_prompt, response in iter_trajectory_samples(
				trajectory_file=traj_file,
				encoding=encoding,
			):
				execution_has_error = obj.get("execution_has_error")
				if is_error_tool_call_sample(execution_has_error):
					filtered_error_tool_call_records += 1
					continue

				record = {
					"system_prompt": system_prompt,
					"user_prompt": user_prompt,
					"response": response,
					"weight": float(weight),
					"source_type": source_type,
					"sample_id": f"{rel_posix(logs_dir, traj_file)}#L{line_no}",
					"trajectory_file": rel_posix(logs_dir, traj_file),
					"line_no": line_no,
					"timestamp": obj.get("timestamp"),
					"step": obj.get("step"),
					"floor_index": obj.get("floor_index"),
					"battle_index": obj.get("battle_index"),
					"execution_has_error": execution_has_error,
				}

				if weight_info is not None:
					record.update(
						{
							"run_id": weight_info.run_id,
							"floor_id": weight_info.floor_id,
							"battle_id": weight_info.battle_id,
							"battle_session_key": weight_info.battle_session_key,
							"battle_group_key": weight_info.battle_group_key,
							"battle_group_size": weight_info.group_size,
							"hp_loss": weight_info.hp_loss,
							"mean_hp_loss": weight_info.mean_hp_loss,
							"advantage": weight_info.advantage,
							"mapped_advantage": weight_info.mapped_advantage,
						}
					)

				out.write(json.dumps(record, ensure_ascii=False) + "\n")

				total_records += 1
				if source_type == "battle":
					battle_records += 1
				elif source_type == "non_battle":
					non_battle_records += 1
				else:
					battle_missing_reward_records += 1

	print("=== SFT 数据构建完成 ===")
	print(f"logs_dir                    : {logs_dir}")
	print(f"battle 对局目录数            : {len(battles)}")
	print(f"battle 分组数(同战斗多对局)  : {battle_group_count}")
	print(f"battle 权重样本条数          : {battle_records}")
	print(f"non-battle 样本条数         : {non_battle_records}")
	print(f"battle缺少reward样本条数     : {battle_missing_reward_records}")
	print(f"过滤错误工具调用样本条数     : {filtered_error_tool_call_records}")
	print(f"总样本条数                  : {total_records}")
	print(f"reward 缺失/异常 battle 数   : {invalid_battle_count}")
	print(f"temperature                 : {temperature}")
	print(f"output                      : {output_path}")


if __name__ == "__main__":
	main()
