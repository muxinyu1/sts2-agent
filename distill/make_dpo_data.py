from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class BattleInfo:
	battle_id: str
	hp_loss: int
	reward_file: Path
	trajectory_file: Path


@dataclass(frozen=True)
class Sample:
	sample_id: str
	state_hash: str
	system_prompt: str
	user_prompt: str
	llm_response: str
	battle_id: str
	hp_loss: int
	trajectory_file: Path
	line_no: int
	step: Optional[int]


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"从 logs 目录构建 DPO 偏好数据：同状态(system+user prompt 相同)"
			" 且来自不同 battle 的样本两两配对。"
		)
	)
	parser.add_argument(
		"--logs-dir",
		type=Path,
		default=Path("logs"),
		help="日志根目录，默认: logs",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("distill") / "dpo_pairs.jsonl",
		help="输出 JSONL 路径，默认: distill/dpo_pairs.jsonl",
	)
	parser.add_argument(
		"--allow-equal-hp-loss",
		action="store_true",
		help="是否保留 hp_loss 相等的样本对（默认不保留）。",
	)
	parser.add_argument(
		"--min-margin",
		type=int,
		default=1,
		help="最小 margin（|hp_loss 差值|），默认 1。",
	)
	parser.add_argument(
		"--encoding",
		type=str,
		default="utf-8",
		help="读取日志文件编码，默认 utf-8",
	)
	return parser


def rel_battle_id(logs_dir: Path, battle_dir: Path) -> str:
	# 标准化为相对 logs 的稳定 ID，便于去重与追踪。
	return battle_dir.relative_to(logs_dir).as_posix()


def parse_last_reward_record(reward_file: Path, encoding: str) -> Optional[dict]:
	last_obj: Optional[dict] = None
	with reward_file.open("r", encoding=encoding) as f:
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


def collect_battles(logs_dir: Path, encoding: str) -> Dict[str, BattleInfo]:
	battles: Dict[str, BattleInfo] = {}
	for reward_file in logs_dir.rglob("battle_replay_rewards.jsonl"):
		battle_dir = reward_file.parent
		trajectory_file = battle_dir / "trajectory.jsonl"
		if not trajectory_file.exists():
			continue

		record = parse_last_reward_record(reward_file, encoding=encoding)
		if not record:
			continue

		hp_loss_val = record.get("hp_loss")
		if hp_loss_val is None:
			continue
		try:
			hp_loss = int(hp_loss_val)
		except (TypeError, ValueError):
			continue

		battle_id = rel_battle_id(logs_dir, battle_dir)
		battles[battle_id] = BattleInfo(
			battle_id=battle_id,
			hp_loss=hp_loss,
			reward_file=reward_file,
			trajectory_file=trajectory_file,
		)

	return battles


def hash_state(system_prompt: str, user_prompt: str) -> str:
	h = hashlib.sha256()
	# 分隔符使用不可见控制符，避免拼接歧义。
	h.update(system_prompt.encode("utf-8"))
	h.update(b"\x1f")
	h.update(user_prompt.encode("utf-8"))
	return h.hexdigest()


def iter_samples_for_battle(battle: BattleInfo, encoding: str) -> Iterable[Sample]:
	with battle.trajectory_file.open("r", encoding=encoding) as f:
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

			system_prompt_raw = obj.get("system_prompt")
			user_prompt_raw = obj.get("user_prompt")
			llm_response_raw = obj.get("llm_response")
			if not isinstance(system_prompt_raw, str):
				continue
			if not isinstance(user_prompt_raw, str):
				continue
			if not isinstance(llm_response_raw, str):
				continue

			system_prompt = system_prompt_raw
			user_prompt = user_prompt_raw
			llm_response = llm_response_raw

			state_hash = hash_state(system_prompt, user_prompt)
			step_raw = obj.get("step")
			try:
				step = int(step_raw) if step_raw is not None else None
			except (TypeError, ValueError):
				step = None

			sample_id = f"{battle.battle_id}#L{line_no}"
			yield Sample(
				sample_id=sample_id,
				state_hash=state_hash,
				system_prompt=system_prompt,
				user_prompt=user_prompt,
				llm_response=llm_response,
				battle_id=battle.battle_id,
				hp_loss=battle.hp_loss,
				trajectory_file=battle.trajectory_file,
				line_no=line_no,
				step=step,
			)


def collect_all_samples(
	battles: Dict[str, BattleInfo],
	encoding: str,
) -> Dict[str, List[Sample]]:
	groups: Dict[str, List[Sample]] = defaultdict(list)
	for battle in battles.values():
		for sample in iter_samples_for_battle(battle, encoding=encoding):
			groups[sample.state_hash].append(sample)
	return groups


def build_pair_record(chosen: Sample, rejected: Sample, margin: int, pair_id: str) -> dict:
	return {
		"pair_id": pair_id,
		"state_hash": chosen.state_hash,
		"system_prompt": chosen.system_prompt,
		"user_prompt": chosen.user_prompt,
		"prompt": chosen.user_prompt,
		"chosen": chosen.llm_response,
		"rejected": rejected.llm_response,
		"margin": margin,
		"chosen_hp_loss": chosen.hp_loss,
		"rejected_hp_loss": rejected.hp_loss,
		"chosen_sample_id": chosen.sample_id,
		"rejected_sample_id": rejected.sample_id,
		"chosen_battle_id": chosen.battle_id,
		"rejected_battle_id": rejected.battle_id,
		"chosen_step": chosen.step,
		"rejected_step": rejected.step,
		"chosen_line_no": chosen.line_no,
		"rejected_line_no": rejected.line_no,
		"chosen_trajectory_file": str(chosen.trajectory_file),
		"rejected_trajectory_file": str(rejected.trajectory_file),
	}


def generate_pairs(
	groups: Dict[str, List[Sample]],
	allow_equal_hp_loss: bool,
	min_margin: int,
) -> Iterable[dict]:
	for state_hash, samples in groups.items():
		n = len(samples)
		if n < 2:
			continue

		# 稳定排序，确保输出可复现。
		ordered = sorted(samples, key=lambda s: (s.battle_id, s.line_no, s.sample_id))

		for i in range(n):
			s1 = ordered[i]
			for j in range(i + 1, n):
				s2 = ordered[j]

				# 只保留“不同 battle”的配对。
				if s1.battle_id == s2.battle_id:
					continue

				margin = abs(s1.hp_loss - s2.hp_loss)
				if margin < min_margin:
					continue

				if s1.hp_loss == s2.hp_loss and not allow_equal_hp_loss:
					continue

				if s1.hp_loss < s2.hp_loss:
					chosen, rejected = s1, s2
				elif s1.hp_loss > s2.hp_loss:
					chosen, rejected = s2, s1
				else:
					# hp_loss 完全一致时，默认按 sample_id 稳定落位（若允许）。
					chosen, rejected = (s1, s2) if s1.sample_id <= s2.sample_id else (s2, s1)

				raw_pair_key = f"{state_hash}|{chosen.sample_id}|{rejected.sample_id}|{margin}"
				pair_id = hashlib.sha256(raw_pair_key.encode("utf-8")).hexdigest()[:24]
				yield build_pair_record(chosen=chosen, rejected=rejected, margin=margin, pair_id=pair_id)


def write_jsonl(records: Iterable[dict], output_path: Path, encoding: str) -> Tuple[int, int]:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	count = 0
	max_margin = 0
	with output_path.open("w", encoding=encoding) as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")
			count += 1
			margin = rec.get("margin")
			if isinstance(margin, int):
				max_margin = max(max_margin, margin)
	return count, max_margin


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	logs_dir: Path = args.logs_dir
	output: Path = args.output
	encoding: str = args.encoding
	allow_equal: bool = args.allow_equal_hp_loss
	min_margin: int = max(0, int(args.min_margin))

	if not logs_dir.exists() or not logs_dir.is_dir():
		raise FileNotFoundError(f"logs 目录不存在: {logs_dir}")

	battles = collect_battles(logs_dir=logs_dir, encoding=encoding)
	grouped_samples = collect_all_samples(battles=battles, encoding=encoding)

	all_samples = sum(len(v) for v in grouped_samples.values())
	pair_records = list(
		generate_pairs(
			groups=grouped_samples,
			allow_equal_hp_loss=allow_equal,
			min_margin=min_margin,
		)
	)

	# 基于 pair_id 再做一次全局去重，防止极端情况下重复写入。
	dedup: Dict[str, dict] = {}
	for rec in pair_records:
		dedup[rec["pair_id"]] = rec
	dedup_records = list(dedup.values())

	pair_count, max_margin = write_jsonl(dedup_records, output_path=output, encoding=encoding)

	print("=== DPO 数据构建完成 ===")
	print(f"logs_dir            : {logs_dir}")
	print(f"battle 数量         : {len(battles)}")
	print(f"样本总数            : {all_samples}")
	print(f"状态组数量          : {len(grouped_samples)}")
	print(f"输出文件            : {output}")
	print(f"偏好对数量          : {pair_count}")
	print(f"max_margin          : {max_margin}")
	print(f"allow_equal_hp_loss : {allow_equal}")
	print(f"min_margin          : {min_margin}")


if __name__ == "__main__":
	main()
