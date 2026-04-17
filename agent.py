from typing import Any, List
import json
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from string import Template
from pydantic import BaseModel

from config import Config
from exception import ToolNotExistException
from game import Act, Floor, Round
from game_env import GameEnv
from llm import LLM
from memory import LongtermMemory, ShorttermMemory
from network import Response
from state import State
from tools import Tool, ToolManager, generate_markdown_tools, optimize_tools_for_state
from util import (
    extract_last_action,
    extract_plan,
    extract_summary,
    extract_think_preview,
    record_trajectory_sample,
)
from loguru import logger
from game_env import game_env_instance
from rich.console import Console
from rich.panel import Panel


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _safe_path_component(value: str, default: str = "unknown") -> str:
    raw = (value or "").strip()
    if not raw:
        raw = default

    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    safe = safe.strip("._-")
    return safe or default

BATTLE_STATE_TYPES = {"monster", "elite", "boss"}
console = Console()


@lru_cache(maxsize=8)
def load_prompt_template(filename: str) -> Template:
    template_path = PROMPTS_DIR / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return Template(template_path.read_text(encoding="utf-8").strip())


class Sts2Agent(BaseModel):
    config: Config

    longterm_memories: List[LongtermMemory]
    shorterm_memories: List[ShorttermMemory]
    recent_state_history: List[str]

    llm: LLM
    tool_manager: ToolManager

    last_action: str | None
    error: str | None

    state: State

    game_env: GameEnv

    act: Act
    floor: Floor
    round: Round

    tool_call_counts: dict[str, int]
    tool_call_total: int
    tool_call_error_total: int
    tool_prompt_tokens_before: int
    tool_prompt_tokens_after: int
    tool_prompt_optimization_steps: int
    run_log_dir_name: str

    def _debug(self, title: str, body: str, style: str = "cyan") -> None:
        if not self.config.agent.debug:
            return
        console.print(Panel(body, title=title, border_style=style, expand=False))

    def _summarize_state(self) -> str:
        state_type = "unknown"
        if isinstance(self.state.d, dict):
            raw_state_type = self.state.d.get("state_type")
            if isinstance(raw_state_type, str) and raw_state_type.strip():
                state_type = raw_state_type

        run_info = {}
        if isinstance(self.state.d, dict):
            raw_run = self.state.d.get("run")
            run_info = raw_run if isinstance(raw_run, dict) else {}

        act = run_info.get("act", "?")
        floor = run_info.get("floor", "?")
        error_text = self.error if self.error else "none"
        round_plan = self.round.plan.strip() if self.round.plan else "none"

        return (
            f"state_type: {state_type}\n"
            f"act/floor: {act}/{floor}\n"
            f"round_index: {self.round.round_index}\n"
            f"recent_actions: {len(self.round.actions)}\n"
            f"round_plan: {round_plan}\n"
            f"last_error: {error_text}"
        )

    def _sampling_params_for_state(self) -> tuple[float, float, str]:
        sampling_cfg = self.config.llm.sampling

        if sampling_cfg.use_default_sampling:
            return sampling_cfg.temperature, sampling_cfg.top_p, "default_forced"

        state_type = None
        if isinstance(self.state.d, dict):
            raw_state_type = self.state.d.get("state_type")
            if isinstance(raw_state_type, str) and raw_state_type.strip():
                state_type = raw_state_type.strip()

        if state_type in BATTLE_STATE_TYPES:
            return (
                sampling_cfg.temperature_battle,
                sampling_cfg.top_p_battle,
                "battle",
            )

        if state_type in set(sampling_cfg.high_diversity_state_types):
            return (
                sampling_cfg.temperature_decision,
                sampling_cfg.top_p_decision,
                "high_diversity",
            )

        return sampling_cfg.temperature, sampling_cfg.top_p, "default"

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        # Approximate token count using a common 4-chars-per-token heuristic.
        return max(1, (len(text) + 3) // 4)

    def _record_tool_prompt_optimization(self, before_tokens: int, after_tokens: int) -> None:
        self.tool_prompt_tokens_before += max(before_tokens, 0)
        self.tool_prompt_tokens_after += max(after_tokens, 0)
        self.tool_prompt_optimization_steps += 1

    def _record_tool_call(self, tool_name: str | None) -> None:
        self.tool_call_total += 1
        if not tool_name:
            tool_name = "unknown_tool"
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

    def _record_tool_call_error(self) -> None:
        self.tool_call_error_total += 1

    def _report_run_statistics(self) -> None:
        total_calls = self.tool_call_total
        error_calls = self.tool_call_error_total
        error_ratio = (error_calls / total_calls * 100.0) if total_calls > 0 else 0.0
        before = self.tool_prompt_tokens_before
        after = self.tool_prompt_tokens_after
        reduced = max(before - after, 0)
        reduction_ratio = (reduced / before * 100.0) if before > 0 else 0.0

        lines: list[str] = []
        lines.append(f"tool_calls_total: {total_calls}")
        lines.append(f"tool_calls_error_total: {error_calls}")
        lines.append(f"tool_calls_error_ratio: {error_ratio:.2f}%")
        lines.append("tool_calls_by_name:")

        if self.tool_call_counts:
            for tool_name, count in sorted(
                self.tool_call_counts.items(), key=lambda item: (-item[1], item[0])
            ):
                lines.append(f"- {tool_name}: {count}")
        else:
            lines.append("- none")

        lines.append("")
        lines.append("tool_selection_token_optimization:")
        lines.append(f"- steps: {self.tool_prompt_optimization_steps}")
        lines.append(f"- total_before: {before}")
        lines.append(f"- total_after: {after}")
        lines.append(f"- reduced_tokens: {reduced}")
        lines.append(f"- reduced_ratio: {reduction_ratio:.2f}%")

        stats_text = "\n".join(lines)
        self._debug("Run Statistics", stats_text, "bright_magenta")
        logger.info("Run statistics:\n{}", stats_text)

    def optimize_tool_selection(self) -> str:
        all_tools = self.tool_manager.tools
        state = self.state.d if isinstance(self.state.d, dict) else {}
        if self.config.agent.enable_tool_optimization:
            optimized_tools, details = optimize_tools_for_state(all_tools, state)
        else:
            optimized_tools = all_tools
            state_type = state.get("state_type") if isinstance(state.get("state_type"), str) else None
            details = {
                "state_type": state_type,
                "selection_reason": "tool_optimization_disabled",
                "all_tools": len(all_tools),
                "state_matched": len(all_tools),
                "optimized_tools": len(all_tools),
                "selected_tools": [t.name for t in all_tools],
            }

        optimized_tool_names = details.get("selected_tools", [])
        tool_names_preview = ", ".join(optimized_tool_names[:10]) if optimized_tool_names else "none"
        if len(optimized_tool_names) > 10:
            tool_names_preview += ", ..."

        all_tools_markdown = generate_markdown_tools([t.func for t in all_tools])
        optimized_functions = [t.func for t in optimized_tools]
        optimized_markdown = generate_markdown_tools(optimized_functions)

        before_tokens = self._estimate_tokens(all_tools_markdown)
        after_tokens = self._estimate_tokens(optimized_markdown)
        reduced_tokens = max(before_tokens - after_tokens, 0)
        reduced_ratio = (reduced_tokens / before_tokens * 100.0) if before_tokens > 0 else 0.0
        self._record_tool_prompt_optimization(before_tokens, after_tokens)

        self._debug(
            "Tool Selection",
            (
                f"state_type: {details.get('state_type') or 'none'}\n"
                f"selection_reason: {details.get('selection_reason')}\n"
                f"all_tools: {details.get('all_tools')}\n"
                f"state_matched: {details.get('state_matched')}\n"
                f"optimized_tools: {details.get('optimized_tools')}\n"
                f"selected_tools: {tool_names_preview}\n"
                f"token_before: {before_tokens}\n"
                f"token_after: {after_tokens}\n"
                f"token_reduced: {reduced_tokens} ({reduced_ratio:.2f}%)"
            ),
            "bright_blue",
        )
        return optimized_markdown

    def _recent_state_history_text(self) -> str:
        recent = self.recent_state_history[-10:] if self.recent_state_history else []
        return "\n".join(f"- {item}" for item in recent) if recent else "- none"

    def _fallback_summary(self, tool_payload: str | None) -> str:
        state_type = "unknown"
        act = "?"
        floor = "?"
        if isinstance(self.state.d, dict):
            raw_state_type = self.state.d.get("state_type")
            if isinstance(raw_state_type, str) and raw_state_type.strip():
                state_type = raw_state_type
            raw_run = self.state.d.get("run")
            if isinstance(raw_run, dict):
                act = raw_run.get("act", "?")
                floor = raw_run.get("floor", "?")

        action_desc = tool_payload if tool_payload else "no_tool_payload"
        return f"state={state_type}, act/floor={act}/{floor}, action={action_desc}"

    def _fallback_action_summary(self, tool_name: str, message: str) -> str:
        text = " ".join(str(message).split())
        max_len = 220
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        return f"{tool_name}: {text}"

    def _summarize_action_message(self, tool_name: str, message: str) -> str:
        if tool_name != "query_cards_info":
            return message

        raw_message = str(message).strip()
        if not raw_message:
            return f"{tool_name}: no result"

        clipped = raw_message[:4000]
        system_prompt = (
            "You are a concise game-memory summarizer. "
            "Summarize the tool result into exactly one short sentence for future action context. "
            "Keep only card names and the most decision-relevant stats. "
            "Do not output XML tags, markdown, lists, or extra explanation."
        )
        user_prompt = (
            f"tool_name: {tool_name}\n"
            "Please summarize this tool output:\n"
            f"{clipped}"
        )

        try:
            summary_rsp = self.llm.make_response(system_prompt, user_prompt)
            summary_text = " ".join(str(summary_rsp).split()).strip()
            if not summary_text:
                return self._fallback_action_summary(tool_name, raw_message)

            max_len = 280
            if len(summary_text) > max_len:
                summary_text = summary_text[: max_len - 3] + "..."
            return f"{tool_name}: {summary_text}"
        except Exception as e:
            logger.warning(f"Action summary fallback used for {tool_name}: {e}")
            return self._fallback_action_summary(tool_name, raw_message)

    def build_prompt(self) -> tuple[str, str]:
        tools = self.optimize_tool_selection().strip()
        if not tools:
            tools = "No tools available."

        state_text = (self.state.md or "").strip()
        if not state_text and self.state.d:
            state_text = json.dumps(self.state.d, ensure_ascii=False, indent=2)
        if not state_text:
            state_text = "State is empty."

        recent_actions = self.round.actions[-5:] if self.round.actions else []
        recent_actions_text = "\n".join(f"- {a}" for a in recent_actions) or "- none"
        recent_state_history = self._recent_state_history_text()

        act_floor_count = len(self.act.floors) if self.act.floors else 0
        floor_turn_count = len(self.floor.turns) if self.floor.turns else 0
        floor_summary = self.floor.summary or "No floor summary."
        round_index = self.round.round_index
        last_action = self.last_action.strip() if self.last_action else "none"
        current_round_plan = self.round.plan.strip() if self.round.plan else "none"
        last_error = self.error.strip() if self.error else "none"
        is_round_start = len(self.round.actions) == 0
        round_phase = "start" if is_round_start else "mid_round"
        plan_instruction = (
            "You are at the start of this combat round and no card has been played yet. "
            "Please think ahead for this round and include a concise natural-language plan in "
            "<plan></plan> before the tool call. The plan may include contingencies (e.g., draw outcomes) "
            "and does not need exact card-by-card certainty."
            if is_round_start
            else "Continue this round by following the existing plan when possible. "
            "If the plan is still valid, you may skip <think> and output only <tool>{...}</tool>. "
            "Do not output a new <plan> unless you need to revise it due to new information."
        )

        system_prompt = load_prompt_template("system_prompt.md").safe_substitute(
            tools=tools,
        )

        user_prompt = load_prompt_template("user_prompt.md").safe_substitute(
            act_floor_count=act_floor_count,
            turn_index=round_index,
            floor_turn_count=floor_turn_count,
            floor_summary=floor_summary,
            round_phase=round_phase,
            current_round_plan=current_round_plan,
            plan_instruction=plan_instruction,
            last_action=last_action,
            last_error=last_error,
            recent_actions_text=recent_actions_text,
            recent_state_history=recent_state_history,
            state_text=state_text,
        )
        return system_prompt, user_prompt

    @classmethod
    def build(
        cls,
        config: Config,
        longterm_memories: List[LongtermMemory],
        llm: LLM,
        all_available_tools: List[Tool],
    ) -> "Sts2Agent":
        return Sts2Agent(
            config=config,
            longterm_memories=longterm_memories,
            shorterm_memories=[],
            recent_state_history=[],
            last_action=None,
            error=None,
            llm=llm,
            state=State(),
            tool_manager=ToolManager(tools=all_available_tools),
            game_env=game_env_instance,
            act=Act(floors=[]),
            floor=Floor(turns=[], summary=""),
            round=Round(round_index=0, actions=[]),
            tool_call_counts={},
            tool_call_total=0,
            tool_call_error_total=0,
            tool_prompt_tokens_before=0,
            tool_prompt_tokens_after=0,
            tool_prompt_optimization_steps=0,
            run_log_dir_name=(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]}"
                f"_seed-{_safe_path_component(config.run.seed)}"
            ),
        )

    def refresh_state(self):
        previous_state = self.state.d or {}
        self.state = self.game_env.state()
        current_state = self.state.d or {}

        self._sync_act_and_floor(previous_state, current_state)
        self._sync_round(current_state)

    def _sync_act_and_floor(
        self, previous_state: dict[str, Any], current_state: dict[str, Any]
    ) -> None:
        prev_run_raw = previous_state.get("run")
        curr_run_raw = current_state.get("run")
        prev_run: dict[str, Any] = (
            prev_run_raw if isinstance(prev_run_raw, dict) else {}
        )
        curr_run: dict[str, Any] = (
            curr_run_raw if isinstance(curr_run_raw, dict) else {}
        )

        prev_act = prev_run.get("act")
        curr_act = curr_run.get("act")
        prev_floor = prev_run.get("floor")
        curr_floor = curr_run.get("floor")

        # 新开局或进入新 Act 时，重置 act/floor/round 结构
        if (
            isinstance(curr_act, int)
            and isinstance(prev_act, int)
            and curr_act != prev_act
        ):
            self.act = Act(floors=[])
            self.floor = Floor(turns=[], summary="")
            self.round = Round(round_index=0, actions=[])

        if not isinstance(curr_floor, int) or curr_floor < 0:
            return

        # run.floor 表示已访问楼层数：维护 floors 列表长度与之对齐
        while len(self.act.floors) < curr_floor:
            self.act.add_floor(Floor(turns=[], summary=""))

        if len(self.act.floors) > curr_floor:
            self.act.floors = self.act.floors[:curr_floor]

        if self.act.floors:
            self.floor = self.act.floors[-1]
            if self.floor.turns is None:
                self.floor.turns = []
        else:
            self.floor = Floor(turns=[], summary="")

        # 楼层变化时重置当前 round 指针；具体 round 内容由 _sync_round 在战斗态补齐
        if curr_floor != prev_floor:
            self.round = Round(round_index=0, actions=[])

    def _sync_round(self, current_state: dict[str, Any]) -> None:
        battle = current_state.get("battle")
        if not isinstance(battle, dict):
            return

        battle_round = battle.get("round")
        if not isinstance(battle_round, int) or battle_round < 0:
            return

        if self.floor.turns is None:
            self.floor.turns = []

        # 如果当前 floor 已有该 round，复用；否则创建新 round
        for existing_round in self.floor.turns:
            if existing_round.round_index == battle_round:
                self.round = existing_round
                return

        new_round = Round(round_index=battle_round, actions=[])
        self.floor.add_turn(new_round)
        self.round = new_round

    def _is_game_over_overlay(self) -> bool:
        if not isinstance(self.state.d, dict):
            return False

        if self.state.d.get("state_type") != "overlay":
            return False

        overlay = self.state.d.get("overlay")
        if not isinstance(overlay, dict):
            return False

        return overlay.get("screen_type") == "NGameOverScreen"

    def play(self):
        step = 0
        while True:
            had_previous_error = bool(self.error)
            execution_has_error = False
            tool_call_started = False
            tool_call_error_recorded = False
            system_prompt = ""
            prompt = ""
            decision: str | None = None
            selected_tool_name: str | None = None
            model_summary: str | None = None
            try:
                step += 1
                self._debug("Agent Loop", f"step: {step}", "blue")
                settle_seconds = self.config.agent.state_settle_seconds
                if step > 1 and settle_seconds > 0:
                    self._debug(
                        "Settle Wait",
                        f"sleeping {settle_seconds:.1f}s before state refresh",
                        "yellow",
                    )
                    time.sleep(settle_seconds)
                # 刷新最新状态
                self.refresh_state()
                self._debug("State Snapshot", self._summarize_state(), "magenta")
                if self._is_game_over_overlay():
                    self.error = "GameOver: NGameOverScreen"
                    self._debug(
                        "Game Over",
                        "Detected overlay screen 'NGameOverScreen'. Exiting play loop.",
                        "red",
                    )
                    break
                # 构建prompt
                system_prompt, prompt = self.build_prompt()
                self._debug(
                    "Prompt Stats",
                    f"system_prompt_len: {len(system_prompt)}\nuser_prompt:\n{prompt}",
                    "cyan",
                )
                temperature, top_p, sampling_profile = self._sampling_params_for_state()
                self._debug(
                    "Sampling",
                    (
                        f"profile: {sampling_profile}\n"
                        f"temperature: {temperature:.3f}\n"
                        f"top_p: {top_p:.3f}"
                    ),
                    "bright_cyan",
                )
                # 大模型做决策（理论上只有post）
                decision = self.llm.make_response(
                    system_prompt,
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                )  # decision是个json字符串
                think_preview = extract_think_preview(decision)
                plan_text = extract_plan(decision)
                model_summary = extract_summary(decision)
                tool_payload = extract_last_action(decision)
                if not model_summary:
                    model_summary = self._fallback_summary(tool_payload)
                self.recent_state_history.append(model_summary)
                if len(self.recent_state_history) > 10:
                    self.recent_state_history = self.recent_state_history[-10:]
                if plan_text:
                    self.round.plan = plan_text
                if tool_payload:
                    self.last_action = tool_payload
                    try:
                        payload_obj = json.loads(tool_payload)
                        if isinstance(payload_obj, dict):
                            maybe_tool_name = payload_obj.get("name")
                            if isinstance(maybe_tool_name, str) and maybe_tool_name.strip():
                                selected_tool_name = maybe_tool_name.strip()
                    except json.JSONDecodeError:
                        selected_tool_name = None
                decision_debug_lines = [
                    f"decision_len: {len(decision)}",
                    f"tool_payload: {tool_payload if tool_payload else 'none'}",
                ]
                if think_preview:
                    decision_debug_lines.append(f"think_preview: {think_preview}")
                if plan_text:
                    decision_debug_lines.append(f"plan: {plan_text}")
                self._debug("Model Decision", "\n".join(decision_debug_lines), "green")
                # 解析并执行该决策，返回执行结果，理论上只有两种：
                #  ```jsonc
                # { "status": "ok", "message": "Playing 'Strike' targeting Jaw Worm" }
                # ```
                # ```jsonc
                # { "status": "error", "error": "Card requires a target. Provide 'target' with an entity_id." }
                # ```
                tool_call_started = True
                self._record_tool_call(selected_tool_name)
                rsp = self.tool_manager.call(
                    decision
                )  # 这里decision是完整的回复，包括think，tool_manager会自动处理
                if isinstance(rsp, Response):
                    if rsp.is_ok():
                        assert rsp.message
                        action_message = rsp.message
                        if selected_tool_name:
                            action_message = self._summarize_action_message(
                                selected_tool_name, rsp.message
                            )
                        self.round.add_action(action_message)
                        self.error = None
                        execution_has_error = False
                        self._debug(
                            "Tool Response",
                            f"status: ok\nmessage: {action_message}",
                            "bright_green",
                        )
                    else:
                        # 重试
                        self.error = rsp.error
                        execution_has_error = True
                        self._record_tool_call_error()
                        tool_call_error_recorded = True
                        self._debug(
                            "Tool Response",
                            f"status: error\nerror: {rsp.error}",
                            "red",
                        )
            except ToolNotExistException as e:
                logger.error(e)
                self.error = f"{type(e).__name__}: {e}"
                execution_has_error = True
                if tool_call_started and not tool_call_error_recorded:
                    self._record_tool_call_error()
                self._debug("Exception", "ToolNotExistException\n" + str(e), "red")
                pass
            except Exception as e:
                logger.error(e)
                self.error = f"{type(e).__name__}: {e}"
                execution_has_error = True
                if tool_call_started and not tool_call_error_recorded:
                    self._record_tool_call_error()
                self._debug("Exception", f"{type(e).__name__}\n{e}", "red")
            finally:
                if decision is not None:
                    resolved_previous_error = had_previous_error and (not execution_has_error)
                    record_trajectory_sample(
                        LOGS_DIR,
                        run_dir_name=self.run_log_dir_name,
                        step=step,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        llm_response=decision,
                        execution_has_error=execution_has_error,
                        resolved_previous_error=resolved_previous_error,
                        model_summary=model_summary,
                        recent_state_history=self.recent_state_history[-10:],
                    )

        self._report_run_statistics()
