from typing import Any, List
import json
from functools import lru_cache
from pathlib import Path
from string import Template
from pydantic import BaseModel

from exception import ToolNotExistException
from game import Act, Floor, Round
from game_env import GameEnv
from llm import LLM
from memory import LongtermMemory, ShorttermMemory
from network import Response
from state import State
from tools import Tool, ToolManager, generate_markdown_tools
from loguru import logger


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_prompt_template(filename: str) -> Template:
    template_path = PROMPTS_DIR / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return Template(template_path.read_text(encoding="utf-8").strip())


class Sts2Agent(BaseModel):
    longterm_memories: List[LongtermMemory]
    shorterm_memories: List[ShorttermMemory]

    llm: LLM
    tool_manager: ToolManager

    last_action: str | None
    error: str | None

    state: State

    game_env: GameEnv

    act: Act
    floor: Floor
    round: Round


    def optimize_tool_selection(self) -> str:
        all_tools = self.tool_manager.tools
        # TODO 根据当前状态优化可选工具，减少工具数量，暂时不实现
        optimized_tools = all_tools
        optimized_functions = [t.func for t in optimized_tools]
        return generate_markdown_tools(optimized_functions)

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

        act_floor_count = len(self.act.floors) if self.act.floors else 0
        floor_turn_count = len(self.floor.turns) if self.floor.turns else 0
        floor_summary = self.floor.summary or "No floor summary."
        round_index = self.round.round_index
        last_error = self.error.strip() if self.error else "none"

        system_prompt = load_prompt_template("system_prompt.md").safe_substitute(
            tools=tools,
        )

        user_prompt = load_prompt_template("user_prompt.md").safe_substitute(
            act_floor_count=act_floor_count,
            turn_index=round_index,
            floor_turn_count=floor_turn_count,
            floor_summary=floor_summary,
            last_error=last_error,
            recent_actions_text=recent_actions_text,
            state_text=state_text,
        )
        return system_prompt, user_prompt

    @classmethod
    def build(cls) -> 'Sts2Agent':
        return Sts2Agent(
            longterm_memories=[],
            shorterm_memories=[],
            last_action=None,
            error=None,
            llm=LLM(),
            state=State(),
            tool_manager=ToolManager(tools=[]),
            game_env=GameEnv(),
            act=Act(floors=[]),
            floor=Floor(turns=[], summary=""),
            round=Round(round_index=0, actions=[]),
        )
    
    def refresh_state(self):
        previous_state = self.state.d or {}
        self.state = self.game_env.state()
        current_state = self.state.d or {}

        self._sync_act_and_floor(previous_state, current_state)
        self._sync_round(current_state)

    def _sync_act_and_floor(self, previous_state: dict[str, Any], current_state: dict[str, Any]) -> None:
        prev_run_raw = previous_state.get("run")
        curr_run_raw = current_state.get("run")
        prev_run: dict[str, Any] = prev_run_raw if isinstance(prev_run_raw, dict) else {}
        curr_run: dict[str, Any] = curr_run_raw if isinstance(curr_run_raw, dict) else {}

        prev_act = prev_run.get("act")
        curr_act = curr_run.get("act")
        prev_floor = prev_run.get("floor")
        curr_floor = curr_run.get("floor")

        # 新开局或进入新 Act 时，重置 act/floor/round 结构
        if isinstance(curr_act, int) and isinstance(prev_act, int) and curr_act != prev_act:
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

    def play(self):
        while True:
            try:
                # 刷新最新状态
                self.refresh_state()
                # 构建prompt
                system_prompt, prompt = self.build_prompt()
                # 大模型做决策（理论上只有post）
                decision = self.llm.make_response(system_prompt, prompt) # decision是个json字符串
                # 解析并执行该决策，返回执行结果，理论上只有两种：
                #  ```jsonc
                # { "status": "ok", "message": "Playing 'Strike' targeting Jaw Worm" }
                # ```
                # ```jsonc
                # { "status": "error", "error": "Card requires a target. Provide 'target' with an entity_id." }
                # ```
                rsp = self.tool_manager.call(decision) # 这里decision是完整的回复，包括think，tool_manager会自动处理
                if isinstance(rsp, Response):
                    # TODO 从decision解析出<tool></tool>中的json，赋值给self.last_action
                    if rsp.is_ok():
                        assert rsp.message
                        self.round.add_action(rsp.message)
                        self.error = None
                    else:
                        # 重试
                        self.error = rsp.error
            except ToolNotExistException as e:
                logger.error(e)
                pass
            except Exception as e:
                logger.error(e)
            finally:
                pass
