from typing import Any, List
import json
from pydantic import BaseModel

from exception import ToolNotExistException
from game import Act, Floor, Round
from game_env import GameEnv
from llm import LLM
from memory import LongtermMemory, ShorttermMemory
from network import MultipleResponse, SingleResponse
from state import State
from tools import Tool, ToolManager, generate_markdown_tools
from loguru import logger


class Sts2Agent(BaseModel):
    longterm_memories: List[LongtermMemory]
    shorterm_memories: List[ShorttermMemory]

    llm: LLM
    tool_manager: ToolManager

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
        turn_index = self.round.turn_index
        last_error = self.error.strip() if self.error else "none"

        system_prompt = f"""
You are an expert Slay the Spire 2 gameplay agent.
Your job is to choose the single best next action at each step.
Use only the tools listed below.

Available tools:
{tools}

Rules:
1. Choose exactly one tool call.
2. The tool name must match an available tool exactly.
3. Use only valid arguments with correct types.
4. Do not invent fields or extra keys.
5. If the previous action failed, correct the cause and choose a new valid action.
6. Keep reasoning concise and practical.

Output format (strict):
<think>your reasoning</think>
<tool>{{"name": "tool_name", "arguments": {{...}}}}</tool>

Do not output any text outside these two tags.
""".strip()

        user_prompt = f"""
Current run context:
- act_floor_count: {act_floor_count}
- current_turn_index: {turn_index}
- floor_turn_count: {floor_turn_count}
- floor_summary: {floor_summary}
- last_error: {last_error}

Recent actions:
{recent_actions_text}

Current game state:
{state_text}

Pick the best next action now.
""".strip()
        return system_prompt, user_prompt

    @classmethod
    def build(cls) -> 'Sts2Agent':
        return Sts2Agent(
            longterm_memories=[],
            shorterm_memories=[],
            error=None,
            llm=LLM(),
            state=State(),
            tool_manager=ToolManager(tools=[]),
            game_env=GameEnv(),
            act=Act(floors=[]),
            floor=Floor(turns=[], summary=""),
            round=Round(turn_index=0, actions=[]),
        )
    
    def refresh_state(self):
        self.state = self.game_env.state()
        # TODO 更新act、floor、round
        pass

    def play(self):
        while True:
            try:
                # 刷新最新状态
                self.refresh_state()
                # 构建prompt
                system_prompt, prompt = self.build_prompt()
                # 大模型做决策（理论上只有post）
                decison = self.llm.make_response(system_prompt, prompt) # decision是个json字符串
                # 解析并执行该决策，返回执行结果，理论上只有两种：
                #  ```jsonc
                # { "status": "ok", "message": "Playing 'Strike' targeting Jaw Worm" }
                # ```
                # ```jsonc
                # { "status": "error", "error": "Card requires a target. Provide 'target' with an entity_id." }
                # ```
                rsp = self.tool_manager.call(decison) # 这里decision是完整的回复，包括think，tool_manager会自动处理
                if isinstance(rsp, MultipleResponse):
                    pass
                elif isinstance(rsp, SingleResponse):
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


