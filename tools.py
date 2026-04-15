import inspect
import json
import types

from openai import BaseModel
from typing import Any, Callable, List, Literal, Union, get_args, get_origin

from exception import (
    ToolArgumentsValidationException,
    ToolNotExistException,
    ToolResponseFormatException,
)
from game_env import game_env_instance
from network import Response

class Arg(BaseModel):
    arg_name: str
    type: str

class Tool(BaseModel):
    state: str
    func: Callable
    name: str
    args: List[Arg]
    
    def call(self, args: dict[str, Any]) -> Any:
        return self.func(**args)
    
class ToolManager(BaseModel):
    tools: List[Tool]

    def _extract_json_payload(self, rsp: str) -> dict[str, Any]:
        content = rsp.strip()
        if "<tool>" in content and "</tool>" in content:
            start = content.find("<tool>") + len("<tool>")
            end = content.find("</tool>", start)
            content = content[start:end].strip()

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ToolResponseFormatException(
                f"Tool response is not valid JSON: {exc.msg}"
            ) from exc

        if not isinstance(payload, dict):
            raise ToolResponseFormatException("Tool response JSON must be an object")

        return payload

    def _matches_annotation(self, value: Any, annotation: Any) -> bool:
        if annotation in (inspect.Signature.empty, inspect.Parameter.empty, Any):
            return True

        origin = get_origin(annotation)
        union_type = getattr(types, "UnionType", None)

        if origin is Literal:
            return value in get_args(annotation)

        if origin in (Union, union_type):
            return any(self._matches_annotation(value, arg) for arg in get_args(annotation))

        if origin in (list, List):
            if not isinstance(value, list):
                return False
            args = get_args(annotation)
            if not args:
                return True
            return all(self._matches_annotation(v, args[0]) for v in value)

        if origin is dict:
            return isinstance(value, dict)

        if annotation in (None, type(None)):
            return value is None

        if annotation is int and isinstance(value, bool):
            return False

        if isinstance(annotation, type):
            return isinstance(value, annotation)

        return True

    def call(self, rsp: str) -> Response:
        """
        会抛出异常
        """
        payload = self._extract_json_payload(rsp)

        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ToolResponseFormatException("Tool response must include non-empty 'name'")

        arguments = payload.get("arguments", payload.get("args", {}))
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolArgumentsValidationException("'arguments' must be a JSON object")

        tool = self.get_tool(tool_name)
        signature = inspect.signature(tool.func)

        extra_args = [arg for arg in arguments.keys() if arg not in signature.parameters]
        if extra_args:
            raise ToolArgumentsValidationException(
                f"Unexpected arguments for tool '{tool_name}': {extra_args}"
            )

        missing_required_args = [
            param.name
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.default == inspect.Parameter.empty
            and param.name not in arguments
        ]
        if missing_required_args:
            raise ToolArgumentsValidationException(
                f"Missing required arguments for tool '{tool_name}': {missing_required_args}"
            )

        for arg_name, arg_value in arguments.items():
            annotation = signature.parameters[arg_name].annotation
            if not self._matches_annotation(arg_value, annotation):
                expected = str(annotation).replace("typing.", "")
                got = type(arg_value).__name__
                raise ToolArgumentsValidationException(
                    f"Invalid type for argument '{arg_name}' in tool '{tool_name}': "
                    f"expected {expected}, got {got}"
                )

        try:
            result = tool.call(arguments)
        except TypeError as exc:
            raise ToolArgumentsValidationException(
                f"Failed to call tool '{tool_name}' with provided arguments: {exc}"
            ) from exc

        if isinstance(result, Response):
            return result

        if isinstance(result, dict):
            status = result.get("status")
            if status in {"ok", "error"}:
                return Response(
                    status=status,
                    message=result.get("message"),
                    error=result.get("error"),
                )
            return Response(
                status="ok",
                message=json.dumps(result, ensure_ascii=False),
                error=None,
            )

        return Response(status="ok", message=str(result), error=None)
    
    def get_tool(self, tool_name: str) -> Tool:
        for tool in self.tools:
            if tool.name == tool_name or tool.func.__name__ == tool_name:
                return tool
        raise ToolNotExistException(f"Tool '{tool_name}' does not exist")


def generate_markdown_tools(functions: list[Callable]) -> str:
    """将 Python 函数列表转换为适合放入 Prompt 的 Markdown 文本。"""

    def annotation_to_str(annotation: Any) -> str:
        if annotation == inspect.Signature.empty or annotation == inspect.Parameter.empty:
            return "Any"
        if hasattr(annotation, "__name__") and str(annotation).startswith("<class '"):
            return annotation.__name__
        return str(annotation).replace("typing.", "")

    markdown_lines = []

    for i, func in enumerate(functions, 1):
        name = func.__name__
        doc = inspect.getdoc(func) or "No description."
        sig = inspect.signature(func)

        # 拆分 Docstring（主描述 / Args / Returns）
        main_desc = doc
        arg_descs = {}
        returns_desc = ""
        if "Args:" in doc:
            parts = doc.split("Args:")
            main_desc = parts[0].strip()
            args_and_more = parts[1]
            args_text = args_and_more.split("Returns:")[0]
            if "Returns:" in args_and_more:
                returns_desc = args_and_more.split("Returns:", 1)[1].strip().split("\n")[0]
            
            # 简单解析每行参数注释
            for line in args_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('-'):
                    arg_name, arg_desc = line.split(':', 1)
                    arg_descs[arg_name.strip()] = arg_desc.strip()
        elif "Returns:" in doc:
            parts = doc.split("Returns:", 1)
            main_desc = parts[0].strip()
            returns_desc = parts[1].strip().split("\n")[0]

        # 写入工具名称和主要描述
        markdown_lines.append(f"### {i}. {name}")
        markdown_lines.append(f"- description: {main_desc}")

        # 解析并写入参数列表
        if not sig.parameters:
            markdown_lines.append("- params: none")
            markdown_lines.append("")
            continue

        markdown_lines.append("- params:")
        for param_name, param in sig.parameters.items():
            # 获取类型名称，处理 Optional/Union 等复杂类型
            annotation = param.annotation
            type_str = annotation_to_str(annotation)
                
            # 判断是否必填
            is_required = param.default == inspect.Parameter.empty
            req_str = "required" if is_required else "optional"
            
            # 获取参数的自然语言描述
            desc = arg_descs.get(param_name, "No description.")
            markdown_lines.append(f"  - {param_name}: {type_str}, {req_str}. {desc}")

        markdown_lines.append("")

    return "\n".join(markdown_lines)



def play_card(card_index: int, target: str | None = None) -> Response:
    """Action tool: play_card.
    
    Args:
        card_index: 0-based index in hand.
        target: Enemy entity_id for target-required cards.
    """
    params: dict[str, Any] = {"card_index": card_index}
    if target:
        params["target"] = target
    return game_env_instance.post(play_card.__name__, params)


def use_potion(slot: int, target: str | None = None) -> Response:
    """Action tool: use_potion.

    Args:
        slot: Potion slot index.
        target: Enemy entity_id for target-required potions.
    """
    params: dict[str, Any] = {"slot": slot}
    if target:
        params["target"] = target
    return game_env_instance.post(use_potion.__name__, params)


def discard_potion(slot: int) -> Response:
    """Action tool: discard_potion.

    Args:
        slot: Potion slot index.
    """
    params: dict[str, Any] = {"slot": slot}
    return game_env_instance.post(discard_potion.__name__, params)


def end_turn() -> Response:
    """Action tool: end_turn."""
    return game_env_instance.post(end_turn.__name__, {})


def combat_select_card(card_index: int) -> Response:
    """Action tool: combat_select_card.

    Args:
        card_index: Index in selectable hand cards.
    """
    params: dict[str, Any] = {"card_index": card_index}
    return game_env_instance.post(combat_select_card.__name__, params)


def combat_confirm_selection() -> Response:
    """Action tool: combat_confirm_selection."""
    return game_env_instance.post(combat_confirm_selection.__name__, {})


def claim_reward(index: int) -> Response:
    """Action tool: claim_reward.

    Args:
        index: 0-based reward index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(claim_reward.__name__, params)


def select_card_reward(card_index: int) -> Response:
    """Action tool: select_card_reward.

    Args:
        card_index: 0-based card reward index.
    """
    params: dict[str, Any] = {"card_index": card_index}
    return game_env_instance.post(select_card_reward.__name__, params)


def skip_card_reward() -> Response:
    """Action tool: skip_card_reward."""
    return game_env_instance.post(skip_card_reward.__name__, {})


def proceed() -> Response:
    """Action tool: proceed.

    Note:
        Works for rewards/rest_site/shop/treasure screens.
        For events, use choose_event_option on the event's Proceed option.
    """
    return game_env_instance.post(proceed.__name__, {})


def choose_event_option(index: int) -> Response:
    """Action tool: choose_event_option.

    Args:
        index: 0-based event option index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(choose_event_option.__name__, params)


def advance_dialogue() -> Response:
    """Action tool: advance_dialogue.

    Polling note:
        Call repeatedly until event.in_dialogue becomes false, then choose an event option.
    """
    return game_env_instance.post(advance_dialogue.__name__, {})


def choose_rest_option(index: int) -> Response:
    """Action tool: choose_rest_option.

    Args:
        index: 0-based rest option index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(choose_rest_option.__name__, params)


def shop_purchase(index: int) -> Response:
    """Action tool: shop_purchase.

    Polling note:
        If state shows shop.error (inventory not ready), retry shortly after re-querying state.

    Args:
        index: 0-based shop item index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(shop_purchase.__name__, params)


def choose_map_node(index: int) -> Response:
    """Action tool: choose_map_node.

    Args:
        index: 0-based map next_options index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(choose_map_node.__name__, params)


def select_card(index: int) -> Response:
    """Action tool: select_card.

    Args:
        index: 0-based card index in card_select overlay.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(select_card.__name__, params)


def confirm_selection() -> Response:
    """Action tool: confirm_selection."""
    return game_env_instance.post(confirm_selection.__name__, {})


def cancel_selection() -> Response:
    """Action tool: cancel_selection."""
    return game_env_instance.post(cancel_selection.__name__, {})


def select_bundle(index: int) -> Response:
    """Action tool: select_bundle.

    Args:
        index: 0-based bundle index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(select_bundle.__name__, params)


def confirm_bundle_selection() -> Response:
    """Action tool: confirm_bundle_selection."""
    return game_env_instance.post(confirm_bundle_selection.__name__, {})


def cancel_bundle_selection() -> Response:
    """Action tool: cancel_bundle_selection."""
    return game_env_instance.post(cancel_bundle_selection.__name__, {})


def select_relic(index: int) -> Response:
    """Action tool: select_relic.

    Args:
        index: 0-based relic index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(select_relic.__name__, params)


def skip_relic_selection() -> Response:
    """Action tool: skip_relic_selection."""
    return game_env_instance.post(skip_relic_selection.__name__, {})


def claim_treasure_relic(index: int) -> Response:
    """Action tool: claim_treasure_relic.

    Polling note:
        Treasure may briefly return "Opening chest..." transitional state.
        Re-query state until relic entries appear, then claim by index.

    Args:
        index: 0-based treasure relic index.
    """
    params: dict[str, Any] = {"index": index}
    return game_env_instance.post(claim_treasure_relic.__name__, params)


def crystal_sphere_set_tool(tool: Literal["big", "small"]) -> Response:
    """Action tool: crystal_sphere_set_tool.

    Args:
        tool: Divination tool, either "big" or "small".
    """
    params: dict[str, Any] = {"tool": tool}
    return game_env_instance.post(crystal_sphere_set_tool.__name__, params)


def crystal_sphere_click_cell(x: int, y: int) -> Response:
    """Action tool: crystal_sphere_click_cell.

    Args:
        x: Grid x-coordinate.
        y: Grid y-coordinate.
    """
    params: dict[str, Any] = {"x": x, "y": y}
    return game_env_instance.post(crystal_sphere_click_cell.__name__, params)


def crystal_sphere_proceed() -> Response:
    """Action tool: crystal_sphere_proceed."""
    return game_env_instance.post(crystal_sphere_proceed.__name__, {})


def _annotation_to_type_name(annotation: Any) -> str:
    if annotation in (inspect.Signature.empty, inspect.Parameter.empty):
        return "Any"
    if hasattr(annotation, "__name__") and str(annotation).startswith("<class '"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _func_to_tool(func: Callable) -> Tool:
    signature = inspect.signature(func)
    args: List[Arg] = [
        Arg(arg_name=param_name, type=_annotation_to_type_name(param.annotation))
        for param_name, param in signature.parameters.items()
    ]
    return Tool(
        state="",
        func=func,
        name=func.__name__,
        args=args,
    )


_TOOL_FUNCTIONS: List[Callable] = [
    play_card,
    use_potion,
    discard_potion,
    end_turn,
    combat_select_card,
    combat_confirm_selection,
    claim_reward,
    select_card_reward,
    skip_card_reward,
    proceed,
    choose_event_option,
    advance_dialogue,
    choose_rest_option,
    shop_purchase,
    choose_map_node,
    select_card,
    confirm_selection,
    cancel_selection,
    select_bundle,
    confirm_bundle_selection,
    cancel_bundle_selection,
    select_relic,
    skip_relic_selection,
    claim_treasure_relic,
    crystal_sphere_set_tool,
    crystal_sphere_click_cell,
    crystal_sphere_proceed,
]


all_tools: List[Tool] = [_func_to_tool(func) for func in _TOOL_FUNCTIONS]