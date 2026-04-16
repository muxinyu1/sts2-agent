import inspect
import json
import types
from functools import lru_cache
from pathlib import Path

from openai import BaseModel
from typing import Any, Callable, List, Literal, Union, get_args, get_origin
import yaml

from exception import (
    ToolArgumentsValidationException,
    ToolNotExistException,
    ToolResponseFormatException,
)
from game_env import game_env_instance
from network import Response


CARDS_DATA_PATH = Path(__file__).resolve().parent / "sts2_cards_data.yaml"
COUNT_LIKE_KEYS = {"offered", "picked", "wins", "skipped_count", "skipped_wins", "count"}


def _normalize_card_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _card_richness(card: dict[str, Any]) -> int:
    stats = card.get("stats") if isinstance(card, dict) else None
    stats_size = len(stats.keys()) if isinstance(stats, dict) else 0
    return len(card.keys()) + stats_size


def _round_floats(value: Any, decimals: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, list):
        return [_round_floats(v, decimals=decimals) for v in value]
    if isinstance(value, dict):
        return {k: _round_floats(v, decimals=decimals) for k, v in value.items()}
    return value


def _prune_card_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_prune_card_payload(v) for v in value]

    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        for key, raw in value.items():
            key_text = str(key)
            if key_text == "image_url":
                continue
            if key_text == "asc_data":
                continue
            if key_text in COUNT_LIKE_KEYS or key_text.endswith("_count"):
                continue
            pruned[key_text] = _prune_card_payload(raw)
        return pruned

    return value


@lru_cache(maxsize=1)
def _load_cards_data() -> dict[str, Any]:
    if not CARDS_DATA_PATH.exists():
        return {}
    with CARDS_DATA_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=1)
def _build_card_query_index() -> tuple[dict[str, list[dict[str, Any]]], dict[str, float | None]]:
    raw_data = _load_cards_data()
    index: dict[str, list[dict[str, Any]]] = {}

    baseline_numerator: dict[str, float] = {}
    baseline_denominator: dict[str, float] = {}

    for section_name, section_cards in raw_data.items():
        if not isinstance(section_cards, list):
            continue

        section_key = str(section_name).strip().upper()
        if section_key not in {"IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT"}:
            continue

        for card in section_cards:
            if not isinstance(card, dict):
                continue

            normalized_name = _normalize_card_key(str(card.get("name", "")))
            normalized_id = _normalize_card_key(str(card.get("id", "")))
            if not normalized_name and not normalized_id:
                continue

            card_entry = dict(card)
            card_entry["character"] = section_key

            if normalized_name:
                index.setdefault(normalized_name, []).append(card_entry)
            if normalized_id:
                index.setdefault(normalized_id, []).append(card_entry)

            stats = card.get("stats")
            if not isinstance(stats, dict):
                continue

            skipped_count = stats.get("skipped_count")
            skipped_winrate = stats.get("skipped_winrate")
            if isinstance(skipped_count, (int, float)) and isinstance(skipped_winrate, (int, float)) and skipped_count > 0:
                baseline_numerator[section_key] = baseline_numerator.get(section_key, 0.0) + skipped_winrate * float(skipped_count)
                baseline_denominator[section_key] = baseline_denominator.get(section_key, 0.0) + float(skipped_count)

    baseline_winrates: dict[str, float | None] = {}
    for character in ["IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT"]:
        denom = baseline_denominator.get(character, 0.0)
        if denom > 0:
            baseline_winrates[character] = round(baseline_numerator[character] / denom, 4)
        else:
            baseline_winrates[character] = None

    return index, baseline_winrates

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


KNOWN_STATE_TYPES = {
    "menu",
    "unknown",
    "monster",
    "elite",
    "boss",
    "hand_select",
    "rewards",
    "card_reward",
    "map",
    "event",
    "rest_site",
    "shop",
    "fake_merchant",
    "treasure",
    "card_select",
    "bundle_select",
    "relic_select",
    "crystal_sphere",
    "overlay",
}


def extract_state_type(state: dict[str, Any] | None) -> str | None:
    if not isinstance(state, dict):
        return None
    state_type = state.get("state_type")
    if isinstance(state_type, str) and state_type.strip():
        return state_type
    return None


def tool_supports_state(tool: Tool, state_type: str) -> bool:
    raw_state = tool.state.strip() if isinstance(tool.state, str) else ""
    if not raw_state:
        return True
    states = {s.strip() for s in raw_state.split(",") if s.strip()}
    return state_type in states or "*" in states


def is_tool_enabled_by_runtime(
    tool_name: str, state_type: str, state: dict[str, Any]
) -> bool:
    if state_type in {"monster", "elite", "boss"}:
        if tool_name not in {"play_card", "use_potion", "discard_potion", "end_turn"}:
            return False
        battle = state.get("battle")
        return (
            isinstance(battle, dict)
            and battle.get("turn") == "player"
            and battle.get("is_play_phase") is True
        )

    if state_type == "hand_select":
        hand_select = state.get("hand_select")
        if not isinstance(hand_select, dict):
            return False
        if tool_name == "query_cards_info":
            cards = hand_select.get("cards")
            return isinstance(cards, list) and len(cards) > 0
        if tool_name == "combat_select_card":
            cards = hand_select.get("cards")
            return isinstance(cards, list) and len(cards) > 0
        if tool_name == "combat_confirm_selection":
            return hand_select.get("can_confirm") is True
        return False

    if state_type == "rewards":
        rewards = state.get("rewards")
        if not isinstance(rewards, dict):
            return False
        if tool_name == "claim_reward":
            items = rewards.get("items")
            return isinstance(items, list) and len(items) > 0
        if tool_name == "proceed":
            return rewards.get("can_proceed") is True
        return False

    if state_type == "card_reward":
        card_reward = state.get("card_reward")
        if not isinstance(card_reward, dict):
            return False
        if tool_name == "query_cards_info":
            cards = card_reward.get("cards")
            return isinstance(cards, list) and len(cards) > 0
        if tool_name == "select_card_reward":
            cards = card_reward.get("cards")
            return isinstance(cards, list) and len(cards) > 0
        if tool_name == "skip_card_reward":
            return card_reward.get("can_skip") is True
        return False

    if state_type == "map":
        if tool_name != "choose_map_node":
            return False
        map_state = state.get("map")
        if not isinstance(map_state, dict):
            return False
        next_options = map_state.get("next_options")
        return isinstance(next_options, list) and len(next_options) > 0

    if state_type == "event":
        event = state.get("event")
        if not isinstance(event, dict):
            return False
        in_dialogue = event.get("in_dialogue") is True
        if tool_name == "advance_dialogue":
            return in_dialogue
        if tool_name == "choose_event_option":
            options = event.get("options")
            return (not in_dialogue) and isinstance(options, list) and len(options) > 0
        return False

    if state_type == "rest_site":
        rest_site = state.get("rest_site")
        if not isinstance(rest_site, dict):
            return False
        if tool_name == "choose_rest_option":
            options = rest_site.get("options")
            return isinstance(options, list) and len(options) > 0
        if tool_name == "proceed":
            return rest_site.get("can_proceed") is True
        return False

    if state_type == "shop":
        shop = state.get("shop")
        if not isinstance(shop, dict):
            return False
        if tool_name == "shop_purchase":
            items = shop.get("items")
            return isinstance(items, list) and any(
                isinstance(item, dict)
                and item.get("is_stocked", True)
                and item.get("can_afford") is True
                for item in items
            )
        if tool_name == "proceed":
            # Shop can still be leave-able even when serialized can_proceed is false.
            # Keep proceed available to avoid deadlocks when no items are affordable.
            return True
        return False

    if state_type == "fake_merchant":
        fake_merchant = state.get("fake_merchant")
        if not isinstance(fake_merchant, dict):
            return False
        shop = fake_merchant.get("shop")
        if not isinstance(shop, dict):
            return False
        if tool_name == "shop_purchase":
            items = shop.get("items")
            return isinstance(items, list) and any(
                isinstance(item, dict)
                and item.get("is_stocked", True)
                and item.get("can_afford") is True
                for item in items
            )
        if tool_name == "proceed":
            return True
        return False

    if state_type == "treasure":
        treasure = state.get("treasure")
        if not isinstance(treasure, dict):
            return False
        if tool_name == "claim_treasure_relic":
            relics = treasure.get("relics")
            return isinstance(relics, list) and len(relics) > 0
        if tool_name == "proceed":
            return treasure.get("can_proceed") is True
        return False

    if state_type == "card_select":
        card_select = state.get("card_select")
        if not isinstance(card_select, dict):
            return False
        if tool_name == "query_cards_info":
            cards = card_select.get("cards")
            return isinstance(cards, list) and len(cards) > 0
        if tool_name == "select_card":
            cards = card_select.get("cards")
            return isinstance(cards, list) and len(cards) > 0
        if tool_name == "confirm_selection":
            return card_select.get("can_confirm") is True
        if tool_name == "cancel_selection":
            return card_select.get("can_cancel") is True or card_select.get("can_skip") is True
        return False

    if state_type == "bundle_select":
        bundle_select = state.get("bundle_select")
        if not isinstance(bundle_select, dict):
            return False
        if tool_name == "select_bundle":
            bundles = bundle_select.get("bundles")
            return isinstance(bundles, list) and len(bundles) > 0
        if tool_name == "confirm_bundle_selection":
            return bundle_select.get("can_confirm") is True
        if tool_name == "cancel_bundle_selection":
            return bundle_select.get("can_cancel") is True
        return False

    if state_type == "relic_select":
        relic_select = state.get("relic_select")
        if not isinstance(relic_select, dict):
            return False
        if tool_name == "select_relic":
            relics = relic_select.get("relics")
            return isinstance(relics, list) and len(relics) > 0
        if tool_name == "skip_relic_selection":
            return relic_select.get("can_skip") is True
        return False

    if state_type == "crystal_sphere":
        crystal_sphere = state.get("crystal_sphere")
        if not isinstance(crystal_sphere, dict):
            return False
        if tool_name == "crystal_sphere_set_tool":
            return (
                crystal_sphere.get("can_use_big_tool") is True
                or crystal_sphere.get("can_use_small_tool") is True
            )
        if tool_name == "crystal_sphere_click_cell":
            clickable_cells = crystal_sphere.get("clickable_cells")
            return isinstance(clickable_cells, list) and len(clickable_cells) > 0
        if tool_name == "crystal_sphere_proceed":
            return crystal_sphere.get("can_proceed") is True
        return False

    if state_type in {"menu", "unknown", "overlay"}:
        return False

    return True


def optimize_tools_for_state(
    all_tools: List[Tool], state: dict[str, Any] | None
) -> tuple[List[Tool], dict[str, Any]]:
    state_dict = state if isinstance(state, dict) else {}
    state_type = extract_state_type(state_dict)

    if not state_type:
        optimized_tools = all_tools
        details = {
            "state_type": None,
            "selection_reason": "missing_state_type_fallback_all",
            "all_tools": len(all_tools),
            "state_matched": len(all_tools),
            "optimized_tools": len(all_tools),
            "selected_tools": [t.name for t in optimized_tools],
        }
        return optimized_tools, details

    state_matched_tools = [t for t in all_tools if tool_supports_state(t, state_type)]

    if not state_matched_tools and state_type not in KNOWN_STATE_TYPES:
        optimized_tools = all_tools
        selection_reason = "unknown_state_fallback_all"
    else:
        optimized_tools = [
            t
            for t in state_matched_tools
            if is_tool_enabled_by_runtime(t.name, state_type, state_dict)
        ]
        selection_reason = "state_and_runtime_filtered"

    details = {
        "state_type": state_type,
        "selection_reason": selection_reason,
        "all_tools": len(all_tools),
        "state_matched": len(state_matched_tools),
        "optimized_tools": len(optimized_tools),
        "selected_tools": [t.name for t in optimized_tools],
    }
    return optimized_tools, details


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


def query_cards_info(card_names: list[str]) -> Response:
    """Info tool: query_cards_info.

    Args:
        card_names: Card names or ids to query. Supports multiple entries in one call.

    Returns:
        Card details and unresolved names.
    """
    index, baseline_winrates = _build_card_query_index()

    requested = [str(name).strip() for name in card_names if str(name).strip()]
    if not requested:
        return Response(
            status="error",
            error="'card_names' must include at least one non-empty card name or id.",
            message=None,
        )

    cards_result: list[dict[str, Any]] = []
    not_found: list[str] = []

    for raw_name in requested:
        key = _normalize_card_key(raw_name)
        candidates = index.get(key, [])

        if not candidates:
            not_found.append(raw_name)
            continue

        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for candidate in candidates:
            character = str(candidate.get("character", "")).upper()
            card_id = str(candidate.get("id", ""))
            dedup_key = (character, card_id)
            existing = dedup.get(dedup_key)
            if existing is None or _card_richness(candidate) > _card_richness(existing):
                dedup[dedup_key] = candidate

        ranked_matches = sorted(dedup.values(), key=_card_richness, reverse=True)
        best_match = dict(ranked_matches[0])
        cleaned_match = _prune_card_payload(best_match)
        cards_result.append(_round_floats(cleaned_match, decimals=4))

    def _fmt_percent(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "N/A"
        v = float(value)
        if 0 <= v <= 1:
            v = v * 100.0
        return f"{v:.4f}%"

    lines: list[str] = []
    lines.append("# Card Information Report")
    lines.append("")
    lines.append(f"Requested cards: {', '.join(requested)}.")
    lines.append("")

    if not cards_result:
        lines.append("## Cards")
        lines.append("No cards matched the query.")
        lines.append("")
    else:
        lines.append("## Cards")
        lines.append("")
        for idx, card in enumerate(cards_result, 1):
            name = card.get("name", "Unknown")
            raw_stats = card.get("stats")
            stats: dict[str, Any] = raw_stats if isinstance(raw_stats, dict) else {}
            act_data = card.get("act_data") if isinstance(card.get("act_data"), list) else []

            lines.append(f"### {idx}. {name}")
            lines.append(
                f"This is a {card.get('rarity', 'N/A')} {card.get('type', 'N/A')} card"
                "."
            )
            lines.append(f"Energy cost: {card.get('cost', 'N/A')} (upgraded: {card.get('cost_upgraded', 'N/A')}).")

            description = card.get("description")
            if isinstance(description, str) and description.strip():
                lines.append(f"Current effect: {description.strip()}")
            description_upgraded = card.get("description_upgraded")
            if isinstance(description_upgraded, str) and description_upgraded.strip():
                lines.append(f"Upgraded effect: {description_upgraded.strip()}")

            lines.append("")
            lines.append("Win-rate snapshot:")
            lines.append(f"- Card win rate after pick: {_fmt_percent(stats.get('win_rate'))}")
            lines.append(f"- Card pick rate: {_fmt_percent(stats.get('pick_rate'))}")
            lines.append(f"- Skipped win rate: {_fmt_percent(stats.get('skipped_winrate'))}")

            if act_data:
                lines.append("- Pick rate by act:")
                for act_idx in range(3):
                    act_entry = act_data[act_idx] if act_idx < len(act_data) else {}
                    act_pick_rate = (
                        act_entry.get("pick_rate")
                        if isinstance(act_entry, dict)
                        else None
                    )
                    lines.append(f"  - Act {act_idx + 1}: {_fmt_percent(act_pick_rate)}")

            pairings = card.get("pairings")
            if isinstance(pairings, list) and pairings:
                pairing_names = []
                for pairing in pairings[:5]:
                    if isinstance(pairing, dict):
                        pair_name = pairing.get("name")
                        if isinstance(pair_name, str) and pair_name.strip():
                            pairing_names.append(pair_name.strip())
                if pairing_names:
                    lines.append(f"- Frequently paired cards: {', '.join(pairing_names)}")

            lines.append("")

    lines.append("## Not Found")
    if not_found:
        for missing in not_found:
            lines.append(f"- {missing}")
    else:
        lines.append("- none")

    return Response(status="ok", message="\n".join(lines), error=None)



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


def _func_to_tool(func: Callable, state: str) -> Tool:
    signature = inspect.signature(func)
    args: List[Arg] = [
        Arg(arg_name=param_name, type=_annotation_to_type_name(param.annotation))
        for param_name, param in signature.parameters.items()
    ]
    return Tool(
        state=state,
        func=func,
        name=func.__name__,
        args=args,
    )


_TOOL_FUNCTIONS: List[tuple[Callable, str]] = [
    (query_cards_info, "card_reward,card_select,hand_select"),
    (play_card, "monster,elite,boss"),
    (use_potion, "monster,elite,boss"),
    (discard_potion, "monster,elite,boss"),
    (end_turn, "monster,elite,boss"),
    (combat_select_card, "hand_select"),
    (combat_confirm_selection, "hand_select"),
    (claim_reward, "rewards"),
    (select_card_reward, "card_reward"),
    (skip_card_reward, "card_reward"),
    (proceed, "rewards,rest_site,shop,fake_merchant,treasure,monster,elite,boss"),
    (choose_event_option, "event"),
    (advance_dialogue, "event"),
    (choose_rest_option, "rest_site"),
    (shop_purchase, "shop,fake_merchant"),
    (choose_map_node, "map"),
    (select_card, "card_select"),
    (confirm_selection, "card_select"),
    (cancel_selection, "card_select"),
    (select_bundle, "bundle_select"),
    (confirm_bundle_selection, "bundle_select"),
    (cancel_bundle_selection, "bundle_select"),
    (select_relic, "relic_select"),
    (skip_relic_selection, "relic_select"),
    (claim_treasure_relic, "treasure"),
    (crystal_sphere_set_tool, "crystal_sphere"),
    (crystal_sphere_click_cell, "crystal_sphere"),
    (crystal_sphere_proceed, "crystal_sphere"),
]


all_tools: List[Tool] = [_func_to_tool(func, state) for func, state in _TOOL_FUNCTIONS]