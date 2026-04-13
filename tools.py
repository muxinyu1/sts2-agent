"""STS2 agent tool function declarations (declarations only, no implementation).

These functions can be parsed by frameworks such as LangChain / LangGraph
from function signatures and docstrings, and then converted into model-readable
tool descriptions automatically.
"""

from typing import Any, Literal


def get_current_game_state() -> dict[str, Any]:
    """Perception tool: fetch the full current game state.

    This tool must be called before making decisions. It is used to read the
    current screen state, player info, hand cards, enemy intents, etc.

    Returns:
        dict[str, Any]: Game state object.
    """
    raise NotImplementedError("Tool declaration placeholder: not implemented yet")


def execute_combat_action(
    action: Literal[
        "play_card",
        "use_potion",
        "discard_potion",
        "end_turn",
        "combat_select_card",
        "combat_confirm_selection",
    ],
    card_index: int | None = None,
    target: str | None = None,
    slot: int | None = None,
) -> dict[str, Any]:
    """Combat core tool: execute combat-phase actions.

    Use only when state_type is monster, elite, boss, or hand_select.

    Args:
        action: Action type.
            Allowed values: play_card, use_potion, discard_potion, end_turn,
            combat_select_card, combat_confirm_selection。
        card_index: Optional. Card index for play_card / combat_select_card.
        target: Optional. Enemy entity_id when a target is required.
        slot: Optional. Potion slot for use_potion / discard_potion.

    Returns:
        dict[str, Any]: Action execution result.
    """
    raise NotImplementedError("Tool declaration placeholder: not implemented yet")


def execute_map_and_event_action(
    action: Literal[
        "choose_map_node",
        "choose_event_option",
        "advance_dialogue",
        "choose_rest_option",
    ],
    index: int | None = None,
) -> dict[str, Any]:
    """Map and event tool: execute map/event/rest-site actions.

    Use only when state_type is map, event, or rest_site.

    Args:
        action: Action type.
            Allowed values: choose_map_node, choose_event_option,
            advance_dialogue, choose_rest_option。
        index: Optional. Used by actions that require an index parameter.

    Returns:
        dict[str, Any]: Action execution result.
    """
    raise NotImplementedError("Tool declaration placeholder: not implemented yet")


def execute_loot_and_shop_action(
    action: Literal[
        "claim_reward",
        "select_card_reward",
        "skip_card_reward",
        "shop_purchase",
        "proceed",
        "claim_treasure_relic",
    ],
    index: int | None = None,
) -> dict[str, Any]:
    """Loot and shop tool: handle rewards, shop, and treasure actions.

    Use only when state_type is rewards, card_reward, shop, fake_merchant,
    or treasure.

    Args:
        action: Action type.
            Allowed values: claim_reward, select_card_reward, skip_card_reward,
            shop_purchase, proceed, claim_treasure_relic。
        index: Optional. Used by actions that require an index parameter.

    Returns:
        dict[str, Any]: Action execution result.
    """
    raise NotImplementedError("Tool declaration placeholder: not implemented yet")


def execute_overlay_management(
    action: Literal[
        "select_card",
        "confirm_selection",
        "cancel_selection",
        "select_bundle",
        "confirm_bundle_selection",
        "cancel_bundle_selection",
        "select_relic",
        "skip_relic_selection",
    ],
    index: int | None = None,
) -> dict[str, Any]:
    """Overlay management tool: handle overlay selection flows.

    Mainly used for selection overlays such as card_select, bundle_select,
    and relic_select.

    Args:
        action: Action type.
            Allowed values: select_card, confirm_selection, cancel_selection,
            select_bundle, confirm_bundle_selection, cancel_bundle_selection,
            select_relic, skip_relic_selection。
        index: Optional. Used by actions that require an index parameter.

    Returns:
        dict[str, Any]: Action execution result.
    """
    raise NotImplementedError("Tool declaration placeholder: not implemented yet")


def play_crystal_sphere_minigame(
    action: Literal[
        "crystal_sphere_set_tool",
        "crystal_sphere_click_cell",
        "crystal_sphere_proceed",
    ],
    tool: Literal["big", "small"] | None = None,
    x: int | None = None,
    y: int | None = None,
) -> dict[str, Any]:
    """Crystal Sphere minigame tool: operate divination tools and the grid.

    Use only when state_type is crystal_sphere.

    Args:
        action: Action type.
            Allowed values: crystal_sphere_set_tool, crystal_sphere_click_cell,
            crystal_sphere_proceed。
        tool: Optional. Used only for crystal_sphere_set_tool;
            value must be "big" or "small".
        x: Optional. Used only for crystal_sphere_click_cell; grid x coordinate.
        y: Optional. Used only for crystal_sphere_click_cell; grid y coordinate.

    Returns:
        dict[str, Any]: Action execution result.
    """
    raise NotImplementedError("Tool declaration placeholder: not implemented yet")

