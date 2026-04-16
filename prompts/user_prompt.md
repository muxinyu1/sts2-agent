Current run context:
- act_floor_count: $act_floor_count
- current_turn_index: $turn_index
- floor_turn_count: $floor_turn_count
- floor_summary: $floor_summary
- round_phase: $round_phase
- current_round_plan: $current_round_plan
- last_action: $last_action
- last_error: $last_error

Planning hint for this step:
$plan_instruction

Shortcut rule:
- If current_round_plan is still valid for this step, you may skip <think> and output only <tool>{...}</tool> action JSON.

Recent actions:
$recent_actions_text

Recent state summary history (latest up to 10):
$recent_state_history

Current game state:
$state_text

Pick the best next action now.
