# Seed

```plain text
XTEH794ZWL
```

# Qwen3.5-397B-A17B

## Baseline

### Exp Config

|Tool Optimization|Query Card Info Tool|Temperature|Top P|
|---|---|---|---|
|False|False|0.25|0.9|

### Result

- Floor: 17
- Tool Call Error Rate: 3.40%

## Ablation Study

### Enable Tool Optimization

- Tool Call Error Rate: 0.63%
- Tokens Reduced: 88.40%

# Qwen 3.5 9B

## Baseline

### No Tool Optimization

游戏开始界面选遗物都选不对，工具太多了，小模型的tool use能力还是有问题，所以启用了工具优化

### Use Tool Optimization

爬到了第七层

```plain text
tool_calls_total: 121
tool_calls_error_total: 7
tool_calls_error_ratio: 5.79%
tool_calls_by_name:
- play_card: 75
- end_turn: 18
- choose_map_node: 6
- claim_reward: 6
- proceed: 5
- select_card_reward: 4
- choose_event_option: 2
- select_card: 2
- use_potion: 2
- unknown_tool: 1

tool_selection_token_optimization:
- steps: 121
- total_before: 113377
- total_after: 15396
- reduced_tokens: 97981
- reduced_ratio: 86.42%

battle_replay:
- replay_enabled: False
- replay_limit_per_floor_battle: 5
- replay_attempt_total: 0
- replay_attempt_error_total: 0
- replay_attempt_error_ratio: 0.00%
- replay_reward_samples: 8
```

## Qwen 3.5 9B Distill

