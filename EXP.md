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

### 筛选原则

- 出现错误工具调用去除
- 输出tokens大于8192去除

### 原始数据格式

```json
{"timestamp": "2026-04-23T06:08:49", "step": 484, "floor_index": 14, "battle_index": 2, "system_prompt": "You are an expert Slay the Spire 2 gameplay agent.\nYour job is to choose the single best next action at each step.\nUse only the tools listed below.\n\nAvailable tools:\n### 1. play_card\n- description: Action tool: play_card.\n- params:\n  - card_index: int, required. 0-based index in hand.\n  - target: str | None, optional. Enemy entity_id for target-required cards.\n\n### 2. use_potion\n- description: Action tool: use_potion.\n- params:\n  - slot: int, required. Potion slot index.\n  - target: str | None, optional. Enemy entity_id for target-required potions.\n\n### 3. discard_potion\n- description: Action tool: discard_potion.\n- params:\n  - slot: int, required. Potion slot index.\n\n### 4. end_turn\n- description: Action tool: end_turn.\n- params: none\n\nRules:\n1. Choose exactly one tool call.\n2. The tool name must match an available tool exactly.\n3. Use only valid arguments with correct types.\n4. Do not invent fields or extra keys.\n5. If the previous action failed, correct the cause and choose a new valid action.\n6. Keep reasoning concise and practical.\n7. Fast path: if the current round plan is still valid and no extra reasoning is needed, you may skip thinking text and output only the action JSON inside <tool>...</tool>.\n8. Always output a concise <summary> that includes: (a) key state facts you used, (b) the action you are about to take.\n9. Always output <think_summary> as one short sentence summarizing your thinking result.\n\nOutput format (strict):\n<think>your reasoning</think>  (optional)\n<think_summary>one short sentence summary of your thinking result</think_summary>\n<plan>your concise natural-language plan for current round</plan>  (optional)\n<summary>key state + planned action in one short sentence</summary>\n<tool>{\"name\": \"tool_name\", \"arguments\": {...}}</tool>\n\nDo not output any text outside these tags.", "user_prompt": "Current run context:\n- act_floor_count: 14\n- current_turn_index: 2\n- floor_turn_count: 2\n- floor_summary: No floor summary.\n- round_phase: mid_round\n- current_round_plan: Play Dismantle on Flyconid (Vulnerable) for 24 damage, then use remaining energy for another attack or defense. Priority is eliminating Flyconid while Vulnerable status lasts.\n- last_action: {\"name\": \"play_card\", \"arguments\": {\"card_index\": 1, \"target\": \"SNAPPING_JAXFRUIT_0\"}}\n- last_error: none\n\nPlanning hint for this step:\nContinue this round by following the existing plan when possible. If the plan is still valid, you may skip <think> and output only <tool>{...}</tool>. Do not output a new <plan> unless you need to revise it due to new information.\n\nShortcut rule:\n- If current_round_plan is still valid for this step, you may skip <think> and output only <tool>{...}</tool> action JSON.\n\nRecent actions:\n- Playing 'Dismantle' targeting Flyconid\n- Playing 'Thrash' targeting Snapping Jaxfruit\n\nRecent state summary history (latest up to 10):\n- At card reward after floor 13 combat victory; selecting Inflame (index 1) as it provides Strength scaling that benefits Ironclad's attack-based strategy in future fights.\n- 49/80 HP, 111 gold at rewards screen after floor 13 victory - claiming gold reward (index 0) before proceeding.\n- At rewards screen with all rewards claimed (Inflame card + 11 gold), 49/80 HP and 122 gold - proceeding to next floor.\n- At map screen after floor 13 victory with 49/80 HP and 122 gold - choosing the only available Monster node (index 0) to proceed to floor 14.\n- Floor 14 combat start vs Snapping Jaxfruit (23 HP) and Flyconid (38 HP); playing Inflame (index 4) to gain 2 Strength for damage scaling.\n- Flyconid at 38 HP with 11 damage intent, 2 energy available - playing Bash on Flyconid to apply Vulnerable as planned.\n- 0 energy remaining after Inflame+Bash; Flyconid Vulnerable at 28 HP; ending turn to draw into Turn 2 with Cruelty ready.\n- Flyconid Vulnerable at 28 HP, 3 energy available - playing Dismantle on Flyconid for 24 damage (12x2 due to Vulnerable).\n- Flyconid at ~4 HP after Dismantle, 2 energy remaining - playing Strike+ on Flyconid to eliminate it.\n- Snapping Jaxfruit at 23 HP is the only remaining enemy; playing Thrash for 16 damage to start eliminating it.\n\nCurrent game state:\n# Game State: monster\r\n\r\n**Act 1** | Floor 14 | Ascension 0\r\n\r\n**Round 2** | Turn: player | Play Phase: True\r\n\r\n## Player (You)\r\n**The Ironclad** - HP: 35/80 | Block: 0 | Energy: 1/3 | Gold: 122\r\n\r\n### Status\r\n- **Strength** (2): Increases attack damage by 2.\r\n\r\n### Relics\r\n- **Burning Blood**: At the end of combat, heal 6 HP.\r\n- **Neow's Talisman**: Upon pickup, Upgrade 1 of your Strikes and 1 of your Defends.\r\n- **Festive Popper**: At the start of each combat, deal 9 damage to ALL enemies.\r\n- **Bag of Preparation**: At the start of each combat, draw 2 additional cards.\r\n\r\n### Potions\r\n- [0] **Stable Serum**: Retain your Hand for 2 turns.\r\n\r\n### Hand\r\n- [0] **Defend** (1 energy) [Skill] ✓ - Gain 5 Block. (target: Self)\r\n- [1] **Strike+** (1 energy) [Attack] ✓ - Deal 11 damage. (target: AnyEnemy)\r\n\r\n### Deck Information\r\n\r\n#### Draw Pile (3 cards, sorted by rarity)\r\n- Strike (1): Deal 6 damage.\r\n- Strike (1): Deal 6 damage.\r\n- Strike (1): Deal 6 damage.\r\n\r\n#### Discard Pile (8 cards)\r\n- Bash (2): Deal 10 damage. Apply 2 Vulnerable.\r\n- Defend+ (1): Gain 8 Block.\r\n- Defend (1): Gain 5 Block.\r\n- Defend (1): Gain 5 Block.\r\n- Headbutt (1): Deal 9 damage. Put a card from your Discard Pile on top of your Draw Pile.\r\n- Cruelty (1): Vulnerable enemies take an additional 25% damage.\r\n- Dismantle (1): Deal 15 damage. If the enemy is Vulnerable, hits twice.\r\n- Thrash (1): Deal 15 damage twice. Exhaust a random Attack in your Hand and add its damage to this card.\r\n\r\n#### Exhaust Pile (1 cards)\r\n- Pommel Strike (1): Deal 9 damage. Draw 1 card.\r\n\r\n## Enemies\r\n### Snapping Jaxfruit (`SNAPPING_JAXFRUIT_0`)\r\nHP: 11/32 | Block: 0\r\n**Intent:** Aggressive (Attack) 5 - This enemy intends to Attack for 5 damage., Empower (Buff) - This enemy intends to use a Buff.\r\n### Status\r\n  - **Strength** (2): Increases attack damage by 2.\r\n\r\n\r\n## Keyword Glossary\r\n- **Block**: Until next turn, prevents damage.\r\n- **Retain**: Retained cards are not discarded at the end of turn.\n\nPick the best next action now.", "llm_response": "<think>Current state:\n- Flyconid has been eliminated (not shown in enemies list, only Snapping Jaxfruit remains)\n- Snapping Jaxfruit at 11 HP with 5 damage intent\n- I have 1 energy remaining, 35/80 HP, 0 Block\n- Hand has Defend (5 Block) and Strike+ (11 damage)\n- Strength is 2, so Strike+ deals 13 damage (11+2)\n\nThe plan was to eliminate Flyconid first (done), then focus on Snapping Jaxfruit. Snapping Jaxfruit is at 11 HP, and my Strike+ with Strength 2 will deal 13 damage, which is enough to kill it.\n\nI should play Strike+ on Snapping Jaxfruit to finish the combat. This will end the fight and I'll heal 6 HP from Burning Blood.\n\nNo need for Defend since the enemy only does 5 damage and I'll win this turn.\n</think>\n\n\n<think_summary>Snapping Jaxfruit at 11 HP can be eliminated with Strike+ (11+2 Strength = 13 damage) to end combat.</think_summary>\n<summary>Snapping Jaxfruit at 11 HP, 1 energy remaining - playing Strike+ for 13 damage to finish the enemy and end combat.</summary>\n<tool>{\"name\": \"play_card\", \"arguments\": {\"card_index\": 1, \"target\": \"SNAPPING_JAXFRUIT_0\"}}</tool>", "execution_has_error": false, "resolved_previous_error": false, "model_summary": "Snapping Jaxfruit at 11 HP, 1 energy remaining - playing Strike+ for 13 damage to finish the enemy and end combat.", "recent_state_history": ["49/80 HP, 111 gold at rewards screen after floor 13 victory - claiming gold reward (index 0) before proceeding.", "At rewards screen with all rewards claimed (Inflame card + 11 gold), 49/80 HP and 122 gold - proceeding to next floor.", "At map screen after floor 13 victory with 49/80 HP and 122 gold - choosing the only available Monster node (index 0) to proceed to floor 14.", "Floor 14 combat start vs Snapping Jaxfruit (23 HP) and Flyconid (38 HP); playing Inflame (index 4) to gain 2 Strength for damage scaling.", "Flyconid at 38 HP with 11 damage intent, 2 energy available - playing Bash on Flyconid to apply Vulnerable as planned.", "0 energy remaining after Inflame+Bash; Flyconid Vulnerable at 28 HP; ending turn to draw into Turn 2 with Cruelty ready.", "Flyconid Vulnerable at 28 HP, 3 energy available - playing Dismantle on Flyconid for 24 damage (12x2 due to Vulnerable).", "Flyconid at ~4 HP after Dismantle, 2 energy remaining - playing Strike+ on Flyconid to eliminate it.", "Snapping Jaxfruit at 23 HP is the only remaining enemy; playing Thrash for 16 damage to start eliminating it.", "Snapping Jaxfruit at 11 HP, 1 energy remaining - playing Strike+ for 13 damage to finish the enemy and end combat."]}
```

### 训练策略

同一战斗多次rollout之间，用 mean hp loss - hp_loss 做组内相对优势，再用 softmax 转成 SFT 损失加权 —— 在同战斗内对"打得更好"的轨迹上加权学习，弱化失败轨迹。非战斗样本默认权重 1.0 全量学习。

### loss

<img width="414" height="379" alt="image" src="https://github.com/user-attachments/assets/82a2efaf-a823-4bad-94d5-8c1e81295a9b" />

### Result


| 模型 | 对局数 | 平均工具调用错误率 | 平均攀爬层数 |
|---|---|---|---|
| `Qwen3.5-9B` | 6 | **8.37%** | **14.67** |
| `Qwen3.5-9B Distill` | 5 | **0.82%** | **17.20** |

> 对局种子相同，重复5次

## Qwen 3.5 9B DPO
