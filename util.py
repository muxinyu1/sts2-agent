import json
from datetime import datetime
from pathlib import Path


def _extract_tag_content(content: str, tag: str) -> str | None:
    if not isinstance(content, str):
        return None

    text = content.strip()
    if not text:
        return None

    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag not in text or close_tag not in text:
        return None

    start = text.find(open_tag) + len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return None

    result = text[start:end].strip()
    return result or None


def extract_think_preview(decision: str, max_len: int = 200) -> str:
    think_text = _extract_tag_content(decision, "think")
    if not think_text:
        return ""

    think_line = think_text.replace("\n", " ")
    if len(think_line) > max_len:
        return think_line[:max_len] + "..."
    return think_line


def extract_last_action(decision: str) -> str | None:
    if not isinstance(decision, str):
        return None

    content = decision.strip()
    if not content:
        return None

    tool_payload = _extract_tag_content(content, "tool")
    if tool_payload:
        content = tool_payload

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False)

    return None


def extract_plan(decision: str) -> str | None:
    return _extract_tag_content(decision, "plan")


def extract_summary(decision: str) -> str | None:
    return _extract_tag_content(decision, "summary")


def trajectory_log_path(
    logs_dir: Path,
    run_dir_name: str | None = None,
    floor_index: int | None = None,
    battle_index: int | None = None,
) -> Path:
    if run_dir_name and str(run_dir_name).strip():
        target_dir = logs_dir / str(run_dir_name).strip()
    else:
        target_dir = logs_dir / datetime.now().strftime("%Y-%m-%d")

    if isinstance(floor_index, int) and floor_index >= 0:
        target_dir = target_dir / f"floor_{floor_index}"

    if isinstance(battle_index, int) and battle_index >= 0:
        target_dir = target_dir / f"battle_{battle_index}"

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / "trajectory.jsonl"


def record_trajectory_sample(
    logs_dir: Path,
    *,
    run_dir_name: str | None = None,
    floor_index: int | None = None,
    battle_index: int | None = None,
    step: int,
    system_prompt: str,
    user_prompt: str,
    llm_response: str,
    execution_has_error: bool,
    resolved_previous_error: bool,
    model_summary: str | None,
    recent_state_history: list[str],
) -> Path:
    sample = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "floor_index": floor_index,
        "battle_index": battle_index,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_response": llm_response,
        "execution_has_error": execution_has_error,
        "resolved_previous_error": resolved_previous_error,
        "model_summary": model_summary,
        "recent_state_history": recent_state_history,
    }

    log_path = trajectory_log_path(
        logs_dir,
        run_dir_name=run_dir_name,
        floor_index=floor_index,
        battle_index=battle_index,
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return log_path


def record_battle_replay_reward(
    logs_dir: Path,
    *,
    run_dir_name: str | None = None,
    floor_index: int | None = None,
    battle_index: int | None = None,
    step: int,
    hp_loss: int,
    replay_count: int,
    judge_reason: str | None,
    battle_session_key: str | None,
    save_and_load_ok: bool,
    save_and_load_error: str | None,
    tool_payload: str | None,
) -> Path:
    trajectory_path = trajectory_log_path(
        logs_dir,
        run_dir_name=run_dir_name,
        floor_index=floor_index,
        battle_index=battle_index,
    )
    reward_log_path = trajectory_path.with_name("battle_replay_rewards.jsonl")

    sample = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "floor_index": floor_index,
        "battle_index": battle_index,
        "battle_session_key": battle_session_key,
        "hp_loss": max(int(hp_loss), 0),
        "replay_count": max(int(replay_count), 0),
        "judge_reason": judge_reason,
        "save_and_load_ok": bool(save_and_load_ok),
        "save_and_load_error": save_and_load_error,
        "tool_payload": tool_payload,
    }

    with reward_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return reward_log_path
