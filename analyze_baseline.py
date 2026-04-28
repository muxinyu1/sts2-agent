"""统计 seed=BASELINE 的对局中各模型的平均工具调用错误率和平均攀爬层数。"""
from __future__ import annotations

import json
import re
from pathlib import Path


LOGS_DIR = Path("logs")
TARGET_MODELS = {"/models/models/Qwen3.5-9B", "qwen-sft", "/models/models/muxinyu/sts2-agent/saves/Qwen3.5-9B/DPO"}
MODEL_COL_WIDTH = 35


def format_model_name_from_end(model: str, width: int = MODEL_COL_WIDTH) -> str:
    """模型名过长时从后往前保留，前缀用 ... 省略。"""
    if len(model) <= width:
        return model
    if width <= 3:
        return "." * width
    return "..." + model[-(width - 3):]


def average_and_variance(values: list[float]) -> tuple[float, float]:
    """返回总体均值和总体方差（除以 N）。"""
    if not values:
        return 0.0, 0.0

    avg = sum(values) / len(values)
    var = sum((v - avg) ** 2 for v in values) / len(values)
    return avg, var


def max_floor_reached(run_dir: Path) -> int:
    """返回 run 目录下最大的 floor_N 目录编号，找不到返回 0。"""
    max_floor = 0
    for d in run_dir.iterdir():
        if d.is_dir():
            m = re.fullmatch(r"floor_(\d+)", d.name)
            if m:
                max_floor = max(max_floor, int(m.group(1)))
    return max_floor


def collect_stats() -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {m: [] for m in TARGET_MODELS}

    for run_dir in sorted(LOGS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        # 只处理 seed=BASELINE 的 run
        if "_seed-BASELINE" not in run_dir.name:
            continue

        stats_file = run_dir / "run_statistics.json"
        if not stats_file.exists():
            print(f"[SKIP] 缺少 run_statistics.json: {run_dir.name}")
            continue

        with stats_file.open(encoding="utf-8") as f:
            data = json.load(f)

        model: str = data.get("config", {}).get("llm", {}).get("model", "")
        if model not in TARGET_MODELS:
            print(f"[SKIP] 未知模型 '{model}': {run_dir.name}")
            continue

        error_ratio: float = data.get("tool_calls", {}).get("error_ratio_pct", 0.0)
        model_call_total: int = data.get("tool_calls", {}).get("total", 0)
        floor: int = max_floor_reached(run_dir)

        results[model].append(
            {
                "run": run_dir.name,
                "error_ratio_pct": error_ratio,
                "model_call_total": model_call_total,
                "max_floor": floor,
            }
        )

    return results


def print_report(results: dict[str, list[dict]]) -> None:
    separator_len = MODEL_COL_WIDTH + 80
    print("=" * separator_len)
    print(
        f"{'模型':<{MODEL_COL_WIDTH}} {'对局数':>5}  {'平均错误率(%)':>13}  {'错误率方差':>12}  "
        f"{'平均攀爬层数':>12}  {'层数方差':>10}  {'平均模型调用次数':>16}"
    )
    print("=" * separator_len)
    for model, runs in results.items():
        model_display = format_model_name_from_end(model, MODEL_COL_WIDTH)
        if not runs:
            print(
                f"{model_display:<{MODEL_COL_WIDTH}} {'0':>5}  {'N/A':>13}  {'N/A':>12}  "
                f"{'N/A':>12}  {'N/A':>10}  {'N/A':>16}"
            )
            continue

        error_values = [float(r["error_ratio_pct"]) for r in runs]
        floor_values = [float(r["max_floor"]) for r in runs]
        model_call_values = [float(r["model_call_total"]) for r in runs]
        avg_err, var_err = average_and_variance(error_values)
        avg_floor, var_floor = average_and_variance(floor_values)
        avg_model_calls = sum(model_call_values) / len(model_call_values)
        print(
            f"{model_display:<{MODEL_COL_WIDTH}} {len(runs):>5}  {avg_err:>13.4f}  {var_err:>12.4f}  "
            f"{avg_floor:>12.2f}  {var_floor:>10.4f}  {avg_model_calls:>16.2f}"
        )

    print()
    for model, runs in results.items():
        if not runs:
            continue
        print(f"\n--- {model} 明细 ---")
        print(f"  {'run':<45} {'错误率(%)':>10}  {'层数':>6}  {'模型调用次数':>12}")
        for r in runs:
            print(
                f"  {r['run']:<45} {r['error_ratio_pct']:>10.4f}  {r['max_floor']:>6}  "
                f"{r['model_call_total']:>12}"
            )


def main() -> None:
    results = collect_stats()
    print_report(results)


if __name__ == "__main__":
    main()
