"""统计 seed=BASELINE 的对局中各模型的平均工具调用错误率和平均攀爬层数。"""
from __future__ import annotations

import json
import re
from pathlib import Path


LOGS_DIR = Path("logs")
TARGET_MODELS = {"/models/models/Qwen3.5-9B", "qwen-sft"}


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
        floor: int = max_floor_reached(run_dir)

        results[model].append(
            {
                "run": run_dir.name,
                "error_ratio_pct": error_ratio,
                "max_floor": floor,
            }
        )

    return results


def print_report(results: dict[str, list[dict]]) -> None:
    print("=" * 60)
    print(f"{'模型':<35} {'对局数':>5}  {'平均错误率(%)':>13}  {'平均攀爬层数':>12}")
    print("=" * 60)
    for model, runs in results.items():
        if not runs:
            print(f"{model:<35} {'0':>5}  {'N/A':>13}  {'N/A':>12}")
            continue
        avg_err = sum(r["error_ratio_pct"] for r in runs) / len(runs)
        avg_floor = sum(r["max_floor"] for r in runs) / len(runs)
        print(f"{model:<35} {len(runs):>5}  {avg_err:>13.4f}  {avg_floor:>12.2f}")

    print()
    for model, runs in results.items():
        if not runs:
            continue
        print(f"\n--- {model} 明细 ---")
        print(f"  {'run':<45} {'错误率(%)':>10}  {'层数':>6}")
        for r in runs:
            print(f"  {r['run']:<45} {r['error_ratio_pct']:>10.4f}  {r['max_floor']:>6}")


def main() -> None:
    results = collect_stats()
    print_report(results)


if __name__ == "__main__":
    main()
