import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class TrajectoryStats:
    path: Path
    json_count: int
    invalid_lines: int


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Count JSON objects in logs/**/trajectory.jsonl files."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=project_root / "logs",
        help="Logs directory to scan (default: ./logs)",
    )
    return parser.parse_args()


def count_json_objects(file_path: Path) -> tuple[int, int]:
    json_count = 0
    invalid_lines = 0

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            try:
                json.loads(line)
                json_count += 1
            except json.JSONDecodeError:
                invalid_lines += 1

    return json_count, invalid_lines


def collect_stats(logs_dir: Path) -> list[TrajectoryStats]:
    stats: list[TrajectoryStats] = []
    for trajectory_file in sorted(logs_dir.rglob("trajectory.jsonl")):
        json_count, invalid_lines = count_json_objects(trajectory_file)
        stats.append(
            TrajectoryStats(
                path=trajectory_file,
                json_count=json_count,
                invalid_lines=invalid_lines,
            )
        )
    return stats


def build_table(stats: list[TrajectoryStats], base_dir: Path) -> Table:
    table = Table(title="Trajectory JSON Count", box=box.ROUNDED)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("File", style="white")
    table.add_column("JSON Count", justify="right", style="green")
    table.add_column("Invalid Lines", justify="right", style="yellow")

    for idx, item in enumerate(stats, start=1):
        try:
            display_path = str(item.path.relative_to(base_dir))
        except ValueError:
            display_path = str(item.path)

        table.add_row(
            str(idx),
            display_path,
            str(item.json_count),
            str(item.invalid_lines),
        )

    return table


def main() -> int:
    console = Console()
    args = parse_args()
    logs_dir = args.logs_dir.expanduser().resolve()

    if not logs_dir.exists() or not logs_dir.is_dir():
        console.print(
            Panel.fit(
                f"Logs directory not found: {logs_dir}",
                title="Count Samples",
                border_style="red",
            )
        )
        return 1

    stats = collect_stats(logs_dir)

    if not stats:
        console.print(
            Panel.fit(
                f"No trajectory.jsonl found under: {logs_dir}",
                title="Count Samples",
                border_style="yellow",
            )
        )
        return 0

    total_files = len(stats)
    total_json = sum(item.json_count for item in stats)
    total_invalid = sum(item.invalid_lines for item in stats)

    console.print(build_table(stats, logs_dir.parent))

    summary_lines = [
        f"Logs dir: {logs_dir}",
        f"trajectory.jsonl files: {total_files}",
        f"total JSON objects: {total_json}",
        f"total invalid lines: {total_invalid}",
    ]
    summary_style = "green" if total_invalid == 0 else "yellow"
    console.print(
        Panel.fit(
            "\n".join(summary_lines),
            title="Summary",
            border_style=summary_style,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())