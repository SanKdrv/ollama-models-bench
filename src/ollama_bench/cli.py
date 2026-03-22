from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .benchmarking import BenchmarkEngine
from .config import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local LLMs served by Ollama")
    parser.add_argument("--models", default="default_top", help="Comma-separated models or default_top")
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", choices=("md", "csv", "json"), default="md")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig.from_args(
            models_arg=args.models,
            mode=args.mode,
            rounds=args.rounds,
            output_format=args.output,
            output_dir=args.output_dir,
            ollama_base_url=args.ollama_url,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    last_message = {"text": "Starting"}
    with progress:
        task_id = progress.add_task("Preparing benchmark", total=len(config.models))

        def on_progress(message: str) -> None:
            last_message["text"] = message
            completed = sum(
                1 for prefix in ("Cleaning",) if prefix in message
            )
            progress.update(task_id, description=message, completed=progress.tasks[0].completed + completed)

        try:
            engine = BenchmarkEngine(config)
            results, report_path = engine.run(progress_callback=on_progress)
        except Exception as exc:
            console.print(f"[red]Benchmark failed:[/red] {exc}")
            return 1

        progress.update(task_id, completed=len(config.models), description="Benchmark complete")

    console.print(f"Report saved to {report_path}")
    for result in results:
        status = "OK" if not result.error else f"ERROR: {result.error}"
        console.print(
            f"{result.model}: TTFT={result.ttft_seconds or 'n/a'}s, "
            f"TPS={result.tokens_per_second or 'n/a'}, JSON={result.json_match}, {status}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
