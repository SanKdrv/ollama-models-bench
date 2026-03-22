from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .models import BenchmarkResult


def write_report(results: list[BenchmarkResult], output_dir: Path, output_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_report_{timestamp}.{output_format}"
    if output_format == "md":
        path.write_text(render_markdown(results), encoding="utf-8")
    elif output_format == "csv":
        write_csv(results, path)
    elif output_format == "json":
        path.write_text(
            json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return path


def render_markdown(results: list[BenchmarkResult]) -> str:
    headers = [
        "Модель",
        "Режим",
        "TTFT (сек)",
        "Speed (т/с)",
        "RAM/VRAM",
        "Качество (RU)",
        "Score",
        "Factual",
        "Instr",
        "Format",
        "RAG",
        "JSON Match",
        "Контекст",
        "Квантизация",
        "Ошибка",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.model,
                    result.mode.upper(),
                    format_metric(result.ttft_seconds, suffix="s"),
                    format_metric(result.tokens_per_second),
                    result.ram_vram_display,
                    result.quality_ru_label,
                    format_metric(result.quality_ru_score),
                    format_metric(result.factual_score),
                    format_metric(result.instruction_following_score),
                    format_metric(result.formatting_score),
                    "Pass" if result.rag_passed else "Fail",
                    "Pass" if result.json_match else "Fail",
                    result.context_window,
                    result.quantization,
                    result.error or "",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    headers = [
        "model",
        "mode",
        "ttft_seconds",
        "tokens_per_second",
        "memory_peak_mb",
        "vram_peak_mb",
        "quality_ru_label",
        "quality_ru_score",
        "factual_score",
        "instruction_following_score",
        "formatting_score",
        "rag_passed",
        "json_match",
        "context_window",
        "quantization",
        "model_type",
        "rounds",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.to_dict().get(key) for key in headers})


def format_metric(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"
