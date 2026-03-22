from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass

import psutil

from .config import BenchmarkConfig
from .evaluators import (
    GOLD_DATASET,
    JSON_PROMPT,
    RAG_PROMPT,
    evaluate_json,
    evaluate_keywords,
    evaluate_rag,
    summarize_quality,
    summarize_quality_by_category,
)
from .models import BenchmarkResult, QualityCheck
from .ollama_client import OllamaClient, OllamaError
from .reporting import write_report


@dataclass(slots=True)
class ResourceSnapshot:
    memory_peak_mb: float | None
    vram_peak_mb: float | None


class BenchmarkEngine:
    def __init__(self, config: BenchmarkConfig, client: OllamaClient | None = None) -> None:
        self.config = config
        self.client = client or OllamaClient(
            base_url=config.ollama_base_url,
            timeout_seconds=config.timeout_seconds,
        )
    def run(self, progress_callback=None) -> tuple[list[BenchmarkResult], str]:
        self._validate_environment()
        self.client.healthcheck()
        results: list[BenchmarkResult] = []
        total_models = len(self.config.models)
        for index, model in enumerate(self.config.models, start=1):
            notify(progress_callback, f"[{index}/{total_models}] Downloading {model}")
            try:
                self.client.pull_model(model, on_progress=lambda line: notify(progress_callback, line))
                notify(progress_callback, f"[{index}/{total_models}] Benchmarking {model}")
                result = self._benchmark_model(model)
            except Exception as exc:
                result = BenchmarkResult(
                    model=model,
                    mode=self.config.mode,
                    ttft_seconds=None,
                    tokens_per_second=None,
                    memory_peak_mb=None,
                    vram_peak_mb=None,
                    quality_ru_label="Low",
                    quality_ru_score=0.0,
                    factual_score=0.0,
                    instruction_following_score=0.0,
                    formatting_score=0.0,
                    rag_passed=False,
                    json_match=False,
                    context_window="unknown",
                    quantization="unknown",
                    model_type="local/open-source",
                    rounds=self.config.rounds,
                    error=str(exc),
                )
            finally:
                notify(progress_callback, f"[{index}/{total_models}] Cleaning {model}")
                try:
                    self.client.remove_model(model, on_progress=lambda line: notify(progress_callback, line))
                except Exception as cleanup_error:
                    if result.error:
                        result.error = f"{result.error}; cleanup failed: {cleanup_error}"
                    else:
                        result.error = f"cleanup failed: {cleanup_error}"
            results.append(result)
        report_path = write_report(results, self.config.output_dir, self.config.output_format)
        return (results, str(report_path))

    def _benchmark_model(self, model: str) -> BenchmarkResult:
        metadata = self.client.show_model(model)
        ttfts: list[float] = []
        tps_values: list[float] = []
        resource_peaks: list[ResourceSnapshot] = []
        for _ in range(self.config.rounds):
            ttft, tps, snapshot = self._measure_generation_metrics(
                model,
                "Назови три преимущества локальных LLM на русском языке.",
            )
            ttfts.append(ttft)
            tps_values.append(tps)
            resource_peaks.append(snapshot)

        quality_checks: list[QualityCheck] = []
        for item in GOLD_DATASET:
            response = self.client.generate_text(model, item.prompt)
            quality_checks.append(
                evaluate_keywords(
                    item.prompt,
                    response,
                    item.keyword_groups,
                    category=item.category,
                )
            )
        quality_label, quality_score = summarize_quality(quality_checks)
        category_scores = summarize_quality_by_category(quality_checks)

        rag_response = self.client.generate_text(model, RAG_PROMPT)
        json_response = self.client.generate_text(model, JSON_PROMPT)
        peak_memory = max(
            (snapshot.memory_peak_mb for snapshot in resource_peaks if snapshot.memory_peak_mb is not None),
            default=None,
        )
        peak_vram = max(
            (snapshot.vram_peak_mb for snapshot in resource_peaks if snapshot.vram_peak_mb is not None),
            default=None,
        )
        return BenchmarkResult(
            model=model,
            mode=self.config.mode,
            ttft_seconds=round(sum(ttfts) / len(ttfts), 4),
            tokens_per_second=round(sum(tps_values) / len(tps_values), 3),
            memory_peak_mb=peak_memory,
            vram_peak_mb=peak_vram,
            quality_ru_label=quality_label,
            quality_ru_score=quality_score,
            factual_score=category_scores.get("factual", 0.0),
            instruction_following_score=category_scores.get("instruction_following", 0.0),
            formatting_score=category_scores.get("formatting", 0.0),
            rag_passed=evaluate_rag(rag_response),
            json_match=evaluate_json(json_response),
            context_window=extract_context_window(metadata),
            quantization=extract_quantization(metadata),
            model_type="local/open-source",
            rounds=self.config.rounds,
            quality_checks=quality_checks,
        )

    def _measure_generation_metrics(self, model: str, prompt: str) -> tuple[float, float, ResourceSnapshot]:
        start = time.perf_counter()
        first_token_at: float | None = None
        token_count = 0
        peak_memory = self._memory_usage_mb()
        peak_vram = self._gpu_memory_mb()
        for chunk in self.client.generate_stream(model, prompt):
            now = time.perf_counter()
            if first_token_at is None and chunk.get("response"):
                first_token_at = now
            token_count += estimate_token_count(chunk.get("response", ""))
            peak_memory = max_optional(peak_memory, self._memory_usage_mb())
            gpu_usage = self._gpu_memory_mb()
            if gpu_usage is not None:
                peak_vram = max(peak_vram or 0.0, gpu_usage)
            if chunk.get("done"):
                break
        end = time.perf_counter()
        if first_token_at is None:
            raise OllamaError("Model did not return any tokens")
        total_duration = end - start
        generation_duration = max(end - first_token_at, 1e-9)
        ttft = first_token_at - start
        tps = token_count / generation_duration if token_count else 0.0
        return (ttft, tps, ResourceSnapshot(peak_memory, peak_vram))

    def _validate_environment(self) -> None:
        if self.config.mode == "gpu" and not self._gpu_available():
            raise OllamaError(
                "GPU mode requested, but no compatible GPU telemetry backend was found "
                "(expected nvidia-smi or rocm-smi)."
            )

    def _memory_usage_mb(self) -> float | None:
        usages = []
        for process in self._ollama_processes():
            try:
                usages.append(process.memory_info().rss / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return sum(usages) if usages else None

    def _ollama_processes(self) -> list[psutil.Process]:
        processes: list[psutil.Process] = []
        for process in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (process.info.get("name") or "").casefold()
                cmdline = " ".join(process.info.get("cmdline") or []).casefold()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if "ollama" in name or "ollama" in cmdline:
                processes.append(process)
        return processes

    def _gpu_available(self) -> bool:
        return shutil.which("nvidia-smi") is not None or shutil.which("rocm-smi") is not None

    def _gpu_memory_mb(self) -> float | None:
        if self.config.mode != "gpu":
            return None
        if shutil.which("nvidia-smi") is not None:
            return self._read_nvidia_memory_mb()
        if shutil.which("rocm-smi") is not None:
            return self._read_rocm_memory_mb()
        return None

    def _read_nvidia_memory_mb(self) -> float | None:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        values = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
        return max(values) if values else None

    def _read_rocm_memory_mb(self) -> float | None:
        completed = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return None
        values = []
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            used = value.get("VRAM Total Used Memory (B)") or value.get("vram_used")
            if used is None:
                continue
            try:
                values.append(float(used) / (1024 * 1024))
            except (TypeError, ValueError):
                continue
        return max(values) if values else None


def extract_quantization(metadata: dict) -> str:
    details = metadata.get("details", {})
    return details.get("quantization_level", "unknown")


def extract_context_window(metadata: dict) -> str:
    direct_keys = (
        "context_length",
        "num_ctx",
        "context_window",
        "max_context_length",
        "n_ctx_train",
    )
    for key in direct_keys:
        value = metadata.get(key)
        if value:
            return str(value)

    for parent_key in ("details", "model_info", "parameters", "info"):
        value = metadata.get(parent_key)
        found = find_context_value(value)
        if found is not None:
            return found

    found = find_context_value(metadata)
    if found is not None:
        return found
    return "unknown"


def find_context_value(value: object) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_normalized = str(key).casefold()
            if "context" in key_normalized or "num_ctx" in key_normalized or "n_ctx" in key_normalized:
                normalized = normalize_context_value(nested)
                if normalized is not None:
                    return normalized
            found = find_context_value(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_context_value(item)
            if found is not None:
                return found
    return None


def normalize_context_value(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    if isinstance(value, str):
        digits = "".join(char for char in value if char.isdigit())
        if digits:
            return digits
        stripped = value.strip()
        return stripped or None
    return None


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split())) if text.strip() else 0


def max_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def notify(callback, message: str) -> None:
    if callback is not None:
        callback(message)
