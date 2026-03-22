from __future__ import annotations

from collections.abc import Iterable

import pytest

from ollama_bench.benchmarking import BenchmarkEngine, extract_context_window, extract_quantization
from ollama_bench.config import BenchmarkConfig
from ollama_bench.ollama_client import OllamaError


class FakeClient:
    def __init__(self) -> None:
        self.events: list[str] = []

    def healthcheck(self) -> None:
        self.events.append("healthcheck")

    def pull_model(self, model: str, on_progress=None) -> None:
        self.events.append(f"pull:{model}")
        if on_progress:
            on_progress("pulling")

    def remove_model(self, model: str, on_progress=None) -> None:
        self.events.append(f"rm:{model}")
        if on_progress:
            on_progress("removed")

    def show_model(self, model: str) -> dict:
        return {
            "details": {
                "quantization_level": "Q4_K_M",
                "context_length": 8192,
            }
        }

    def generate_stream(self, model: str, prompt: str, options=None) -> Iterable[dict]:
        if "ровно эти поля" in prompt:
            yield {"response": '{"title":"x","bullets":["a"],"score":1,"ok":true}'}
            yield {"response": "", "done": True}
            return
        if "Контекст:" in prompt:
            yield {"response": "2019 LimeDesk Казани"}
            yield {"response": "", "done": True}
            return
        if "квантового компьютера" in prompt:
            yield {"response": "Кубит суперпозиция квант измерен"}
            yield {"response": "", "done": True}
            return
        if "зачем в RAG нужен внешний контекст" in prompt:
            yield {"response": "Внешний контекст и поиск по источникам снижают галлюцинации, потому что ответ опирается на документы."}
            yield {"response": "", "done": True}
            return
        if "CPU от GPU" in prompt:
            yield {"response": "CPU GPU яд паралл"}
            yield {"response": "", "done": True}
            return
        if "ровно двумя короткими пунктами" in prompt:
            yield {"response": "- приватность\n- офлайн"}
            yield {"response": "", "done": True}
            return
        if "Верни на русском JSON без markdown" in prompt:
            yield {"response": '{"summary":"ok","risks":["a","b"],"ready":true}'}
            yield {"response": "", "done": True}
            return
        if "Ответь на русском в формате" in prompt:
            yield {"response": "Заголовок\nШаги:\n1. Первый\n2. Второй\n3. Третий"}
            yield {"response": "", "done": True}
            return
        yield {"response": "локальные llm быстрее"}
        yield {"response": "", "done": True}

    def generate_text(self, model: str, prompt: str, options=None) -> str:
        parts = []
        for chunk in self.generate_stream(model, prompt, options):
            parts.append(chunk.get("response", ""))
            if chunk.get("done"):
                break
        return "".join(parts)


def test_extract_helpers():
    metadata = {"details": {"quantization_level": "Q8_0"}, "model_info": {"general.context_length": 32768}}
    assert extract_quantization(metadata) == "Q8_0"
    assert extract_context_window(metadata) == "32768"


def test_context_window_does_not_fallback_to_parameter_size():
    metadata = {"details": {"parameter_size": "8.0B"}}
    assert extract_context_window(metadata) == "unknown"


def test_context_window_is_found_in_nested_ollama_metadata():
    metadata = {
        "model_info": {
            "general.architecture": "gemma2",
            "tokenizer": {"ggml": {"context_length": "8192 tokens"}},
        }
    }
    assert extract_context_window(metadata) == "8192"


def test_engine_runs_full_cycle(tmp_path):
    config = BenchmarkConfig(
        models=["phi3"],
        mode="cpu",
        rounds=2,
        output_format="json",
        output_dir=tmp_path,
    )
    client = FakeClient()
    engine = BenchmarkEngine(config=config, client=client)
    results, report_path = engine.run()

    assert len(results) == 1
    result = results[0]
    assert result.model == "phi3"
    assert result.error is None
    assert result.json_match is True
    assert result.rag_passed is True
    assert result.quality_ru_label == "High"
    assert result.quality_ru_score == 1.0
    assert result.factual_score == 1.0
    assert result.instruction_following_score == 1.0
    assert result.formatting_score == 1.0
    assert "pull:phi3" in client.events
    assert "rm:phi3" in client.events
    assert report_path.endswith(".json")


def test_gpu_mode_requires_available_gpu(tmp_path, monkeypatch):
    config = BenchmarkConfig(
        models=["phi3"],
        mode="gpu",
        rounds=1,
        output_format="json",
        output_dir=tmp_path,
    )
    engine = BenchmarkEngine(config=config, client=FakeClient())
    monkeypatch.setattr(engine, "_gpu_available", lambda: False)

    with pytest.raises(OllamaError, match="GPU mode requested"):
        engine.run()


def test_memory_usage_reads_ollama_processes(tmp_path, monkeypatch):
    class FakeMemInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self, name: str, rss_mb: float) -> None:
            self.info = {"name": name, "cmdline": [name]}
            self._rss = int(rss_mb * 1024 * 1024)

        def memory_info(self):
            return FakeMemInfo(self._rss)

    config = BenchmarkConfig(
        models=["phi3"],
        mode="cpu",
        rounds=1,
        output_format="json",
        output_dir=tmp_path,
    )
    engine = BenchmarkEngine(config=config, client=FakeClient())
    monkeypatch.setattr(
        "ollama_bench.benchmarking.psutil.process_iter",
        lambda attrs: [
            FakeProcess("ollama", 100),
            FakeProcess("ollama serve", 50),
            FakeProcess("python", 999),
        ],
    )

    assert engine._memory_usage_mb() == 150
