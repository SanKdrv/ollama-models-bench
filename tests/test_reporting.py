from ollama_bench.models import BenchmarkResult
from ollama_bench.reporting import render_markdown


def test_markdown_report_contains_core_columns():
    result = BenchmarkResult(
        model="llama3",
        mode="cpu",
        ttft_seconds=0.5,
        tokens_per_second=10.0,
        memory_peak_mb=512,
        vram_peak_mb=None,
        quality_ru_label="High",
        quality_ru_score=0.9,
        rag_passed=True,
        json_match=True,
        context_window="8192",
        quantization="Q4_K_M",
        model_type="local/open-source",
        rounds=3,
    )
    rendered = render_markdown([result])
    assert "llama3" in rendered
    assert "Q4_K_M" in rendered
    assert "Pass" in rendered
