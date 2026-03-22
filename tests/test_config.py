from ollama_bench.config import BenchmarkConfig, DEFAULT_MODELS


def test_default_cpu_models_match_selected_ollama_search_page():
    assert DEFAULT_MODELS["cpu"] == [
        "command-r7b-arabic",
        "magicoder",
        "notus",
        "medllama2",
        "mistrallite",
        "meta-llama/Llama-3.1-8B",
    ]


def test_from_args_uses_default_top_for_cpu():
    config = BenchmarkConfig.from_args(
        models_arg="default_top",
        mode="cpu",
        rounds=3,
        output_format="md",
        output_dir=".",
        ollama_base_url="http://127.0.0.1:11434",
    )
    assert config.models == DEFAULT_MODELS["cpu"]
