from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS = {
    "cpu": [
        "command-r7b-arabic",
        "notus",
        "medllama2",
        "mistrallite",
        "llama3.1:8b",
    ],
    "gpu": ["llama3:8b", "mistral", "phi3", "qwen2"],
}


@dataclass(slots=True)
class BenchmarkConfig:
    models: list[str]
    mode: str
    rounds: int = 3
    output_format: str = "md"
    output_dir: Path = Path(".")
    ollama_base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 300

    @classmethod
    def from_args(
        cls,
        models_arg: str,
        mode: str,
        rounds: int,
        output_format: str,
        output_dir: str,
        ollama_base_url: str,
    ) -> "BenchmarkConfig":
        models = DEFAULT_MODELS[mode] if models_arg == "default_top" else [
            item.strip() for item in models_arg.split(",") if item.strip()
        ]
        if not models:
            raise ValueError("No models were provided")
        return cls(
            models=models,
            mode=mode,
            rounds=rounds,
            output_format=output_format,
            output_dir=Path(output_dir),
            ollama_base_url=ollama_base_url,
        )
