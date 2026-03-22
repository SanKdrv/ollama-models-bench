from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QualityCheck:
    prompt: str
    response: str
    score: float
    matched_keywords: list[str]
    label: str


@dataclass(slots=True)
class BenchmarkResult:
    model: str
    mode: str
    ttft_seconds: float | None
    tokens_per_second: float | None
    memory_peak_mb: float | None
    vram_peak_mb: float | None
    quality_ru_label: str
    quality_ru_score: float
    rag_passed: bool
    json_match: bool
    context_window: str
    quantization: str
    model_type: str
    rounds: int
    error: str | None = None
    quality_checks: list[QualityCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ram_vram"] = self.ram_vram_display
        return payload

    @property
    def ram_vram_display(self) -> str:
        ram = f"{self.memory_peak_mb:.1f}MB" if self.memory_peak_mb is not None else "n/a"
        vram = f"{self.vram_peak_mb:.1f}MB" if self.vram_peak_mb is not None else "n/a"
        return f"RAM {ram} / VRAM {vram}"
