from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .models import QualityCheck


@dataclass(frozen=True, slots=True)
class GoldPrompt:
    prompt: str
    keywords: tuple[str, ...]


GOLD_DATASET = [
    GoldPrompt(
        prompt="Объясни принцип работы квантового компьютера простыми словами на русском языке.",
        keywords=("кубит", "суперпози", "измерен", "квант"),
    ),
    GoldPrompt(
        prompt="Кратко опиши, чем отличается CPU от GPU, на русском языке.",
        keywords=("паралл", "яд", "cpu", "gpu"),
    ),
]

RAG_CONTEXT = (
    "Контекст:\n"
    "Компания Polar Lime основана в 2019 году.\n"
    "Флагманский продукт называется LimeDesk.\n"
    "Штаб-квартира расположена в Казани.\n"
)
RAG_PROMPT = (
    f"{RAG_CONTEXT}\n"
    "Ответь строго по контексту: в каком году основана компания, "
    "как называется продукт и где находится штаб-квартира?"
)
RAG_EXPECTED_KEYWORDS = ("2019", "LimeDesk", "Казани")
JSON_PROMPT = (
    "Сгенерируй валидный JSON без markdown-обертки. "
    'Поля: title:string, bullets:array of strings, score:number, ok:boolean.'
)


def evaluate_keywords(prompt: str, response: str, keywords: tuple[str, ...]) -> QualityCheck:
    normalized = response.casefold()
    matched = [kw for kw in keywords if kw.casefold() in normalized]
    score = len(matched) / len(keywords) if keywords else 0.0
    if score >= 0.75:
        label = "High"
    elif score >= 0.4:
        label = "Medium"
    else:
        label = "Low"
    return QualityCheck(
        prompt=prompt,
        response=response,
        score=score,
        matched_keywords=matched,
        label=label,
    )


def summarize_quality(checks: list[QualityCheck]) -> tuple[str, float]:
    if not checks:
        return ("Low", 0.0)
    score = sum(item.score for item in checks) / len(checks)
    if score >= 0.75:
        label = "High"
    elif score >= 0.4:
        label = "Medium"
    else:
        label = "Low"
    return (label, round(score, 3))


def evaluate_rag(response: str) -> bool:
    normalized = response.casefold()
    return all(keyword.casefold() in normalized for keyword in RAG_EXPECTED_KEYWORDS)


def evaluate_json(response: str) -> bool:
    candidate = response.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
        candidate = re.sub(r"\n?```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("title"), str)
        and isinstance(payload.get("bullets"), list)
        and all(isinstance(item, str) for item in payload.get("bullets", []))
        and isinstance(payload.get("score"), (int, float))
        and isinstance(payload.get("ok"), bool)
    )
