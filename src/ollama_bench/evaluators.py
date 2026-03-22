from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from .models import QualityCheck


@dataclass(frozen=True, slots=True)
class GoldPrompt:
    prompt: str
    keyword_groups: tuple[tuple[str, ...], ...]


GOLD_DATASET = [
    GoldPrompt(
        prompt="Объясни принцип работы квантового компьютера простыми словами на русском языке.",
        keyword_groups=(
            ("кубит", "qubit"),
            ("суперпози", "одновременн", "нескольких состояни"),
            ("измерен", "наблюден"),
            ("квант",),
        ),
    ),
    GoldPrompt(
        prompt="Кратко опиши, чем отличается CPU от GPU, на русском языке.",
        keyword_groups=(
            ("cpu", "процессор"),
            ("gpu", "графическ"),
            ("яд", "core"),
            ("паралл", "массов", "одновременн"),
        ),
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
    "Сгенерируй валидный JSON и верни только один JSON-объект в одну строку, без markdown и пояснений. "
    'Используй ровно эти поля: {"title": string, "bullets": [string, string, string], "score": number, "ok": boolean}.'
)


def evaluate_keywords(
    prompt: str,
    response: str,
    keyword_groups: tuple[tuple[str, ...], ...] | tuple[str, ...],
) -> QualityCheck:
    normalized = response.casefold()
    groups = normalize_keyword_groups(keyword_groups)
    matched = [group[0] for group in groups if any(alias.casefold() in normalized for alias in group)]
    score = len(matched) / len(groups) if groups else 0.0
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


def normalize_keyword_groups(
    keyword_groups: tuple[tuple[str, ...], ...] | tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if not keyword_groups:
        return ()
    first = keyword_groups[0]
    if isinstance(first, str):
        return tuple((item,) for item in keyword_groups)  # type: ignore[arg-type]
    return tuple(tuple(group) for group in keyword_groups)  # type: ignore[arg-type]


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
    candidate = extract_json_candidate(response)
    if candidate is None:
        return False
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


def extract_json_candidate(response: str) -> str | None:
    candidate = response.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
        candidate = re.sub(r"\n?```$", "", candidate)
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = candidate[start : end + 1]
    try:
        json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return snippet
