from ollama_bench.evaluators import (
    evaluate_json,
    evaluate_keywords,
    evaluate_rag,
    summarize_quality,
    summarize_quality_by_category,
)


def test_keyword_evaluation_labels_high():
    check = evaluate_keywords(
        "prompt",
        "Кубит использует суперпозицию, а квантовое измерение разрушает состояние.",
        ("кубит", "суперпози", "измерен", "квант"),
        category="factual",
    )
    assert check.label == "High"
    assert check.score == 1.0


def test_quality_summary_is_average():
    checks = [
        evaluate_keywords(
            "a",
            "графический процессор и cpu используют ядра и параллельность",
            (("gpu", "графическ"), ("cpu", "процессор"), ("яд",), ("паралл",)),
            category="instruction_following",
        ),
        evaluate_keywords("b", "ничего релевантного", ("gpu", "cpu"), category="factual"),
    ]
    label, score = summarize_quality(checks)
    assert label == "Medium"
    assert 0.4 <= score < 0.75


def test_rag_requires_all_markers():
    assert evaluate_rag("Компания основана в 2019, продукт LimeDesk, офис в Казани.")
    assert not evaluate_rag("Основана в 2019, но остальное не знаю.")


def test_json_validation_accepts_expected_shape():
    assert evaluate_json('{"title":"x","bullets":["a","b"],"score":1.2,"ok":true}')
    assert not evaluate_json('{"title":"x","bullets":"bad","score":1.2,"ok":true}')


def test_json_validation_accepts_code_fence_and_extra_text():
    response = """Вот JSON:

```json
{"title":"x","bullets":["a"],"score":1,"ok":true}
```
"""
    assert evaluate_json(response)


def test_keyword_evaluation_accepts_synonyms():
    check = evaluate_keywords(
        "prompt",
        "Графический процессор хорошо подходит для массовых одновременных вычислений.",
        (("gpu", "графическ"), ("паралл", "массов", "одновременн")),
        category="instruction_following",
    )
    assert check.label == "High"


def test_quality_summary_by_category():
    checks = [
        evaluate_keywords("a", "квант кубит", ("квант", "кубит"), category="factual"),
        evaluate_keywords("b", "1. 2. 3. шаги", (("1.",), ("2.",), ("3.",)), category="formatting"),
        evaluate_keywords("c", "cpu gpu", ("cpu", "gpu"), category="instruction_following"),
    ]
    scores = summarize_quality_by_category(checks)
    assert scores["factual"] == 1.0
    assert scores["formatting"] == 1.0
    assert scores["instruction_following"] == 1.0
