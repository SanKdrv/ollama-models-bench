from ollama_bench.evaluators import evaluate_json, evaluate_keywords, evaluate_rag, summarize_quality


def test_keyword_evaluation_labels_high():
    check = evaluate_keywords(
        "prompt",
        "Кубит использует суперпозицию, а квантовое измерение разрушает состояние.",
        ("кубит", "суперпози", "измерен", "квант"),
    )
    assert check.label == "High"
    assert check.score == 1.0


def test_quality_summary_is_average():
    checks = [
        evaluate_keywords("a", "gpu cpu ядра параллельно", ("gpu", "cpu", "яд", "паралл")),
        evaluate_keywords("b", "ничего релевантного", ("gpu", "cpu")),
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
