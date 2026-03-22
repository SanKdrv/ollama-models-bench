# Ollama Models Bench

CLI-инструмент для поочередного скачивания моделей через Ollama, замера базовых метрик и генерации отчетов в `md`, `csv` и `json`.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
ollama-bench --models llama3:8b,mistral --mode cpu --rounds 3 --output md
```

## Сборка в единый файл

Проект совместим с упаковкой через `PyInstaller`, например:

```bash
pyinstaller --onefile -n ollama-bench src/ollama_bench/cli.py
```
