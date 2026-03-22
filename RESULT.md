# Benchmark Result

- Timestamp: 2026-03-22 18:13:36 UTC
- Runner: ai-ollama
- Host: ai-ollama
- OS: Linux 6.8.0-101-generic x86_64
- CPU: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz
- CPU cores: 8
- RAM total: 7940MB
- Mode: cpu
- Models: default_top
- Rounds: 3

| Модель | Режим | TTFT (сек) | Speed (т/с) | RAM/VRAM | Качество (RU) | Score | Factual | Instr | Format | RAG | JSON Match | Контекст | Квантизация | Ошибка |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| command-r7b-arabic | CPU | 1.46s | 6.53 | RAM 5984.4MB / VRAM n/a | High | 0.81 | 0.62 | 1.00 | 0.80 | Pass | Pass | 16384 | Q4_K_M |  |
| magicoder | CPU | n/a | n/a | RAM n/a / VRAM n/a | Low | 0.00 | 0.00 | 0.00 | 0.00 | Fail | Fail | unknown | unknown | 500 Server Error: Internal Server Error for url: http://127.0.0.1:11434/api/generate |
| notus | CPU | 1.54s | 6.85 | RAM 4999.3MB / VRAM n/a | High | 0.80 | 0.50 | 1.00 | 0.90 | Fail | Pass | 32768 | Q4_0 |  |
| medllama2 | CPU | 1.00s | 7.96 | RAM 6184.9MB / VRAM n/a | Medium | 0.41 | 0.25 | 0.25 | 0.73 | Fail | Pass | 4096 | Q4_0 |  |
| mistrallite | CPU | 2.28s | 7.37 | RAM 7431.4MB / VRAM n/a | Medium | 0.55 | 0.38 | 0.54 | 0.73 | Pass | Fail | 32768 | Q4_0 |  |
| meta-llama/Llama-3.1-8B | CPU | n/a | n/a | RAM n/a / VRAM n/a | Low | 0.00 | 0.00 | 0.00 | 0.00 | Fail | Fail | unknown | unknown | Command failed: ollama pull meta-llama/Llama-3.1-8B; cleanup failed: Command failed: ollama rm meta-llama/Llama-3.1-8B |
