# Benchmark Result

- Timestamp: 2026-03-22 19:26:59 UTC
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
| command-r7b-arabic | CPU | 1.59s | 6.49 | RAM 5650.6MB / VRAM n/a | High | 0.97 | 1.00 | 1.00 | 0.90 | Pass | Pass | 16384 | Q4_K_M |  |
| notus | CPU | 1.21s | 7.05 | RAM 4680.3MB / VRAM n/a | High | 0.80 | 0.50 | 1.00 | 0.90 | Fail | Fail | 32768 | Q4_0 |  |
| medllama2 | CPU | 1.16s | 7.58 | RAM 5933.7MB / VRAM n/a | Medium | 0.43 | 0.25 | 0.25 | 0.80 | Pass | Fail | 4096 | Q4_0 |  |
| mistrallite | CPU | 2.12s | 6.91 | RAM 7421.7MB / VRAM n/a | Medium | 0.49 | 0.25 | 0.83 | 0.40 | Pass | Fail | 32768 | Q4_0 |  |
| llama3.1:8b | CPU | 2.01s | 6.54 | RAM 5449.2MB / VRAM n/a | Medium | 0.66 | 0.38 | 0.71 | 0.90 | Pass | Pass | 131072 | Q4_K_M |  |
