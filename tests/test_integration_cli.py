from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ollama_bench.cli import main


def test_cli_with_http_stub_and_fake_ollama_binary(tmp_path, monkeypatch):
    logs: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._write_json({"models": []})
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            logs.append((self.path, payload.get("prompt", "")))
            if self.path == "/api/show":
                self._write_json({"details": {"quantization_level": "Q4_K_M", "context_length": 4096}})
                return
            if self.path == "/api/generate":
                prompt = payload["prompt"]
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                if "валидный JSON" in prompt:
                    chunks = [
                        {"response": '{"title":"demo","bullets":["x"],"score":1,"ok":true}'},
                        {"response": "", "done": True},
                    ]
                elif "Контекст:" in prompt:
                    chunks = [{"response": "2019 LimeDesk Казани"}, {"response": "", "done": True}]
                elif "квантового компьютера" in prompt:
                    chunks = [{"response": "кубит суперпозиция измерение квант"}, {"response": "", "done": True}]
                elif "CPU от GPU" in prompt:
                    chunks = [{"response": "CPU GPU ядра параллельность"}, {"response": "", "done": True}]
                else:
                    chunks = [{"response": "локальные модели полезны"}, {"response": "", "done": True}]
                for chunk in chunks:
                    self.wfile.write((json.dumps(chunk) + "\n").encode("utf-8"))
                return
            self.send_error(404)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    fake_ollama = tmp_path / "ollama"
    fake_ollama.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ollama.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{Path.cwd()}")

    try:
        exit_code = main(
            [
                "--models",
                "stub-model",
                "--mode",
                "cpu",
                "--rounds",
                "1",
                "--output",
                "md",
                "--output-dir",
                str(tmp_path),
                "--ollama-url",
                f"http://127.0.0.1:{server.server_address[1]}",
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    report_files = list(tmp_path.glob("benchmark_report_*.md"))
    assert exit_code == 0
    assert report_files
    assert any(path == "/api/generate" for path, _ in logs)
