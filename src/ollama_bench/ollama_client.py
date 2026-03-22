from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from typing import Any

import requests


ProgressCallback = Callable[[str], None]


class OllamaError(RuntimeError):
    """Raised when the Ollama API or CLI fails."""


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def healthcheck(self) -> None:
        response = self.session.get(
            f"{self.base_url}/api/tags",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def pull_model(self, model: str, on_progress: ProgressCallback | None = None) -> None:
        self._run_ollama_command(["ollama", "pull", model], on_progress=on_progress)

    def remove_model(self, model: str, on_progress: ProgressCallback | None = None) -> None:
        self._run_ollama_command(["ollama", "rm", model], on_progress=on_progress)

    def show_model(self, model: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/show",
            json={"name": model},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def generate_stream(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": options or {},
            },
            stream=True,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)

    def generate_text(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        chunks = []
        for chunk in self.generate_stream(model=model, prompt=prompt, options=options):
            chunks.append(chunk.get("response", ""))
            if chunk.get("done"):
                break
        return "".join(chunks).strip()

    def _run_ollama_command(
        self,
        command: list[str],
        on_progress: ProgressCallback | None = None,
    ) -> None:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if on_progress is not None:
                on_progress(line.strip())
        return_code = process.wait()
        if return_code != 0:
            raise OllamaError(f"Command failed: {' '.join(command)}")
