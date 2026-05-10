import json
from collections.abc import Iterator

import requests


class OllamaClient:
    def __init__(self, ollama_url: str):
        self.ollama_url = ollama_url

    def chat_once(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        num_ctx: int = 2048,
        num_predict: int = 120,
        response_format: str | None = None,
        keep_alive: str | None = None,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }

        if tools:
            payload["tools"] = tools

        if response_format:
            payload["format"] = response_format

        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = requests.post(self.ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    def stream_chat(self, messages: list[dict], model: str) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.35,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
                "num_predict": 120,
            },
        }

        with requests.post(
            self.ollama_url,
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                token = data.get("message", {}).get("content", "")

                if token:
                    yield token

                if data.get("done", False):
                    break
