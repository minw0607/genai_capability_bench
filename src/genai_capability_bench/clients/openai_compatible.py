"""OpenAI-compatible chat client."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from genai_capability_bench.clients.base import ModelClient, ModelResponse
from genai_capability_bench.core.schemas import ModelSpec

load_dotenv()


class OpenAICompatibleClient(ModelClient):
    """Client for OpenAI, Azure OpenAI, and compatible gateways."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_version = spec.api_version or os.environ.get("OPENAI_API_VERSION", "")

        headers = {}
        header_name = os.environ.get("OPENAI_APIM_HEADER_NAME", "")
        header_value = os.environ.get("OPENAI_APIM_SUBSCRIPTION_KEY", "")
        if header_name and header_value:
            headers[header_name] = header_value

        if api_version:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
                default_headers=headers or None,
            )
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)

    def generate(self, prompt: str, system: str | None = None) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.spec.model,
            "messages": messages,
        }
        if self.spec.max_tokens is not None and self.spec.token_parameter:
            params[self.spec.token_parameter] = self.spec.max_tokens
        if self.spec.temperature is not None:
            params["temperature"] = self.spec.temperature

        start = time.time()
        response = self.client.chat.completions.create(**params)
        latency_ms = (time.time() - start) * 1000
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return ModelResponse(
            text=text.strip(),
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            raw=response,
        )
