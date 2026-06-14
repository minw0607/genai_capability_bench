"""Reusable LLM-as-judge utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from genai_capability_bench.clients.base import ModelClient


@dataclass
class JudgeScore:
    score: float
    reason: str
    raw: str


def parse_judge_json(text: str) -> JudgeScore:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        return JudgeScore(score=score, reason=str(data.get("reason", "")), raw=text)
    except Exception:
        return JudgeScore(score=0.5, reason="Judge response could not be parsed.", raw=text)


def judge_with_rubric(
    client: ModelClient,
    task: str,
    answer: str,
    rubric: str,
    reference: str | None = None,
) -> JudgeScore:
    prompt = (
        "Evaluate the model answer using the rubric.\n\n"
        f"Task:\n{task}\n\n"
        f"Reference:\n{reference or 'N/A'}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Rubric:\n{rubric}\n\n"
        'Respond ONLY as JSON: {"score": <float 0.0-1.0>, "reason": "<brief>"}'
    )
    response = client.generate(prompt)
    return parse_judge_json(response.text)

