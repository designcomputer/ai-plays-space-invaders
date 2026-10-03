"""Minimal Python client for the Ollama /v1/systemone endpoint.

Supports all three System One question types:
  - noul:    yes/no            -> probability 0-1
  - choice:  labeled           -> label + probabilities + confidence
  - score:   rubric (N levels) -> score on the 0-(N-1) scale

`state` may be a string, or a dict/list (serialized to JSON text per the
API spec). `images` (base64 PNG/JPEG/WebP) is only supported by Clef and
Clef Flash with vision weights.

The server base URL defaults to http://localhost:11434 and can be overridden
with the SYSTEMONE_BASE_URL environment variable or the base_url argument.

Usage:
    from systemone import systemone

    response = systemone(
        model='tev1:4b',
        state='Hello World',
        questions={
            'says_hello': {
                'type': 'noul',
                'instructions': 'Does the state text contain a greeting?',
                'criteria': {
                    'true': 'The state text contains a greeting.',
                    'false': 'The state text does not contain a greeting.',
                },
            },
        },
    )
    print(response.answers['says_hello'].noul)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

DEFAULT_BASE_URL = os.environ.get("SYSTEMONE_BASE_URL", "http://localhost:11434")
DEFAULT_TIMEOUT = 120.0

# Server limits: 64 KiB without images, 32 MiB with images (base64 + JSON).
MAX_BODY_BYTES = 64 * 1024
MAX_BODY_BYTES_WITH_IMAGES = 32 * 1024 * 1024


class SystemOneError(RuntimeError):
    """Raised when the /v1/systemone endpoint returns an error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


@dataclass
class NoulAnswer:
    """Yes/no question: probability that the 'true' criterion holds (0-1)."""

    type: str
    noul: float


@dataclass
class ChoiceAnswer:
    """Labeled classification: winning label, per-label probabilities, confidence."""

    type: str
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class ScoreAnswer:
    """Rubric question: score on the 0-(N-1) scale defined by the criteria list."""

    type: str
    score: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass
class SystemOneResponse:
    model: str
    answers: dict[str, Answer]
    usage: dict[str, int] = field(default_factory=dict)
    escalated: bool = False  # True when systemone_fallback re-ran with the fallback model


def _parse_answer(value: dict[str, Any]) -> Answer:
    qtype = value["type"]
    if qtype == "noul":
        return NoulAnswer(type=qtype, noul=value["noul"])
    if qtype == "choice":
        return ChoiceAnswer(
            type=qtype,
            choice=value["choice"],
            probabilities=value.get("probabilities", {}),
            confidence=value.get("confidence", 0.0),
        )
    if qtype == "score":
        return ScoreAnswer(type=qtype, score=value["score"])
    raise SystemOneError(0, f"unknown answer type: {qtype!r}")


def _normalize_state(state: str | dict | list) -> str:
    """State must be a nonempty string or a JSON-serialized object/array."""
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state must be a nonempty string")
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state)
    raise TypeError("state must be a str, dict, or list")


def systemone(
    model: str,
    state: str | dict | list,
    questions: dict[str, dict[str, Any]],
    *,
    images: list[str] | None = None,
    keep_alive: str | int | float | None = None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> SystemOneResponse:
    """Send a state + questions payload to the /v1/systemone endpoint.

    Args:
        model:      System One model tag, e.g. 'tev1:4b', 'nimble:9b', 'clef:27b'.
        state:      Nonempty string, or a dict/list serialized as JSON text.
        questions:  Named questions; each has 'type' ('noul'|'choice'|'score'),
                    'instructions', and 'criteria'. Answers are not passed to
                    later questions.
        images:     Optional base64-encoded PNG/JPEG/WebP strings, shared by all
                    questions in request order. Clef / Clef Flash only.
        keep_alive: How long to keep the model loaded after the request:
                    duration string ('5m') or seconds. 0 unloads after the
                    request; a negative value keeps it loaded. Defaults to the
                    server setting (usually 5m).
        base_url:   Override the server base URL.
        timeout:    Request timeout in seconds.

    Returns:
        SystemOneResponse with .answers keyed by question name.

    Raises:
        SystemOneError: on non-200 responses (400/404/413/500).
        ValueError:     on an empty state or an oversized request body.
    """
    payload: dict[str, Any] = {
        "model": model,
        "state": _normalize_state(state),
        "questions": questions,
    }
    if images:
        payload["images"] = images
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    body = json.dumps(payload).encode("utf-8")
    limit = MAX_BODY_BYTES_WITH_IMAGES if images else MAX_BODY_BYTES
    if len(body) > limit:
        raise ValueError(
            f"request body is {len(body)} bytes; the server limit is "
            f"{limit} bytes {'with' if images else 'without'} images"
        )

    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/v1/systemone"
    resp = requests.post(url, data=body, headers={"Content-Type": "application/json"}, timeout=timeout)
    if resp.status_code != 200:
        raise SystemOneError(resp.status_code, resp.text.strip())

    data = resp.json()
    answers = {key: _parse_answer(value) for key, value in data.get("answers", {}).items()}
    return SystemOneResponse(
        model=data.get("model", model),
        answers=answers,
        usage=data.get("usage", {}),
    )


def _needs_escalation(
    response: SystemOneResponse,
    confidence_threshold: float,
    noul_gray_band: tuple[float, float],
) -> list[str]:
    """Return the names of answers that warrant a second opinion."""
    lo, hi = noul_gray_band
    uncertain = []
    for name, answer in response.answers.items():
        if isinstance(answer, ChoiceAnswer) and answer.confidence < confidence_threshold:
            uncertain.append(f"{name} (choice confidence {answer.confidence:.3f})")
        elif isinstance(answer, NoulAnswer) and lo <= answer.noul <= hi:
            uncertain.append(f"{name} (noul {answer.noul:.3f} in gray band)")
    return uncertain


def systemone_fallback(
    model: str,
    state: str | dict | list,
    questions: dict[str, dict[str, Any]],
    *,
    fallback_model: str | None = None,
    confidence_threshold: float = 0.6,
    noul_gray_band: tuple[float, float] = (0.3, 0.7),
    client: Callable[..., SystemOneResponse] | None = None,
    **kwargs: Any,
) -> SystemOneResponse:
    """Run systemone with optional confidence-based escalation.

    Fully optional: with fallback_model=None this behaves exactly like
    systemone(). With a fallback_model, the primary model answers first;
    if any choice confidence is below confidence_threshold or any noul
    score falls inside noul_gray_band, the whole request is re-run with
    the fallback model and that response is returned (escalated=True).

    `client` is the transport: any callable with the systemone() signature
    (model, state, questions, **kwargs) that returns a SystemOneResponse.
    Defaults to this module's systemone(). Swap in the official ollama
    client (once it supports System One) without changing the escalation
    policy.

    Extra keyword arguments (images, keep_alive, base_url, timeout) are
    passed through to both calls.
    """
    call = client or systemone
    response = call(model, state, questions, **kwargs)
    if fallback_model is None:
        return response
    uncertain = _needs_escalation(response, confidence_threshold, noul_gray_band)
    if not uncertain:
        return response
    escalated = call(fallback_model, state, questions, **kwargs)
    escalated.escalated = True
    return escalated
