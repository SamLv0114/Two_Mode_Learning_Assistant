"""
One path for every model call whose reply the code reads as data.

    Pydantic schema
      → strict JSON Schema response format (the model can only produce that shape)
      → parse and validate (types, enums, and the checks a schema cannot express)
      → on a malformed, cut-off or invalid reply: one repair turn showing the model its error
      → StructuredOutputError, which each caller turns into its own deterministic fallback

Every outcome is counted in structured_output_total{task, outcome}. A prompt or model
change that makes replies fail more often then shows up as a rising "repaired" or
"failed" rate, instead of only as quietly worse fallback answers.

Range limits (a score in 0..1) belong in Pydantic validators, not in Field(ge=...):
validators run after parsing, while unsupported schema keywords would make the API
reject the request.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai.lib._pydantic import to_strict_json_schema
from prometheus_client import Counter
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

STRUCTURED_OUTPUTS = Counter("structured_output_total", "Structured model replies by outcome",
                             ["task", "outcome"])
REPAIR_ATTEMPTS = 1
_ERRORS_SHOWN = 3


class StructuredOutputError(Exception):
    """The model did not produce a valid reply, even after the repair turn."""


def response_format(schema: Type[BaseModel], name: str) -> Dict[str, Any]:
    return {"type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": to_strict_json_schema(schema)}}


def record_outcome(task: str, outcome: str) -> None:
    STRUCTURED_OUTPUTS.labels(task=task, outcome=outcome).inc()


def repair_messages(messages: List[Dict[str, Any]], reply: str, problem: str) -> List[Dict[str, Any]]:
    """The original request plus the bad reply and what was wrong with it."""
    return list(messages) + [
        {"role": "assistant", "content": reply[:4000] or "(empty reply)"},
        {"role": "user", "content": f"That reply could not be used: {problem}. "
                                    "Reply again with only the corrected JSON."},
    ]


def describe(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = [f"{'.'.join(str(p) for p in item['loc']) or 'reply'}: {item['msg']}"
                 for item in error.errors()[:_ERRORS_SHOWN]]
        return "; ".join(parts)
    return str(error)[:300]


def structured_call(client: Any, schema: Type[T], *, task: str, messages: List[Dict[str, Any]],
                    model: str, max_tokens: int, temperature: float = 0.0,
                    repair_attempts: int = REPAIR_ATTEMPTS) -> T:
    """The validated reply as an instance of `schema`, or StructuredOutputError."""
    fmt = response_format(schema, task)
    conversation = list(messages)
    problem = ""
    for attempt in range(repair_attempts + 1):
        response = client.chat.completions.create(
            model=model, messages=conversation, max_completion_tokens=max_tokens,
            temperature=temperature, response_format=fmt,
        )
        choice = response.choices[0]
        reply = choice.message.content or ""
        refusal = getattr(choice.message, "refusal", None)
        if isinstance(refusal, str) and refusal:
            record_outcome(task, "refused")
            raise StructuredOutputError(f"{task}: model refused: {refusal[:200]}")
        try:
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError(f"the reply was cut off at {max_tokens} tokens; be more concise")
            value = schema.model_validate_json(reply)
        except (ValidationError, ValueError) as exc:
            problem = describe(exc)
            logger.warning("[%s] invalid structured reply (attempt %d): %s", task, attempt + 1, problem)
            conversation = repair_messages(messages, reply, problem)
            continue
        record_outcome(task, "ok" if attempt == 0 else "repaired")
        return value
    record_outcome(task, "failed")
    raise StructuredOutputError(f"{task}: no valid reply after {repair_attempts + 1} attempts: {problem}")


def parse_json_object(reply: str) -> Optional[Dict[str, Any]]:
    """A JSON object from a free-form JSON reply, or None (for json_object mode without a schema)."""
    try:
        value = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None
