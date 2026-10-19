"""
What every model call costs, and how much one request may spend.

All agents get their client from tracing.openai_client(), which routes
chat.completions.create through `accounted_create` here. Each call then:

  1. is refused with BudgetExceeded when the request's token budget is spent,
     so a looping or over-eager agent cannot run up an unbounded bill;
  2. is retried once on LLM_FALLBACK_MODEL (when set) after a provider outage,
     rate limit or retired model, instead of failing the request;
  3. has its tokens and estimated cost recorded: Prometheus counters per model
     and task, and the request's running total.

A request opens a budget with `request_budget(...)` (chat turns, Coding Agent
runs). Optional work asks `budget_allows(reserve)` first and skips itself when
only the reserve is left, so the step that produces the answer still fits.
Calls outside any budget (nightly jobs, background extraction) are only counted.
"""
import contextvars
import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import openai
from prometheus_client import Counter, Histogram

from src.utils.config import settings

logger = logging.getLogger(__name__)

# USD per million tokens, from the versioned table the evaluation harness also prices
# runs with (each entry records its source page). Longest matching prefix wins, so dated
# snapshots ("gpt-4o-mini-2024-07-18") resolve to their family. A model missing from the
# table is counted at zero cost and logged once: add it there, with its source.
PRICES: Dict[str, Dict[str, float]] = json.loads(
    (Path(__file__).with_name("model_prices.json")).read_text(encoding="utf-8"))["models"]

LLM_TOKENS = Counter("llm_tokens_total", "Tokens sent to and received from the model",
                     ["model", "direction"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated model spend in USD", ["model"])
LLM_FALLBACKS = Counter("llm_fallback_total", "Calls retried on the fallback model", ["from_model", "error"])
REQUEST_COST = Histogram("llm_request_cost_usd", "Estimated model spend per budgeted request", ["label"],
                         buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
BUDGET_EXHAUSTED = Counter("llm_budget_exhausted_total", "Requests that used up their token budget", ["label"])

# Failures a different model can get past. A 400 is the request's own fault and
# would fail on any model, so it is not among them.
FALLBACK_ERRORS = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError,
                   openai.APITimeoutError, openai.NotFoundError)

_unpriced_logged: set = set()
_bad_routes_logged: set = set()


class BudgetExceeded(RuntimeError):
    """The request's token budget is spent; the caller falls back to what it already has."""


def _routes() -> Dict[str, str]:
    raw = (settings.MODEL_ROUTES or "").strip()
    if not raw:
        return {}
    try:
        routes = json.loads(raw)
        if isinstance(routes, dict):
            return {str(k): str(v) for k, v in routes.items()}
    except json.JSONDecodeError:
        pass
    if raw not in _bad_routes_logged:
        _bad_routes_logged.add(raw)
        logger.warning("MODEL_ROUTES is not a JSON object of task -> model; ignoring it")
    return {}


def model_for(task: str, default: Optional[str] = None) -> str:
    """The model for a task: MODEL_ROUTES[task] when configured, else the default, else LLM_MODEL."""
    return _routes().get(task) or default or settings.LLM_MODEL


def price(model: str) -> Tuple[float, float, float]:
    """(input, cached input, output) USD per million tokens."""
    matches = [key for key in PRICES if (model or "").startswith(key)]
    if not matches:
        if model not in _unpriced_logged:
            _unpriced_logged.add(model)
            logger.warning("No price for model %r; its calls are counted at zero cost", model)
        return 0.0, 0.0, 0.0
    rate = PRICES[max(matches, key=len)]
    return rate["input"], rate.get("cached_input", rate["input"]), rate["output"]


def cost_usd(model: str, usage: Any) -> float:
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = int(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
    input_price, cached_price, output_price = price(model)
    return ((prompt - cached) * input_price + cached * cached_price + completion * output_price) / 1_000_000


class RequestBudget:
    """Token allowance for one request, shared by every call made on its behalf (thread-safe)."""

    def __init__(self, label: str, max_tokens: int):
        self.label = label
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def charge(self, tokens: int, cost: float) -> None:
        with self._lock:
            self.used_tokens += tokens
            self.cost_usd += cost
            self.calls += 1

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def allows(self, reserve: int = 0) -> bool:
        return self.remaining > reserve

    def check(self) -> None:
        if self.used_tokens >= self.max_tokens:
            raise BudgetExceeded(f"{self.label}: token budget of {self.max_tokens} used up")


_current: contextvars.ContextVar[Optional[RequestBudget]] = contextvars.ContextVar("llm_request_budget", default=None)


def current_budget() -> Optional[RequestBudget]:
    return _current.get()


def budget_allows(reserve: int = 0) -> bool:
    """False when the current request has only `reserve` tokens (or fewer) left; True outside any budget."""
    budget = _current.get()
    return budget is None or budget.allows(reserve)


@contextmanager
def request_budget(label: str, max_tokens: int) -> Iterator[RequestBudget]:
    """
    Calls made inside share one budget, including from threads started with a copied
    context (LangGraph nodes, ToolRegistry workers). Plain threading.Thread does not copy it.
    """
    budget = RequestBudget(label, max_tokens)
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)
        REQUEST_COST.labels(label=label).observe(budget.cost_usd)
        if budget.used_tokens >= budget.max_tokens:
            BUDGET_EXHAUSTED.labels(label=label).inc()
            logger.warning("%s used its whole token budget (%d tokens, %d calls)",
                           label, budget.used_tokens, budget.calls)


def record_usage(model: str, usage: Any) -> None:
    if usage is None:
        return
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = cost_usd(model, usage)
    LLM_TOKENS.labels(model=model, direction="input").inc(prompt)
    LLM_TOKENS.labels(model=model, direction="output").inc(completion)
    LLM_COST.labels(model=model).inc(cost)
    budget = _current.get()
    if budget is not None:
        budget.charge(prompt + completion, cost)


class _UsageStream:
    """A streamed completion that records the usage OpenAI sends in its last chunk, and hides that chunk."""

    def __init__(self, inner: Any, model: str):
        self._inner = inner
        self._model = model

    def __iter__(self):
        for chunk in self._inner:
            if getattr(chunk, "usage", None) is not None and not getattr(chunk, "choices", None):
                record_usage(self._model, chunk.usage)
                continue
            yield chunk

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close:
            close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def accounted_create(completions: Any, **kwargs: Any) -> Any:
    budget = _current.get()
    if budget is not None:
        budget.check()
    if kwargs.get("stream"):
        kwargs.setdefault("stream_options", {"include_usage": True})
    try:
        response = completions.create(**kwargs)
    except FALLBACK_ERRORS as exc:
        fallback = settings.LLM_FALLBACK_MODEL
        if not fallback or fallback == kwargs.get("model"):
            raise
        LLM_FALLBACKS.labels(from_model=str(kwargs.get("model")), error=type(exc).__name__).inc()
        logger.warning("%s failed on %s (%s); retrying once on %s",
                       "Streamed call" if kwargs.get("stream") else "Call", kwargs.get("model"), exc, fallback)
        kwargs = {**kwargs, "model": fallback}
        response = completions.create(**kwargs)
    model = str(kwargs.get("model") or "")
    if kwargs.get("stream"):
        return _UsageStream(response, model)
    record_usage(model, getattr(response, "usage", None))
    return response


class _Completions:
    def __init__(self, inner: Any):
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        return accounted_create(self._inner, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner: Any):
        self._inner = inner
        self.completions = _Completions(inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AccountedClient:
    """An OpenAI client whose chat completions go through accounted_create; everything else passes through."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.chat = _Chat(inner.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
