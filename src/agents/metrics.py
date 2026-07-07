"""
Prometheus metrics for the agent system.

Metrics exposed at GET /metrics (scraped by Prometheus every 15s):

  agent_requests_total{intent, agent}       — request counts by routing outcome
  agent_latency_seconds{agent}              — response time histogram
  tool_calls_total{tool}                    — which tools get called and how often
  intent_recognition_method_total{method}   — keyword vs embedding vs llm usage
  eval_score_gauge{dimension}               — latest LLM-as-Judge scores
"""
from prometheus_client import Counter, Histogram, Gauge

# ── Request tracking ──────────────────────────────────────────────────────────

AGENT_REQUESTS = Counter(
    "agent_requests_total",
    "Total chat requests handled, labelled by detected intent and agent used",
    ["intent", "agent"],
)

# ── Latency ───────────────────────────────────────────────────────────────────

AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "End-to-end agent response latency in seconds",
    ["agent"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# ── Tool usage ────────────────────────────────────────────────────────────────

TOOL_CALLS = Counter(
    "tool_calls_total",
    "Total tool invocations made by agents during the tool-calling loop",
    ["tool"],
)

# ── Intent recognition ────────────────────────────────────────────────────────

INTENT_RECOGNITION_METHOD = Counter(
    "intent_recognition_method_total",
    "How often each recognition stage was the final decision-maker",
    ["method"],  # keyword | embedding | llm
)

# ── LLM-as-Judge scores ───────────────────────────────────────────────────────

EVAL_SCORE = Gauge(
    "agent_eval_score",
    "Most recent LLM-as-Judge score per quality dimension (0-10)",
    ["dimension"],  # relevance | accuracy | completeness | usefulness | overall
)


# ── Helper called from chat.py ────────────────────────────────────────────────

def record_request(
    intent: str,
    agent: str,
    method: str,
    latency_ms: int,
    tools_called: list,
    eval_scores: dict = None,
) -> None:
    """Record all metrics for a single /chat request."""
    AGENT_REQUESTS.labels(intent=intent, agent=agent).inc()
    AGENT_LATENCY.labels(agent=agent).observe(latency_ms / 1000)
    INTENT_RECOGNITION_METHOD.labels(method=method).inc()

    for tool in tools_called:
        TOOL_CALLS.labels(tool=tool).inc()

    if eval_scores:
        for dimension, score in eval_scores.items():
            if dimension != "reasoning" and isinstance(score, (int, float)):
                EVAL_SCORE.labels(dimension=dimension).set(score)
