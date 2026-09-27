"""
Shared helpers for the evaluation runners.

Everything here lives only in the evaluation process — production code is not
modified. Token accounting works by wrapping the OpenAI SDK's
`Completions.create` at class level for the lifetime of the eval run, so every
LLM call made anywhere in the codebase (intent classification, agent loop,
critic, planner, ...) is counted without touching call sites.
"""
import hashlib
import inspect
import json
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"


# ── Token accounting ──────────────────────────────────────────────────────────

class UsageTracker:
    """Counts OpenAI token usage across all call sites in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0
            self.by_model = {}
            self.by_stage = {}

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, usage, model: str = "unknown", stage: str = "unknown") -> None:
        if usage is None:
            return
        with self._lock:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.calls += 1
            bucket = self.by_model.setdefault(model, {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0})
            bucket["prompt_tokens"] += prompt
            bucket["cached_tokens"] += cached
            bucket["completion_tokens"] += completion
            bucket["calls"] += 1
            stage_models = self.by_stage.setdefault(stage, {})
            stage_bucket = stage_models.setdefault(model, {
                "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0,
            })
            stage_bucket["prompt_tokens"] += prompt
            stage_bucket["cached_tokens"] += cached
            stage_bucket["completion_tokens"] += completion
            stage_bucket["calls"] += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "llm_calls": self.calls,
                "by_model": {model: dict(values) for model, values in self.by_model.items()},
                "by_stage": {stage: {model: dict(values) for model, values in models.items()}
                             for stage, models in self.by_stage.items()},
            }


def estimated_cost_usd(snapshot: Dict) -> Optional[float]:
    """Use a versioned price table; unknown models are not priced as zero."""
    prices = json.loads((EVAL_DIR / "pricing.json").read_text(encoding="utf-8"))["models"]
    total = 0.0
    for model, usage in snapshot.get("by_model", {}).items():
        alias = next((name for name in sorted(prices, key=len, reverse=True) if model.startswith(name)), None)
        if alias is None:
            return None
        rate = prices[alias]
        uncached = usage["prompt_tokens"] - usage["cached_tokens"]
        total += (uncached * rate["input"] + usage["cached_tokens"] * rate["cached_input"]
                  + usage["completion_tokens"] * rate["output"]) / 1_000_000
    return total


USAGE = UsageTracker()
_patched = False


def install_usage_patch() -> bool:
    """Wrap Completions.create so every LLM call reports into USAGE.

    Returns False (and leaves the SDK untouched) if the expected attribute path
    isn't there — token columns then read 0 rather than the eval crashing.
    """
    global _patched
    if _patched:
        return True
    try:
        from openai.resources.chat.completions import Completions
    except Exception as e:  # pragma: no cover - depends on SDK layout
        print(f"[harness] token accounting unavailable: {e}")
        return False

    original = Completions.create

    def wrapped(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        stage = "unknown"
        for frame in inspect.stack()[1:]:
            module = frame.frame.f_globals.get("__name__", "")
            if module.startswith("src.agents.") or module.startswith("src.evaluation."):
                stage = f"{module.rsplit('.', 1)[-1]}.{frame.function}"
                break
        USAGE.record(getattr(response, "usage", None), str(kwargs.get("model", "unknown")), stage)
        return response

    Completions.create = wrapped
    _patched = True
    return True


@contextmanager
def measure():
    """Per-item measurement: wall-clock plus the tokens that item consumed."""
    USAGE.reset()
    t0 = time.time()
    box: Dict[str, Any] = {}
    try:
        yield box
    finally:
        box["latency_ms"] = int((time.time() - t0) * 1000)
        box.update(USAGE.snapshot())
        box["estimated_cost_usd"] = estimated_cost_usd(box)


# ── Dataset IO ────────────────────────────────────────────────────────────────

def load_jsonl(name: str) -> List[Dict]:
    path = DATASETS_DIR / name
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(name: str, rows: Iterable[Dict]) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASETS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# ── Stats ─────────────────────────────────────────────────────────────────────

def pct(values: List[float], q: float) -> float:
    """Percentile with nearest-rank on the sorted sample (q in [0,1])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# ── Reporting ─────────────────────────────────────────────────────────────────

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EVAL_DIR.parent, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def source_fingerprint() -> str:
    """Hash evaluated code, including new untracked Python files."""
    digest = hashlib.sha256()
    for root in (EVAL_DIR.parent / "src", EVAL_DIR / "runners"):
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(EVAL_DIR.parent)).encode())
            digest.update(path.read_bytes())
    digest.update((EVAL_DIR / "_harness.py").read_bytes())
    return digest.hexdigest()


def worktree_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "evaluation/runners",
             "evaluation/_harness.py", "evaluation/datasets", "evaluation/rubrics"],
            cwd=EVAL_DIR.parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except Exception:
        return True


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def result_header(title: str, config: Dict[str, Any]) -> str:
    lines = [f"# {title}", "", "```"]
    entries = [
        ("date", date.today().isoformat()),
        ("git_commit", git_commit()),
        ("worktree_dirty", worktree_dirty()),
        ("source_sha256", source_fingerprint()),
        *config.items(),
    ]
    width = max(16, max(len(str(key)) for key, _ in entries) + 2)
    for key, value in entries:
        lines.append(f"{key:<{width}}{value}")
    lines.append("```")
    return "\n".join(lines)


def write_result(filename: str, content: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"\n-> {path}")
    return path
