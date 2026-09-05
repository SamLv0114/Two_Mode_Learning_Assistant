#!/usr/bin/env python3
"""
Smoke tests — does the system turn on at all?

Deliberately not about correctness. These check the class of failure that takes
the whole app down while looking fine locally, which is exactly what this project
has shipped before:

  - src/pipelines/__init__.py imported a module that had been deleted, so
    POST /feed/generate raised ModuleNotFoundError for three commits. It survived
    because the import was inside a function, in a background task, so the server
    started cleanly and the error only reached a log.
  - `redis` was used throughout but never declared or installed. Every import sat
    behind try/except, so Redis features silently did nothing and /feed/refine
    always 404'd.

Both are caught below in under a second.

Runs standalone so there is nothing to install first:

    python tests/smoke.py            all checks
    python tests/smoke.py -v         show every check, not just failures

Exit code is 0 when everything passes, 1 otherwise — usable as a CI gate.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Keep third-party noise out of the report; a failure here is ours to see.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

# Distribution name -> import name, where they differ.
IMPORT_NAMES = {
    "python-dotenv": "dotenv",
    "pydantic-settings": "pydantic_settings",
    "psycopg2-binary": "psycopg2",
    "python-multipart": "multipart",
    "python-jose": "jose",
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "prometheus-client": "prometheus_client",
    "beautifulsoup4": "bs4",
    "tavily-python": "tavily",
    "PyMuPDF": "fitz",
    "uvicorn": "uvicorn",
}

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Register a check. The body raises AssertionError to fail."""
    def wrap(fn):
        try:
            fn()
            _results.append((name, True, ""))
        except Exception as e:
            detail = str(e) or e.__class__.__name__
            if not isinstance(e, AssertionError):
                detail = f"{e.__class__.__name__}: {detail}"
                if VERBOSE:
                    detail += "\n" + traceback.format_exc()
            _results.append((name, False, detail))
        return fn
    return wrap


# ── 1. Every module imports ───────────────────────────────────────────────────

@check("every module under src/ imports")
def _():
    failures = []
    for mod in pkgutil.walk_packages([str(REPO_ROOT / "src")], prefix="src."):
        if "__pycache__" in mod.name:
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as e:
            failures.append(f"{mod.name}: {e.__class__.__name__}: {e}")
    assert not failures, "\n      " + "\n      ".join(failures)


# ── 2. Declared dependencies are actually installed ───────────────────────────

@check("every package in requirements.txt is importable")
def _():
    req = (REPO_ROOT / "requirements.txt").read_text()
    missing = []
    for line in req.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        dist = re.split(r"[><=!\[]", line)[0].strip()
        if not dist:
            continue
        mod = IMPORT_NAMES.get(dist, dist.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(f"{dist} (import {mod})")
    assert not missing, (
        "declared but not installed — code guarded by try/except will "
        "silently do nothing:\n      " + "\n      ".join(missing)
    )


# ── 3. The app builds and its routes are registered ───────────────────────────

@check("FastAPI app constructs")
def _():
    from src.api.main import app
    assert app is not None


@check("expected routes are registered")
def _():
    from src.api.main import app
    # Read paths from the generated schema, not app.routes: included routers
    # appear there as _IncludedRouter objects with no .path, so walking routes
    # directly sees only the handful defined on the app itself.
    paths = set(app.openapi()["paths"])
    required = {
        "/api/v1/auth/login",
        "/api/v1/feed/generate",
        "/api/v1/feed/papers",
        "/api/v1/feed/refine",
        "/api/v1/feed/coldstart",
        "/api/v1/whats_hot",
        "/api/v1/chat/",
        "/api/v1/interactions",
        "/health",
    }
    missing = required - paths
    assert not missing, f"missing routes: {sorted(missing)}"


@check("OpenAPI schema generates")
def _():
    # Catches response_model / type-annotation mistakes that only surface when
    # FastAPI introspects the signatures.
    from src.api.main import app
    schema = app.openapi()
    assert schema.get("paths"), "no paths in generated schema"


# ── 4. Endpoints respond without server errors ────────────────────────────────

@check("unauthenticated endpoints respond (no 5xx)")
def _():
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        bad = []
        for path in ["/health", "/", "/docs", "/openapi.json"]:
            r = client.get(path)
            if r.status_code >= 500:
                bad.append(f"{path} -> {r.status_code}")
        assert not bad, ", ".join(bad)


@check("protected endpoints reject anonymous access with 401/403, not 500")
def _():
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        bad = []
        for method, path in [
            ("GET", "/api/v1/feed/papers"),
            ("GET", "/api/v1/feed/coldstart"),
            ("POST", "/api/v1/feed/refine"),
            ("GET", "/api/v1/whats_hot"),
            ("GET", "/api/v1/interactions/stats"),
        ]:
            r = client.request(method, path)
            if r.status_code not in (401, 403):
                bad.append(f"{method} {path} -> {r.status_code}")
        assert not bad, ", ".join(bad)


# ── 5. Rate limiting is wired, not just configured ────────────────────────────

@check("rate limiting actually enforces the configured limit")
def _():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.middleware import RateLimitMiddleware

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit_per_minute=3)

    @app.get("/ping")
    def ping():
        return {}

    @app.get("/health")
    def health():
        return {}

    with TestClient(app) as client:
        codes = [client.get("/ping").status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429], f"got {codes}"

        exempt = {client.get("/health").status_code for _ in range(6)}
        assert exempt == {200}, f"/health should be exempt, got {exempt}"


# ── 6. Ranking maths stays in bounds ──────────────────────────────────────────

@check("ranking weights always sum to 1 and clamp out-of-range input")
def _():
    from src.pipelines.daily_feed import _blend_weights

    for novelty in [-5, 0.0, 0.25, 0.5, 0.75, 1.0, 99]:
        w = _blend_weights(novelty)
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-9, f"novelty={novelty} sums to {total}"
        assert all(v >= 0 for v in w.values()), f"negative weight at {novelty}"

    assert _blend_weights(-5) == _blend_weights(0.0)
    assert _blend_weights(99) == _blend_weights(1.0)


@check("scoring survives degenerate and malformed candidate sets")
def _():
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    from src.collectors import PaperData
    from src.pipelines.daily_feed import DailyFeedPipeline, _blend_weights

    now = datetime.now(timezone.utc)

    def paper(i, days_old=0, citations=0, dated=True):
        return PaperData(
            arxiv_id=f"p{i}", title=f"P{i}", authors=[], abstract="x", categories=[],
            published_date=(now - timedelta(days=days_old)) if dated else None,
            arxiv_url="", pdf_url="", citation_count=citations,
        )

    def pipeline():
        p = DailyFeedPipeline.__new__(DailyFeedPipeline)
        p.user_id, p.db, p.embedding_manager = 1, MagicMock(), MagicMock()
        p._novelty_preference = None
        p._disliked_embs_cached, p._disliked_embs = False, None
        p._get_saved_paper_embeddings = lambda: None
        p._compute_disliked_embeddings = lambda: None
        return p

    cases = {
        "empty": [],
        "single": [paper(1)],
        "all identical (both signals degenerate)": [paper(i, 5, 0) for i in range(3)],
        "no publication dates": [paper(i, dated=False) for i in range(3)],
        "negative citations / future dates": [paper(1, 0, -5), paper(2, -10, 0)],
        "normal spread": [paper(i, i * 30, i * 100) for i in range(1, 8)],
    }
    for label, papers in cases.items():
        sem = {p.arxiv_id: 0.5 for p in papers}
        out = pipeline()._score_and_rank(
            papers, "x", _blend_weights(0.5), semantic_scores_override=sem, top_k=5
        )
        assert len(out) <= 5, f"{label}: returned {len(out)} for top_k=5"
        for p in out:
            assert p.relevance_score is not None, f"{label}: unscored paper"


@check("exploration is stable within a day and varies across users")
def _():
    from unittest.mock import MagicMock

    from src.pipelines.daily_feed import DailyFeedPipeline

    def rng_for(uid):
        p = DailyFeedPipeline.__new__(DailyFeedPipeline)
        p.user_id, p.db = uid, MagicMock()
        return p._daily_rng()

    pool = list(range(50))
    first = rng_for(1).sample(pool, 5)
    again = rng_for(1).sample(pool, 5)
    assert first == again, "same user+day must reproduce the same picks"

    others = [tuple(rng_for(u).sample(pool, 5)) for u in range(1, 8)]
    assert len(set(others)) > 1, "every user drew identical picks"


# ── 7. Documented behaviour matches the code ──────────────────────────────────

@check("module docstring references only symbols that exist")
def _():
    import ast

    src = (REPO_ROOT / "src/pipelines/daily_feed.py").read_text()
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""

    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            defined |= {m.name for m in node.body if isinstance(m, ast.FunctionDef)}

    referenced = {
        m for m in re.findall(r"\b(_[a-z][a-z_]{3,}|get_coldstart_seeds)\b", doc)
        if not m.startswith("__")
    }
    stale = referenced - defined
    assert not stale, f"docstring names symbols that no longer exist: {sorted(stale)}"


@check("no imports left inside functions that shadow module-level ones")
def _():
    # A function-local import placed *after* first use raises NameError only at
    # runtime; this repo hit that twice in one sitting.
    import ast

    problems = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            imported: dict[str, int] = {}
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)) and node.col_offset > 0:
                    for alias in node.names:
                        imported[alias.asname or alias.name.split(".")[0]] = node.lineno
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    line = imported.get(node.id)
                    if line is not None and node.lineno < line:
                        problems.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} "
                            f"uses '{node.id}' before its local import on line {line}"
                        )
    assert not problems, "\n      " + "\n      ".join(sorted(set(problems)))


# ── Report ────────────────────────────────────────────────────────────────────

def main() -> int:
    passed = [r for r in _results if r[1]]
    failed = [r for r in _results if not r[1]]

    for name, ok, detail in _results:
        if ok and VERBOSE:
            print(f"  PASS  {name}")
        elif not ok:
            print(f"  FAIL  {name}")
            print(f"        {detail}")

    print(f"\n  {len(passed)}/{len(_results)} checks passed")
    if failed:
        print("  Smoke test failed — the build is not safe to deploy.")
        return 1
    print("  Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
