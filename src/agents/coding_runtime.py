"""Shared checkpoint and execution helpers for Coding Agent jobs.

The API only enqueues work. A separate process calls these helpers to drive
LangGraph and can resume from the durable checkpoint after a restart.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import update

from src.agents import research_profile
from src.agents.graph_runtime import close_checkpointer, get_checkpointer  # noqa: F401  (re-exported)
from src.database.models import PracticeComparison, PracticeEvent, PracticeRun, SessionLocal
from src.utils.config import settings
from src.utils.llm_usage import request_budget
from src.utils.tracing import graph_config

logger = logging.getLogger(__name__)
_checkpoint_use_lock = threading.RLock()


def graph_for(kind: str, services=None):
    from src.agents.coding_practice import build_paper_graph, build_repo_graph
    saver = get_checkpointer()
    return build_paper_graph(services, saver) if kind == "paper" else build_repo_graph(services, saver)


def graph_state(run_id: str, kind: str) -> Dict[str, Any]:
    # FastAPI can execute synchronous endpoints concurrently. The Postgres
    # saver owns one connection per process, so serialize shared-saver reads.
    with _checkpoint_use_lock:
        graph = graph_for(kind)
        return graph.get_state({"configurable": {"thread_id": run_id}}).values or {}


def _set_status(db, run_id: str, value: str, stage: str, error: str | None = None):
    db.execute(update(PracticeRun).where(
        PracticeRun.run_id == run_id,
        PracticeRun.status != "cancelled",
    ).values(status=value, stage=stage, error=error,
             updated_at=datetime.now(timezone.utc)))
    db.commit()


def record_event(db, run_id: str, kind: str, payload: Dict[str, Any]) -> None:
    """Persist one progress event; the API streams new rows to the page."""
    db.add(PracticeEvent(run_id=run_id, kind=str(kind)[:40],
                         payload_json=json.dumps(payload, ensure_ascii=False, default=str)[:4000]))
    db.commit()


def _record_studied(db, row: PracticeRun, values: Dict[str, Any]) -> None:
    """A finished implementation tells recommendation what the learner now knows (research_profile)."""
    try:
        research_profile.record_studied(db, row.user_id, values.get("paper_id") or row.source,
                                        values.get("title") or row.source, values.get("objective") or "")
    except Exception as exc:  # the run itself succeeded; this is a side record
        db.rollback()
        logger.warning("Could not record run %s on the research profile: %s", row.run_id, exc)


def run_graph_job(run_id: str) -> None:
    """Run or resume one claimed Coding Agent graph job."""
    from langgraph.types import Command

    from src.agents.coding_practice import PracticeCancelled
    from src.agents.coding_services import CodingServices
    from src.api.deps import get_embedding_manager

    db = SessionLocal()
    services = None
    recorded_input = 0
    recorded_output = 0
    try:
        row = db.get(PracticeRun, run_id)
        if row is None or row.status == "cancelled":
            return
        services = CodingServices(
            db,
            embedding_manager=get_embedding_manager() if row.kind == "paper" else None,
            run_id=run_id,
            user_id=row.user_id,
        )
        graph = graph_for(row.kind, services)
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 40,
                  **graph_config(f"coding_agent_{row.kind}", ["coding-agent", row.kind],
                                 practice_run_id=run_id, user_id=row.user_id, source=row.source)}
        existing = graph.get_state(config)
        if existing.values and existing.interrupts and row.resume_json:
            payload = Command(resume=json.loads(row.resume_json))  # the learner answered the plan check
            record_event(db, run_id, "resumed", {"stage": "generating"})
        elif existing.values:
            payload = None  # Resume from the last durable checkpoint.
        else:
            request = json.loads(row.request_json or "{}")
            payload = ({"paper_id": row.source, "objective": request.get("objective", ""),
                        "attempts": 0, "execution_enabled": request.get("execution_enabled", False),
                        "confirm_plan": bool(request.get("confirm_plan", False))}
                       if row.kind == "paper" else {"repo_url": row.source})
            record_event(db, run_id, "started", {"kind": row.kind, "source": row.source})
        # A safety ceiling on what one run (or one resumption of it) may spend on the model;
        # past it, model calls raise BudgetExceeded and the run fails with that reason.
        with request_budget(f"coding_{row.kind}", settings.CODING_RUN_TOKEN_BUDGET):
            for mode, update_item in graph.stream(payload, config, stream_mode=["updates", "custom"],
                                                  durability="sync"):
                if mode == "custom" and isinstance(update_item, dict):
                    event = dict(update_item)
                    record_event(db, run_id, event.pop("kind", "progress"), event)
                    continue
                if isinstance(update_item, dict):
                    stage = next(iter(update_item), "running")
                    if stage != "__interrupt__":
                        _set_status(db, run_id, "running", stage)
                        delta_input = services.input_tokens - recorded_input
                        delta_output = services.output_tokens - recorded_output
                        if delta_input or delta_output:
                            db.execute(update(PracticeRun).where(PracticeRun.run_id == run_id).values(
                                input_tokens=PracticeRun.input_tokens + delta_input,
                                output_tokens=PracticeRun.output_tokens + delta_output))
                            db.commit()
                            recorded_input = services.input_tokens
                            recorded_output = services.output_tokens
        pending = graph.get_state(config).interrupts
        if pending:
            # Paused at the plan check; POST /runs/{id}/plan stores the decision and requeues the run.
            manifest = (pending[0].value or {}).get("manifest", {}) if isinstance(pending[0].value, dict) else {}
            record_event(db, run_id, "awaiting_confirmation", {"open_choices": len(manifest.get("open_choices", []))})
            _set_status(db, run_id, "waiting", "confirm_plan")
            return
        _set_status(db, run_id, "completed", "finished")
        if row.kind == "paper":
            _record_studied(db, row, graph.get_state(config).values)
    except PracticeCancelled:
        logger.info("Coding Agent run %s cancelled", run_id)
    except Exception as exc:
        logger.exception("Coding Agent run %s failed", run_id)
        _set_status(db, run_id, "failed", "failed", str(exc)[:500])
    finally:
        if services is not None:
            delta_input = services.input_tokens - recorded_input
            delta_output = services.output_tokens - recorded_output
            if delta_input or delta_output:
                db.execute(update(PracticeRun).where(PracticeRun.run_id == run_id).values(
                    input_tokens=PracticeRun.input_tokens + delta_input,
                    output_tokens=PracticeRun.output_tokens + delta_output))
                db.commit()
        db.close()


def run_comparison_job(run_id: str) -> None:
    from src.agents.coding_practice import PracticeError
    from src.agents.coding_services import CodingServices

    db = SessionLocal()
    try:
        row = db.get(PracticeRun, run_id)
        comparison = db.get(PracticeComparison, run_id)
        if row is None or comparison is None or comparison.status != "running":
            return
        state = graph_state(run_id, row.kind)
        services = CodingServices(db, worker=None)
        result = services.compare_paper_repo(state)
        comparison.result_json = json.dumps(result)
        comparison.status = "completed"
        comparison.error = None
        row.input_tokens = (row.input_tokens or 0) + services.input_tokens
        row.output_tokens = (row.output_tokens or 0) + services.output_tokens
        db.commit()
    except Exception as exc:
        db.rollback()
        comparison = db.get(PracticeComparison, run_id)
        if comparison:
            comparison.status = "failed"
            comparison.error = str(exc)[:500] if isinstance(exc, PracticeError) else "Comparison failed; please retry later"
            db.commit()
        logger.exception("Paper/repository comparison failed for %s", run_id)
    finally:
        db.close()
