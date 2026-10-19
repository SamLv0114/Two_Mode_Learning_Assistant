"""Process-wide LangGraph checkpointer shared by the graph-based agents.

DeepResearch and the Coding Agent both persist graph state per thread_id
(their run_id). One saver per process: the Postgres saver owns a single
connection and both savers serialise access to it with their own lock.
"""
import atexit
import threading

from src.utils.config import settings

_checkpoint_lock = threading.Lock()
_checkpoint_saver = None
_checkpoint_context = None


def get_checkpointer():
    """Create one checkpointer per process instead of rebuilding it per request."""
    global _checkpoint_saver, _checkpoint_context
    if _checkpoint_saver is not None:
        return _checkpoint_saver
    with _checkpoint_lock:
        if _checkpoint_saver is not None:
            return _checkpoint_saver
        if settings.DATABASE_URL.startswith("sqlite"):
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpoint_path = settings.DATA_DIR / "practice_checkpoints.sqlite"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _checkpoint_context = SqliteSaver.from_conn_string(str(checkpoint_path))
            _checkpoint_saver = _checkpoint_context.__enter__()
        else:
            from langgraph.checkpoint.postgres import PostgresSaver
            url = settings.DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
            _checkpoint_context = PostgresSaver.from_conn_string(url)
            _checkpoint_saver = _checkpoint_context.__enter__()
            _checkpoint_saver.setup()
    return _checkpoint_saver


def close_checkpointer():
    global _checkpoint_saver, _checkpoint_context
    if _checkpoint_context is not None:
        _checkpoint_context.__exit__(None, None, None)
    _checkpoint_saver = None
    _checkpoint_context = None


atexit.register(close_checkpointer)
