"""Ownership, status and resume leases for DeepResearch runs.

The research state itself (plan, notes, report) lives in the LangGraph
checkpoint keyed by thread_id == run_id. This table only answers who owns a
run, what state it is in, and whether a resume may claim it — keeping a
single source of truth for the research progress.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import and_, or_

from src.database.models import ResearchRun

LEASE_SECONDS = 600


class ResearchRunStore:
    def __init__(self, db, user_id: int):
        self.db = db
        self.user_id = user_id

    def create(self, question: str, model_versions: Dict) -> Dict:
        run_id = str(uuid.uuid4())
        meta = {"question": question, "model_versions": model_versions}
        self.db.add(ResearchRun(
            run_id=run_id, user_id=self.user_id, question=question,
            state_json=json.dumps(meta), status="running",
        ))
        self.db.commit()
        return {"run_id": run_id, "status": "running", **meta}

    def _row(self, run_id: str) -> Optional[ResearchRun]:
        return self.db.query(ResearchRun).filter(
            ResearchRun.run_id == run_id, ResearchRun.user_id == self.user_id,
        ).first()

    def load(self, run_id: str) -> Optional[Dict]:
        row = self._row(run_id)
        if row is None:
            return None
        meta = json.loads(row.state_json or "{}")
        return {"run_id": row.run_id, "question": row.question, "status": row.status,
                "model_versions": meta.get("model_versions"),
                "stop_reason": meta.get("stop_reason")}

    def claim(self, run_id: str) -> Optional[Dict]:
        """Atomically claim a paused or stale run; reject active duplicates."""
        now = datetime.now(timezone.utc)
        stale_before = now - timedelta(seconds=LEASE_SECONDS)
        changed = self.db.query(ResearchRun).filter(
            ResearchRun.run_id == run_id, ResearchRun.user_id == self.user_id,
            or_(ResearchRun.status == "paused",
                and_(ResearchRun.status == "running", ResearchRun.updated_at < stale_before)),
        ).update({ResearchRun.status: "running", ResearchRun.updated_at: now}, synchronize_session=False)
        self.db.commit()
        return self.load(run_id) if changed == 1 else None

    def set_status(self, run_id: str, status: str, stop_reason: Optional[str] = None) -> None:
        row = self._row(run_id)
        if row is None:
            raise ValueError("Run does not exist for this user")
        meta = json.loads(row.state_json or "{}")
        meta["stop_reason"] = stop_reason
        row.state_json = json.dumps(meta)
        row.status = status
        row.updated_at = datetime.now(timezone.utc)
        self.db.commit()

    def list_recent(self, limit: int = 20) -> list[Dict]:
        rows = self.db.query(ResearchRun).filter(
            ResearchRun.user_id == self.user_id,
        ).order_by(ResearchRun.updated_at.desc()).limit(limit).all()
        stale_before = datetime.now(timezone.utc) - timedelta(seconds=LEASE_SECONDS)
        return [{"run_id": row.run_id, "question": row.question,
                 "status": row.status,
                 "resumable": row.status == "paused" or (
                     row.status == "running" and row.updated_at is not None and
                     row.updated_at.replace(tzinfo=timezone.utc) < stale_before),
                 "updated_at": row.updated_at.isoformat() if row.updated_at else None}
                for row in rows]
