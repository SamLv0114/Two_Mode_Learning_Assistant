"""Durable checkpoints for resumable DeepResearch tasks."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from src.database.models import ResearchRun
from sqlalchemy import and_, or_


LEASE_SECONDS = 600


class ResearchRunStore:
    def __init__(self, db, user_id: int):
        self.db = db
        self.user_id = user_id

    def create(self, question: str, plan: list, model_versions: Dict) -> Dict:
        state = {
            "run_id": str(uuid.uuid4()), "question": question,
            "plan": plan, "notes": {}, "status": "running",
            "stop_reason": None, "model_versions": model_versions,
        }
        self.db.add(ResearchRun(
            run_id=state["run_id"], user_id=self.user_id, question=question,
            state_json=json.dumps(state), status="running",
        ))
        self.db.commit()
        return state

    def load(self, run_id: str) -> Optional[Dict]:
        row = self.db.query(ResearchRun).filter(
            ResearchRun.run_id == run_id, ResearchRun.user_id == self.user_id,
        ).first()
        return json.loads(row.state_json) if row else None

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
        if changed != 1:
            return None
        state = self.load(run_id)
        state["status"] = "running"
        self.save(state)
        return state

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

    def save(self, state: Dict) -> None:
        row = self.db.query(ResearchRun).filter(
            ResearchRun.run_id == state["run_id"], ResearchRun.user_id == self.user_id,
        ).first()
        if row is None:
            raise ValueError("Run does not exist for this user")
        row.state_json = json.dumps(state)
        row.status = state["status"]
        row.updated_at = datetime.now(timezone.utc)
        self.db.commit()
