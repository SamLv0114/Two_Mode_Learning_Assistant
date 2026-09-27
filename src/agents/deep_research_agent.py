"""
DeepResearchAgent — TODO-driven multi-stage research pipeline.

Inspired by Hello-Agents ch14 Deep Research Agent architecture.

Three-stage pipeline:
  1. TodoPlanner    (gpt-4o)       — decomposes question → JSON sub-task list
  2. TaskSummarizer (gpt-4o-mini)  — searches KB + web per sub-task → summary
  3. ReportWriter   (gpt-4o)       — synthesizes summaries → structured Markdown report

The stream() method emits SSE-friendly events so the frontend can show
real-time research progress:
  {"type": "plan",         "tasks": [{id, title}, ...]}
  {"type": "task_started", "id": 1, "title": "..."}
  {"type": "task_done",    "id": 1, "citations": 3}
  {"type": "generating"}
  {"type": "token",        "value": "..."}    ← repeats per token
  {"type": "done",         "tools_called": [...], "citations": [...]}
"""
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, List

from openai import OpenAI

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.source_safety import SOURCE_POLICY, untrusted_block
from src.agents.research_run_store import ResearchRunStore
from src.agents.tools import (
    SEARCH_KNOWLEDGE_BASE,
    SEARCH_WEB,
    execute_tool,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)

PLANNER_MODEL = "gpt-4o"
SUMMARIZER_MODEL = "gpt-4o-mini"
WRITER_MODEL = "gpt-4o"
MAX_TASKS = 5


def research_stop_reason(notes_by_id: Dict[int, "ResearchNote"], total_tasks: int,
                         elapsed_seconds: float, max_seconds: int,
                         early_stop_enabled: bool = False) -> str | None:
    """Return an auditable stop decision at a completed-task boundary."""
    if len(notes_by_id) >= total_tasks:
        return "plan_complete"
    if elapsed_seconds >= max_seconds:
        return "time_budget"
    if early_stop_enabled and len(notes_by_id) >= 2:
        sources = {c.get("url") for note in notes_by_id.values() for c in note.citations if c.get("url")}
        if len(sources) >= 4:
            return "evidence_sufficient"
    return None


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TodoItem:
    id: int
    title: str
    query: str                  # focused search query for this sub-task
    intent: str = ""            # brief description of what to find


@dataclass
class ResearchNote:
    task_id: int
    title: str
    summary: str
    citations: List[Dict] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)


# ── Stage 1: TodoPlanner ──────────────────────────────────────────────────────

class TodoPlanner:
    """
    Decomposes a complex question into 3–5 focused sub-tasks using gpt-4o.
    Returns a list of TodoItem objects.
    """

    _PROMPT = """\
You are a research planning assistant for a research platform focused on deep learning, LLMs, and AI agents.

Break the following complex question into {max_tasks} or fewer focused sub-tasks.
Each sub-task should be independently searchable and together cover the full question.

Question: {question}

Return ONLY a JSON array (no markdown fences):
[
  {{
    "id": 1,
    "title": "<short sub-task label>",
    "query": "<specific search query to answer this sub-task>",
    "intent": "<one phrase: what type of information to find>"
  }},
  ...
]"""

    def __init__(self, client: OpenAI):
        self.client = client

    def plan(self, question: str) -> List[TodoItem]:
        prompt = self._PROMPT.format(question=question[:800], max_tasks=MAX_TASKS)
        try:
            response = self.client.chat.completions.create(
                model=PLANNER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            items = []
            for i, item in enumerate(data[:MAX_TASKS]):
                items.append(TodoItem(
                    id=item.get("id", i + 1),
                    title=str(item.get("title", f"Sub-task {i + 1}")),
                    query=str(item.get("query", question)),
                    intent=str(item.get("intent", "")),
                ))
            logger.info(f"[TodoPlanner] Planned {len(items)} tasks for: {question[:60]}...")
            return items
        except Exception as e:
            logger.warning(f"[TodoPlanner] Planning failed: {e} — falling back to single task")
            return [TodoItem(id=1, title="Full research", query=question, intent="comprehensive answer")]


# ── Stage 2: TaskSummarizer ───────────────────────────────────────────────────

class TaskSummarizer:
    """
    For each sub-task:
      1. Calls search_knowledge_base and search_web
      2. Summarizes findings with gpt-4o-mini
    Returns a ResearchNote with the summary and citations.
    """

    _SYSTEM = """\
You are a research summarizer. Given search results for a specific sub-task,
produce a concise, factual summary (3–5 sentences) covering the key findings.
Cite paper titles or URLs inline when relevant. Be specific — avoid filler phrases."""

    def __init__(self, client: OpenAI, fire_listener=None):
        self.client = client
        self._fire_listener = fire_listener or (lambda *a: None)

    def summarize(self, task: TodoItem, context: Dict[str, Any]) -> ResearchNote:
        cancel = context.get("_cancel_event")
        if cancel is not None and cancel.is_set():
            return ResearchNote(task_id=task.id, title=task.title, summary="Cancelled")
        tools_called: List[str] = []
        citations: List[Dict] = []
        search_snippets: List[str] = []

        # Search 1: knowledge base
        try:
            import time as _time
            _t0 = _time.time()
            kb_result = execute_tool("search_knowledge_base", {"query": task.query, "n_results": 4}, context)
            self._fire_listener("search_knowledge_base", {"query": task.query, "n_results": 4},
                                kb_result, int((_time.time() - _t0) * 1000), context)
            tools_called.append("search_knowledge_base")
            for r in kb_result.get("results", []):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:400]
                if title or url:
                    citations.append({"title": title, "url": url, "type": r.get("type", "unknown"), "evidence": content})
                if content:
                    search_snippets.append(untrusted_block(content, title or url or "knowledge_base"))
        except Exception as e:
            logger.warning(f"[TaskSummarizer] KB search failed for task {task.id}: {e}")

        # Search 2: web
        if cancel is not None and cancel.is_set():
            return ResearchNote(task_id=task.id, title=task.title, summary="Cancelled", citations=citations, tools_called=tools_called)
        try:
            _t0 = _time.time()
            web_result = execute_tool("search_web", {"query": task.query, "max_results": 3}, context)
            self._fire_listener("search_web", {"query": task.query, "max_results": 3},
                                web_result, int((_time.time() - _t0) * 1000), context)
            tools_called.append("search_web")
            for r in web_result.get("results", []):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:400]
                if title or url:
                    citations.append({"title": title, "url": url, "type": "web", "evidence": content})
                if content:
                    search_snippets.append(untrusted_block(content, title or url or "web"))
        except Exception as e:
            logger.warning(f"[TaskSummarizer] Web search failed for task {task.id}: {e}")

        # Summarize with gpt-4o-mini
        if cancel is not None and cancel.is_set():
            return ResearchNote(task_id=task.id, title=task.title, summary="Cancelled", citations=citations, tools_called=tools_called)
        if not search_snippets:
            summary = f"No relevant results found for: {task.query}"
        else:
            snippets_str = "\n\n".join(search_snippets[:6])
            user_msg = (
                f"Sub-task: {task.title}\n"
                f"Intent: {task.intent}\n\n"
                f"Search results:\n{snippets_str}"
            )
            try:
                response = self.client.chat.completions.create(
                    model=SUMMARIZER_MODEL,
                    messages=[
                        {"role": "system", "content": self._SYSTEM + "\n\n" + SOURCE_POLICY},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=400,
                    temperature=0.2,
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"[TaskSummarizer] Summarization failed for task {task.id}: {e}")
                summary = snippets_str[:600]

        logger.info(f"[TaskSummarizer] Task {task.id} done — {len(citations)} citations")
        return ResearchNote(
            task_id=task.id,
            title=task.title,
            summary=summary,
            citations=citations,
            tools_called=tools_called,
        )


# ── Stage 3: ReportWriter ─────────────────────────────────────────────────────

class ReportWriter:
    """Synthesizes all ResearchNotes into a structured Markdown report using gpt-4o."""

    _SYSTEM = """\
You are a senior research writer covering deep learning, LLMs, and AI agents. Synthesize the provided research notes into
a well-structured Markdown report with clear sections. Requirements:
- Use ## section headers for each major theme
- Be specific and cite sources inline (e.g., "According to [Paper Title]...")
- Conclude with a "## Summary" section with 3–5 bullet-point takeaways
- Professional but accessible tone; no filler phrases"""

    def __init__(self, client: OpenAI):
        self.client = client

    def _build_prompt(self, question: str, notes: List[ResearchNote]) -> str:
        notes_str = "\n\n".join(
            f"### Sub-task {n.task_id}: {n.title}\n{untrusted_block(n.summary, n.title)}" for n in notes
        )
        return (
            f"Original question: {question}\n\n"
            f"Research notes:\n{notes_str}\n\n"
            f"Write a comprehensive Markdown report answering the original question "
            f"using these research notes."
        )

    def write(self, question: str, notes: List[ResearchNote]) -> str:
        prompt = self._build_prompt(question, notes)
        try:
            response = self.client.chat.completions.create(
                model=WRITER_MODEL,
                messages=[
                    {"role": "system", "content": self._SYSTEM + "\n\n" + SOURCE_POLICY},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[ReportWriter] Report generation failed: {e}")
            return "\n\n".join(f"**{n.title}**\n{n.summary}" for n in notes)

    def stream(self, question: str, notes: List[ResearchNote]) -> Generator[str, None, None]:
        """Yield tokens from a streaming gpt-4o call."""
        prompt = self._build_prompt(question, notes)
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=WRITER_MODEL,
                messages=[
                    {"role": "system", "content": self._SYSTEM + "\n\n" + SOURCE_POLICY},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            logger.error(f"[ReportWriter] Streaming failed: {e}")
            # Fall back to yielding the non-streamed result
            for token in self.write(question, notes):
                yield token
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()


# ── DeepResearchAgent ─────────────────────────────────────────────────────────

class DeepResearchAgent(BaseAgent):
    """
    Multi-agent research pipeline: Plan → Execute tasks → Write report.

    Activated by AgentRouter for complex research_qa queries
    (long questions or those containing multi-faceted keywords).

    Streaming support emits rich progress events so the frontend can display
    the research plan and task-by-task progress in real time.
    """

    name = "DeepResearchAgent"
    system_prompt = ""    # each sub-component owns its own prompt
    tool_schemas = []     # sub-components call tools directly

    def __init__(self):
        super().__init__()  # sets self.client and self.tool_call_listener
        self.planner = TodoPlanner(self.client)
        self.summarizer = TaskSummarizer(self.client, fire_listener=self._fire_listener)
        self.writer = ReportWriter(self.client)

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        import time
        start = time.time()

        # Stage 1: plan
        if self._cancelled(context):
            return AgentResult(reply="")
        plan = self.planner.plan(message)
        if self._cancelled(context):
            return AgentResult(reply="")

        # Stage 2: execute tasks concurrently (V4: ThreadPoolExecutor, max 3 workers)
        notes_by_id: Dict[int, "ResearchNote"] = {}

        def _run_task(task):
            return task.id, self.summarizer.summarize(task, context)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_run_task, task): task for task in plan}
            for future in as_completed(futures):
                if self._cancelled(context):
                    return AgentResult(reply="")
                task_id, note = future.result()
                notes_by_id[task_id] = note

        # Preserve task order for coherent report
        notes: List[ResearchNote] = [notes_by_id[t.id] for t in plan if t.id in notes_by_id]

        # Stage 3: synthesize report
        if self._cancelled(context):
            return AgentResult(reply="")
        report = self.writer.write(message, notes)

        all_tools = [t for n in notes for t in n.tools_called]
        all_citations = [c for n in notes for c in n.citations]
        # Deduplicate citations by URL
        seen_urls = set()
        unique_citations = []
        for c in all_citations:
            key = c.get("url") or c.get("title", "")
            if key and key not in seen_urls:
                seen_urls.add(key)
                unique_citations.append(c)

        return AgentResult(
            reply=report,
            tools_called=all_tools,
            citations=unique_citations,
            processing_time_ms=int((time.time() - start) * 1000),
        )

    def stream(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Generator:
        """
        Yields SSE-compatible event dicts as research progresses.

        Frontend can render the plan immediately, then fill in results
        task-by-task, then stream the final report token by token.
        """
        if self._cancelled(context):
            yield {"type": "cancelled"}
            return
        db, user = context.get("db"), context.get("user")
        store = ResearchRunStore(db, user.id) if db is not None and user is not None else None
        resume_id = context.get("resume_run_id")
        if resume_id:
            state = store.load(resume_id) if store else None
            if state is None:
                yield {"type": "error", "value": "Research run not found for this user"}
                return
            if state["status"] != "completed":
                expected = {"planner": PLANNER_MODEL, "summarizer": SUMMARIZER_MODEL,
                            "writer": WRITER_MODEL, "schema": 1}
                if state.get("model_versions") != expected:
                    yield {"type": "error", "value": "Research run model version changed; restart this run"}
                    return
                claimed = store.claim(resume_id)
                if claimed is None:
                    yield {"type": "error", "value": "Research run is already active"}
                    return
                state = claimed
            message = state["question"]
            plan = [TodoItem(**item) for item in state["plan"]]
        else:
            plan = self.planner.plan(message)
            if self._cancelled(context):
                yield {"type": "cancelled"}
                return
            versions = {"planner": PLANNER_MODEL, "summarizer": SUMMARIZER_MODEL, "writer": WRITER_MODEL, "schema": 1}
            state = store.create(message, [asdict(item) for item in plan], versions) if store else {
                "run_id": None, "question": message, "plan": [asdict(item) for item in plan],
                "notes": {}, "status": "running", "stop_reason": None, "model_versions": versions,
            }

        completed = False
        try:
            if state["run_id"]:
                yield {"type": "run", "run_id": state["run_id"]}
            yield {"type": "plan", "tasks": [
                {"id": t.id, "title": t.title, "intent": t.intent} for t in plan
            ]}
            if state["status"] == "completed":
                yield {"type": "token", "value": state.get("report", "")}
                yield {"type": "done", "tools_called": state.get("tools", []), "citations": state.get("citations", [])}
                completed = True
                return

            notes_by_id = {int(task_id): ResearchNote(**note) for task_id, note in state["notes"].items()}
            for task in plan:
                if task.id in notes_by_id:
                    yield {"type": "task_done", "id": task.id, "citations": len(notes_by_id[task.id].citations)}
                else:
                    yield {"type": "task_started", "id": task.id, "title": task.title}

            def _run_task(task):
                return task.id, self.summarizer.summarize(task, context)

            started_at = time.monotonic()
            stop_reason = None
            executor = ThreadPoolExecutor(max_workers=2)
            try:
                remaining = [task for task in plan if task.id not in notes_by_id]
                while remaining:
                    stop_reason = research_stop_reason(
                        notes_by_id, len(plan), time.monotonic() - started_at,
                        settings.RESEARCH_MAX_SECONDS, settings.RESEARCH_EARLY_STOP_ENABLED,
                    )
                    if stop_reason:
                        break
                    batch, remaining = remaining[:2], remaining[2:]
                    pending = {executor.submit(_run_task, task) for task in batch}
                    while pending:
                        if self._cancelled(context):
                            yield {"type": "cancelled"}
                            return
                        finished, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                        for future in finished:
                            task_id, note = future.result()
                            notes_by_id[task_id] = note
                            state["notes"][str(task_id)] = asdict(note)
                            if store:
                                store.save(state)
                            yield {"type": "task_done", "id": task_id, "citations": len(note.citations)}
            finally:
                executor.shutdown(wait=not self._cancelled(context), cancel_futures=True)

            stop_reason = stop_reason or research_stop_reason(
                notes_by_id, len(plan), time.monotonic() - started_at,
                settings.RESEARCH_MAX_SECONDS, settings.RESEARCH_EARLY_STOP_ENABLED,
            ) or "plan_complete"

            notes = [notes_by_id[t.id] for t in plan if t.id in notes_by_id]
            all_tools = [tool for note in notes for tool in note.tools_called]
            seen_urls = set()
            citations = []
            for citation in (c for note in notes for c in note.citations):
                key = citation.get("url") or citation.get("title", "")
                if key and key not in seen_urls:
                    seen_urls.add(key)
                    citations.append(citation)

            yield {"type": "generating"}
            parts = []
            with closing(self.writer.stream(message, notes)) as tokens:
                for token in tokens:
                    if self._cancelled(context):
                        yield {"type": "cancelled"}
                        return
                    parts.append(token)
                    yield {"type": "token", "value": token}
            unfinished = [t.title for t in plan if t.id not in notes_by_id]
            if unfinished:
                suffix = "\n\n## Unfinished research tasks\n" + "\n".join(f"- {title}" for title in unfinished)
                suffix += f"\n\nStopped because: {stop_reason}."
                parts.append(suffix)
                yield {"type": "token", "value": suffix}
            resumable = stop_reason == "time_budget" and bool(unfinished)
            state.update(status="paused" if resumable else "completed",
                         stop_reason=stop_reason, report="".join(parts),
                         tools=all_tools, citations=citations)
            if store:
                store.save(state)
            completed = True
            yield {"type": "done", "tools_called": all_tools, "citations": citations,
                   "stop_reason": stop_reason, "run_id": state["run_id"],
                   "resumable": resumable}
        finally:
            if store and not completed and state["status"] != "completed":
                state.update(status="paused", stop_reason="cancelled" if self._cancelled(context) else "interrupted")
                store.save(state)
