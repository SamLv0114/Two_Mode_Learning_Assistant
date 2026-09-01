"""
PlanAndSolveAgent — structured two-phase analytical reasoning.

Implements the Plan-and-Solve paradigm from Hello-Agents ch4.3.

Unlike DeepResearchAgent (which searches for unknown information),
PlanAndSolveAgent is designed for *analytical* questions where the answer
is derived by reasoning through a structured plan:

  "Compare transformer vs. LSTM for time-series forecasting"
  "What are the pros and cons of LoRA fine-tuning?"
  "When should I use RLHF vs. DPO?"

Two phases:
  Phase 1 — Plan  (gpt-4o):     Produce a numbered analysis blueprint
  Phase 2 — Solve (gpt-4o-mini): Execute each step with full prior context,
                                  optionally calling search tools for facts
  Synthesis     (gpt-4o):     Merge all step results into a structured response

The streaming path emits the same plan/task_started/task_done/token/done
events as DeepResearchAgent, so the frontend renders both identically.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.tools import SEARCH_KNOWLEDGE_BASE, SEARCH_WEB, execute_tool
from src.utils.config import settings

logger = logging.getLogger(__name__)

PLANNER_MODEL = "gpt-4o"
SOLVER_MODEL = "gpt-4o-mini"


@dataclass
class AnalysisStep:
    number: int
    step: str           # what to analyse in this step
    focus: str = ""     # analytical lens (e.g. "performance", "cost", "use-case")
    result: str = ""    # filled in during Phase 2


class PlanAndSolveAgent(BaseAgent):
    """
    Structured analytical agent using a plan-then-execute loop.

    Activated by the router for comparison / trade-off / when-to-use queries.
    """

    name = "PlanAndSolveAgent"
    system_prompt = "You are a structured analytical assistant for an ML research platform."
    tool_schemas = []   # each phase uses its own tools

    _PLAN_PROMPT = """\
You are an analytical planning assistant for a machine learning research platform.

Break the following analytical question into 3–5 structured analysis steps.
Each step should be a specific analytical action: define, compare, evaluate, analyse.

Question: {question}

Return ONLY a JSON array (no markdown):
[
  {{
    "number": 1,
    "step": "<what to analyse in this step>",
    "focus": "<analytical lens — e.g. performance, cost, use-case, trade-offs>"
  }},
  ...
]"""

    _SOLVE_PROMPT = """\
You are executing step {step_num} of {total} in a structured analysis.

Original question: {question}

Analysis plan:
{plan_summary}

Steps completed so far:
{completed}

Current step {step_num}: {step_description}
Focus: {focus}

Search results (use if relevant):
{search_results}

Execute this step concisely and precisely. Stay focused on the step's scope."""

    _SYNTHESIZE_PROMPT = """\
You are completing a structured analytical response.

Original question: {question}

All analysis steps:
{steps_and_results}

Write a comprehensive, well-structured Markdown answer with:
- ## headers for each major theme
- Specific technical comparisons and concrete examples
- A brief ## Conclusion summarising the key takeaway
Be precise and directly answer the original question."""

    def __init__(self):
        super().__init__()

    # ── Phase 1: Plan ─────────────────────────────────────────────────────────

    def _generate_plan(self, question: str) -> List[AnalysisStep]:
        prompt = self._PLAN_PROMPT.format(question=question[:800])
        try:
            resp = self.client.chat.completions.create(
                model=PLANNER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            steps = []
            for item in data[:5]:
                steps.append(AnalysisStep(
                    number=int(item.get("number", len(steps) + 1)),
                    step=str(item.get("step", "")),
                    focus=str(item.get("focus", "")),
                ))
            logger.info(f"[PlanAndSolveAgent] {len(steps)}-step plan for: {question[:60]}...")
            return steps
        except Exception as e:
            logger.warning(f"[PlanAndSolveAgent] Plan generation failed: {e}")
            return [AnalysisStep(number=1, step=question, focus="direct answer")]

    # ── Phase 2: Solve ────────────────────────────────────────────────────────

    def _execute_step(
        self,
        question: str,
        step: AnalysisStep,
        all_steps: List[AnalysisStep],
        completed: List[AnalysisStep],
        context: Dict[str, Any],
    ) -> str:
        plan_summary = "\n".join(f"{s.number}. {s.step}" for s in all_steps)
        completed_text = (
            "\n\n".join(f"Step {s.number} ({s.step}):\n{s.result}" for s in completed)
            or "None yet."
        )

        # Optional KB search to ground the step
        search_text = ""
        try:
            kb_args = {"query": f"{question} {step.step}", "n_results": 3}
            t0 = time.time()
            kb = execute_tool("search_knowledge_base", kb_args, context)
            self._fire_listener("search_knowledge_base", kb_args, kb,
                                int((time.time() - t0) * 1000), context)
            snippets = [f"[{r.get('title','')}] {r.get('content','')[:300]}" for r in kb.get("results", [])]
            search_text = "\n".join(snippets[:3]) or "No relevant results found."
        except Exception:
            search_text = "Search unavailable."

        prompt = self._SOLVE_PROMPT.format(
            step_num=step.number,
            total=len(all_steps),
            question=question[:500],
            plan_summary=plan_summary,
            completed=completed_text[:1200],
            step_description=step.step,
            focus=step.focus or "general analysis",
            search_results=search_text[:600],
        )

        try:
            resp = self.client.chat.completions.create(
                model=SOLVER_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"[PlanAndSolveAgent] Step {step.number} failed: {e}")
            return f"Could not complete step {step.number}."

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, question: str, steps: List[AnalysisStep]) -> str:
        steps_text = "\n\n".join(
            f"**Step {s.number}: {s.step}**\n{s.result}" for s in steps
        )
        try:
            resp = self.client.chat.completions.create(
                model=PLANNER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior ML research writer."},
                    {"role": "user", "content": self._SYNTHESIZE_PROMPT.format(
                        question=question,
                        steps_and_results=steps_text[:4000],
                    )},
                ],
                max_tokens=1500,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[PlanAndSolveAgent] Synthesis failed: {e}")
            return steps_text

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        start = time.time()

        plan = self._generate_plan(message)
        completed: List[AnalysisStep] = []

        for step in plan:
            step.result = self._execute_step(message, step, plan, completed, context)
            completed.append(step)

        final = self._synthesize(message, plan)
        tools_called = ["search_knowledge_base"] * len(plan)  # each step searches

        return AgentResult(
            reply=final,
            tools_called=tools_called,
            citations=[],
            processing_time_ms=int((time.time() - start) * 1000),
        )

    def stream(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Generator:
        plan = self._generate_plan(message)

        yield {
            "type": "plan",
            "tasks": [{"id": s.number, "title": s.step, "intent": s.focus} for s in plan],
        }

        completed: List[AnalysisStep] = []
        for step in plan:
            yield {"type": "task_started", "id": step.number, "title": step.step}
            step.result = self._execute_step(message, step, plan, completed, context)
            completed.append(step)
            yield {"type": "task_done", "id": step.number, "citations": 0}

        yield {"type": "generating"}

        steps_text = "\n\n".join(
            f"**Step {s.number}: {s.step}**\n{s.result}" for s in plan
        )
        try:
            stream = self.client.chat.completions.create(
                model=PLANNER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior ML research writer."},
                    {"role": "user", "content": self._SYNTHESIZE_PROMPT.format(
                        question=message,
                        steps_and_results=steps_text[:4000],
                    )},
                ],
                max_tokens=1500,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield {"type": "token", "value": token}
        except Exception as e:
            logger.error(f"[PlanAndSolveAgent] Stream synthesis failed: {e}")
            for char in steps_text:
                yield {"type": "token", "value": char}

        yield {"type": "done", "tools_called": ["search_knowledge_base"] * len(plan), "citations": []}
