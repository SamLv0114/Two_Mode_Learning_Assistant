"""LangGraph workflows for a paper implementation and repository exploration.

The graph knows nothing about HTTP or database users. Callers supply an
authorized run ID and a service object; graph state contains only serializable
data and is persisted by the caller's checkpointer.
"""
import ast
import builtins
import difflib
import hashlib
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents import method_notes

logger = logging.getLogger(__name__)

MAX_PAPER_EXECUTIONS = 5
MAX_REVIEW_FIXES = 1      # static evidence review may trigger one targeted revision
MAX_GATHER_ROUNDS = 2     # targeted evidence searches for details the first reading left open
MAX_CODE_BYTES = 20_000
# Third-party packages a teaching program may import. The isolated runner image
# must provide exactly these (see sandbox_worker/README.md).
ALLOWED_THIRD_PARTY = frozenset({"numpy"})


class PracticeError(Exception):
    pass


class PracticeCancelled(PracticeError):
    pass


class PaperState(TypedDict, total=False):
    paper_id: str
    objective: str
    execution_enabled: bool
    title: str
    source_url: str
    evidence: List[Dict[str, str]]
    evidence_source: str
    abstract: str
    code: str
    explanation: str
    attempts: int
    execution: Dict[str, Any]
    previous_hash: str
    previous_error: str
    stop_reason: str
    questions: List[str]
    manifest: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    pending_repair: Dict[str, Any]
    faithfulness: Dict[str, Any]
    detail_review: Dict[str, Any]
    draft_revision: Dict[str, Any]
    linked_repo: Dict[str, str]
    confirm_plan: bool                # pause after the method check for the learner's decisions
    plan: Dict[str, Any]              # those decisions: answered choices, simplifications, instructions
    gather_rounds: int
    gathered: List[Dict[str, Any]]    # what each evidence search looked for and found
    repo_evidence: Dict[str, Any]     # the repository agent's answer behind any code (C) excerpts
    personalized: bool
    method_notes: List[str]           # method-family notes used in generation and review (agents/method_notes)


class RepoState(TypedDict, total=False):
    repo_url: str
    commit_sha: str
    file_tree: List[str]
    readme: str
    files: List[Dict[str, str]]
    tree_truncated: bool
    architecture: str
    entry_points: List[str]
    file_roles: List[Dict[str, str]]
    limitations: str
    linked_paper: Dict[str, str]
    focus: str
    overview: Dict[str, Any]
    tour: List[Dict[str, Any]]


def _check_code(code: str) -> None:
    _check_size(code)
    compile(code, "practice.py", "exec")


def _check_size(code: str) -> None:
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise PracticeError("Python code must be nonempty and at most 20 KB")


def _draft_issue(code: str) -> str:
    """Static draft check only; it never executes or verifies the algorithm."""
    try:
        tree = ast.parse(code, filename="practice.py")
    except SyntaxError as exc:
        return f"Syntax error on line {exc.lineno}: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return "Relative imports are unavailable in a single-file program"
            modules = [(node.module or "").split(".", 1)[0]]
        else:
            continue
        for module in modules:
            if module not in sys.stdlib_module_names and module not in ALLOWED_THIRD_PARTY:
                allowed = ", ".join(sorted(ALLOWED_THIRD_PARTY))
                return f"Dependency '{module}' is unavailable in the runner (standard library and {allowed} only)"
    return _undefined_module_name(tree)


def _static_repair(services, state: Dict[str, Any], draft: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Static-check a draft and allow the model one revision when it fails.
    Returns the draft to keep and the issue still open ('' when it passes).
    A failed revision keeps the original draft and its issue.
    """
    _check_size(draft.get("code") or "")
    issue = _draft_issue(draft["code"])
    if not issue:
        return draft, ""
    try:
        revised = services.revise_paper_draft(state, draft, issue)
        if revised.get("code"):
            _check_size(revised["code"])
            draft, issue = revised, _draft_issue(revised["code"])
    except PracticeCancelled:
        raise
    except Exception as exc:
        logger.warning("Static-check revision failed, keeping the draft: %s", exc)
    return draft, issue



_COMPARE_CALLS = frozenset({"allclose", "array_equal", "array_equiv", "isclose", "assert_allclose",
                            "assert_array_equal", "assert_array_almost_equal", "assert_equal"})
_RANDOM_SOURCES = ("np.random.", "numpy.random.", "random.", "rng.", "generator.")


def quality_issues(code: str) -> List[str]:
    """
    Deterministic checks for two ways a draft can look right and teach nothing: a self-check
    that compares an expression with itself (it always passes), and "gradients" that are
    random numbers instead of being derived from the loss. Unlike _draft_issue these do not
    stop the run: the review passes them to the one targeted revision with its own findings.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    issues: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in _COMPARE_CALLS and len(node.args) >= 2 and ast.dump(node.args[0]) == ast.dump(node.args[1]):
                issues.append(f"Line {node.lineno}: the self-check {name}() compares {ast.unparse(node.args[0])} "
                              "with itself, so it always passes; compare with a copy saved before training or "
                              "with a value computed another way")
        elif (isinstance(node, ast.Compare) and len(node.comparators) == 1 and isinstance(node.ops[0], ast.Eq)
              and ast.dump(node.left) == ast.dump(node.comparators[0])):
            issues.append(f"Line {node.lineno}: '{ast.unparse(node)}' compares an expression with itself, "
                          "so it always holds")
        elif isinstance(node, ast.Assign):
            # A gradient or loss whose formula draws fresh random numbers is noise, not optimisation:
            # a model asked to remove grad = np.random.randn(...) once moved the random call inside the
            # formula instead. Plain assignment only, so adding noise to a computed gradient
            # (grad += noise) stays legitimate; random data belongs in its own variable.
            targets = [ast.unparse(t) for t in node.targets]
            if not any(word in t.lower() for t in targets for word in ("grad", "loss")):
                continue
            sources = [ast.unparse(call.func) for call in ast.walk(node.value) if isinstance(call, ast.Call)]
            random_source = next((src for src in sources if src.startswith(_RANDOM_SOURCES)), None)
            if random_source:
                issues.append(f"Line {node.lineno}: {targets[0]} is computed from fresh random numbers "
                              f"({random_source}); compute it from the model's output and fixed training data, "
                              "so training really optimises the objective")
    return list(dict.fromkeys(issues))[:4]

_NEW_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _undefined_module_name(tree: ast.Module) -> str:
    """
    A name read at module level that nothing at module level ever binds, so the
    draft would stop with NameError (e.g. a variable that only exists inside a
    function). Function bodies are not checked; this only catches the plain case.
    """
    nodes, stack = [], list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if not isinstance(node, _NEW_SCOPES):  # their bodies are separate scopes
            stack.extend(ast.iter_child_nodes(node))
    bound = set(dir(builtins)) | {"__builtins__", "__file__"}
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".", 1)[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    for node in sorted((n for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)),
                       key=lambda n: n.lineno):
        if node.id not in bound:
            return f"Name '{node.id}' on line {node.lineno} is never defined at module level (NameError)"
    return ""


EVIDENCE_MARKER = re.compile(r"\[((?:E|C)\d+)\]")
MAX_BLANKS = 3


def _functions(tree: ast.AST) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def visit(nodes, prefix: str):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{node.name}"
                found.append({"name": name, "start": node.lineno, "end": node.end_lineno or node.lineno,
                              "kind": "class" if isinstance(node, ast.ClassDef) else "function"})
                visit(node.body, name + ".")
    visit(getattr(tree, "body", []), "")
    return found


def code_map(code: str) -> Dict[str, Any]:
    """Which evidence excerpt each '# [E3]' marker points at, and the function that line belongs to."""
    lines = (code or "").splitlines()
    try:
        functions = _functions(ast.parse(code))
    except (SyntaxError, ValueError):
        functions = []
    links = []
    for number, line in enumerate(lines, 1):
        if "#" not in line:
            continue
        ids = EVIDENCE_MARKER.findall(line.split("#", 1)[1])
        if not ids:
            continue
        targets = [number]
        if line.strip().startswith("#"):  # a comment above the code it describes
            following = next((n for n in range(number + 1, min(len(lines), number + 4) + 1)
                              if lines[n - 1].strip() and not lines[n - 1].strip().startswith("#")), None)
            if following:
                targets.append(following)
        inner = [f for f in functions if f["kind"] == "function" and f["start"] <= targets[-1] <= f["end"]]
        owner = min(inner, key=lambda f: f["end"] - f["start"])["name"] if inner else "(module level)"
        for evidence_id in dict.fromkeys(ids):
            links.append({"evidence_id": evidence_id, "lines": targets, "function": owner})
    return {"functions": [f for f in functions if f["kind"] == "function"], "links": links}


def make_exercise(code: str, manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fill-in-the-blank version of the generated program: up to three single-line statements that
    implement cited paper details are blanked (an assignment keeps its target). Answers stay on
    the server; the learner sees the template, hints and the evidence each blank comes from.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    lines = code.splitlines()
    statements = {node.lineno: node for node in ast.walk(tree)
                  if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return, ast.Expr))
                  and node.lineno == node.end_lineno}
    details = manifest.get("known_details", [])
    chosen: List[Tuple[int, str]] = []
    used_ids = set()
    links = code_map(code)["links"]
    for prefer_new_id in (True, False):
        for link in links:
            line = link["lines"][-1]
            node = statements.get(line)
            if node is None or any(line == c[0] for c in chosen) or len(chosen) == MAX_BLANKS:
                continue
            if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call):
                continue  # docstrings and bare names are not worth a blank
            if prefer_new_id and link["evidence_id"] in used_ids:
                continue
            chosen.append((line, link["evidence_id"]))
            used_ids.add(link["evidence_id"])
    if not chosen:
        return None
    template, blanks, answers = list(lines), [], {}
    for index, (line, evidence_id) in enumerate(sorted(chosen), 1):
        node = statements[line]
        statement = ast.get_source_segment(code, node) or lines[line - 1].strip()
        value = getattr(node, "value", None)
        answer = ast.get_source_segment(code, value) if value is not None and not isinstance(node, ast.Expr) else None
        prefix = statement[:statement.rfind(answer)] if answer and statement.endswith(answer) else ""
        answer = answer if prefix else statement
        indent = lines[line - 1][:len(lines[line - 1]) - len(lines[line - 1].lstrip())]
        hint = _hint(details, evidence_id, prefix.rstrip("= +-*/").strip().split(".")[-1])
        template[line - 1] = f"{indent}{prefix}____  # TODO {index} ({evidence_id}): {hint}"
        blanks.append({"id": str(index), "line": line, "evidence_id": evidence_id, "hint": hint,
                       "prefix": prefix.strip()})
        answers[str(index)] = answer
    return {"template": "\n".join(template), "blanks": blanks, "answers": answers}


def _hint(details: List[Dict[str, Any]], evidence_id: str, target: str) -> str:
    """The cited detail that names the blanked variable, else the first detail citing that excerpt."""
    cited = [d["detail"] for d in details if evidence_id in d.get("evidence_ids", [])]
    named = [d for d in cited if target and re.search(rf"\b{re.escape(target)}\b", d)]
    return ((named or cited or [f"see {evidence_id}"])[0])[:120]


def fill_exercise(code: str, exercise: Dict[str, Any], given: Dict[str, str]) -> str:
    """The original program with each blank's line rebuilt from the learner's answer."""
    lines = code.splitlines()
    for blank in exercise["blanks"]:
        original = lines[blank["line"] - 1]
        indent = original[:len(original) - len(original.lstrip())]
        prefix = f"{blank['prefix']} " if blank["prefix"] and not blank["prefix"].endswith(" ") else blank["prefix"]
        lines[blank["line"] - 1] = f"{indent}{prefix}{str(given.get(blank['id'], '')).strip()}"
    return "\n".join(lines)


def _emit(kind: str, **data) -> None:
    """Progress event for the live activity feed; a no-op when the graph is not streamed."""
    get_stream_writer()({"kind": kind, **data})


def evidence_gaps(manifest: Dict[str, Any], include_choices: bool = True) -> List[str]:
    """
    What a further evidence search should look for: open details, a refusal reason, and in the first
    round the open choices (the paper may state them after all; if not, the learner decides).
    """
    targets = list(manifest.get("unknown_details") or [])
    if include_choices:
        targets += [choice["detail"] for choice in manifest.get("open_choices") or []]
    if not manifest.get("feasible") and manifest.get("reason"):
        targets.append(manifest["reason"])
    unique: List[str] = []
    for target in targets:
        if isinstance(target, str) and target.strip() and target.strip() not in unique:
            unique.append(target.strip())
    return unique[:4]


def plan_from_decision(decision: Any, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The learner's decisions, bounded and matched to the choices the method check offered."""
    decision = decision if isinstance(decision, dict) else {}
    answers = decision.get("choices") if isinstance(decision.get("choices"), dict) else {}
    choices = []
    for choice in manifest.get("open_choices") or []:
        answer = str(answers.get(choice["id"]) or choice.get("recommended") or "").strip()[:200]
        if answer:
            choices.append({"id": choice["id"], "detail": choice["detail"], "question": choice["question"],
                            "answer": answer, "recommended": answer == choice.get("recommended")})
    simplifications = decision.get("simplifications")
    if not isinstance(simplifications, list):
        simplifications = manifest.get("simplifications") or []
    return {"choices": choices,
            "simplifications": [str(x).strip()[:200] for x in simplifications if str(x).strip()][:8],
            "instructions": str(decision.get("instructions") or "").strip()[:500]}


def build_paper_graph(services, checkpointer=None):
    """
    Evidence -> (targeted evidence search) -> learner confirms the plan -> generation ->
    static evidence review (one fix) -> draft or isolated execution/repair.
    """
    graph = StateGraph(PaperState)

    def evidence(state: PaperState):
        services.check_cancelled()
        _emit("searching_paper", paper_id=state["paper_id"])
        result = services.paper_evidence(state["paper_id"], state.get("objective", ""))
        if not result.get("evidence"):
            reason = result.get("error") or "No usable full-text method excerpts were found."
            _emit("evidence", found=0, reason=reason)
            return {"stop_reason": "method_detail_insufficient", "title": result.get("title", ""),
                    "source_url": result.get("source_url", ""),
                    "explanation": reason, "evidence": [], "linked_repo": result.get("linked_repo", {}),
                    "manifest": {"feasible": False, "method": "", "reason": reason,
                                 "known_details": [], "unknown_details": ["Full-text method evidence unavailable"]}}
        _emit("evidence", found=len(result["evidence"]), source=result.get("evidence_source", ""),
              title=result.get("title", ""), linked_repo=(result.get("linked_repo") or {}).get("url", ""))
        manifest = services.assess_paper_method(result, state.get("objective", ""))
        _emit("manifest", feasible=manifest["feasible"], method=manifest.get("method", ""),
              known=len(manifest.get("known_details", [])), unknown=manifest.get("unknown_details", []),
              choices=len(manifest.get("open_choices", [])))
        result["manifest"] = manifest
        # Chosen by code from the paper and the method check (deterministic), then kept in state
        # so generation, review and a resumed run all use the same notes.
        result["method_notes"] = [note.id for note in method_notes.select(
            result.get("title", ""), result.get("abstract", ""), manifest.get("method", ""), state.get("objective", ""))]
        if result["method_notes"]:
            _emit("method_notes", families=[note.name for note in method_notes.by_ids(result["method_notes"])])
        result["timeline"] = []
        result["stop_reason"] = "" if manifest["feasible"] else "method_detail_insufficient"
        if not manifest["feasible"]:
            result["explanation"] = manifest["reason"]
        return result

    def gather(state: PaperState):
        """Look again for what the method check left open: the paper's full text, then linked code."""
        services.check_cancelled()
        writer = get_stream_writer()
        return services.gather_evidence(state, on_event=lambda kind, **data: writer({"kind": kind, **data}))

    def after_evidence(state: PaperState):
        if not state.get("evidence"):
            return "finish"
        rounds = state.get("gather_rounds", 0)
        last_found = (state.get("gathered") or [{}])[-1].get("new_pieces", 1) if rounds else 1
        if evidence_gaps(state["manifest"], include_choices=not rounds) and rounds < MAX_GATHER_ROUNDS and last_found:
            return "gather"
        return "confirm" if state["manifest"]["feasible"] else "finish"

    def confirm(state: PaperState):
        """Human in the loop: the learner approves or edits the plan before any code is written."""
        if not state.get("confirm_plan"):
            return {}
        decision = interrupt({"manifest": state["manifest"]})  # the runtime marks the run 'waiting'
        plan = plan_from_decision(decision, state["manifest"])
        manifest = {**state["manifest"], "simplifications": plan["simplifications"]}
        _emit("plan_confirmed", choices=len(plan["choices"]), instructions=bool(plan["instructions"]))
        return {"plan": plan, "manifest": manifest}

    def generate(state: PaperState):
        services.check_cancelled()
        _emit("generating", method=state["manifest"].get("method", ""))
        draft = services.generate_paper(state)
        if not draft.get("code"):
            return {"stop_reason": "method_detail_insufficient", "explanation": draft.get("explanation", "")}
        draft, issue = _static_repair(services, state, draft)
        if issue:
            _emit("static_check", passed=False, issue=issue)
            return {"code": draft["code"], "stop_reason": "draft_validation_failed",
                    "explanation": f"Generated draft failed a static check: {issue}. It was not run."}
        _emit("draft", lines=len(draft["code"].splitlines()))
        return {"code": draft["code"], "explanation": draft.get("explanation", ""),
                "personalized": bool(getattr(services, "personalized", False))}

    def review(state: PaperState):
        services.check_cancelled()
        result = services.review_draft(state)
        result["round"] = (state.get("detail_review") or {}).get("round", 0) + 1
        counts: Dict[str, int] = {}
        for check in result["checks"]:
            counts[check["status"]] = counts.get(check["status"], 0) + 1
        _emit("review", round=result["round"], issues=result["issue_count"], counts=counts,
              core_ok=result["core_mechanism"]["implemented"])
        return {"detail_review": result}

    def after_review(state: PaperState):
        detail_review = state["detail_review"]
        if detail_review["issue_count"] and detail_review["round"] <= MAX_REVIEW_FIXES:
            return "fix"
        return "execute" if state.get("execution_enabled", True) else "finish"

    def fix(state: PaperState):
        services.check_cancelled()
        before = state["code"]
        revision = {"issues": [c["detail"] for c in state["detail_review"]["checks"]
                               if c["status"] in ("contradicted", "missing")]
                    + ([] if state["detail_review"]["core_mechanism"]["implemented"]
                       else ["Core mechanism was a placeholder"])
                    + list(state["detail_review"].get("static_issues") or []),
                    "accepted": False, "diff": "", "reason": ""}
        try:
            draft, issue = _static_repair(services, state, services.fix_draft(state, state["detail_review"]))
            code = draft["code"]
        except PracticeCancelled:
            raise
        except Exception as exc:  # a failed fix keeps the reviewed draft
            code, issue = "", str(exc)[:200]
        if issue:
            revision["reason"] = f"Revision rejected by the static check: {issue}"
            _emit("revision", accepted=False, issues=revision["issues"][:4])
            return {"draft_revision": revision}
        _emit("revision", accepted=True, issues=revision["issues"][:4])
        revision.update(accepted=True, diff="".join(difflib.unified_diff(
            before.splitlines(True), code.splitlines(True), fromfile="draft.py", tofile="revised.py"))[:12000])
        return {"code": code, "draft_revision": revision,
                "explanation": draft.get("explanation") or state.get("explanation", "")}

    def after_fix(state: PaperState):
        if (state.get("draft_revision") or {}).get("accepted"):
            return "review"  # record how the revised draft now matches the evidence
        return "execute" if state.get("execution_enabled", True) else "finish"

    def execute(state: PaperState):
        services.check_cancelled()
        try:
            _check_code(state["code"])
        except SyntaxError as exc:
            result = {"passed": False, "stdout": "", "stderr": f"Syntax error: {exc.msg} (line {exc.lineno})"}
        else:
            result = services.execute(state["code"])
        error = str(result.get("stderr") or result.get("error") or "")[:500]
        code_hash = hashlib.sha256(state["code"].encode()).hexdigest()
        repeated = (code_hash == state.get("previous_hash", "")
                    or bool(error and error == state.get("previous_error", "")))
        _emit("execution", attempt=state.get("attempts", 0) + 1, passed=bool(result.get("passed")))
        pending = state.get("pending_repair") or {}
        timeline = list(state.get("timeline") or [])
        timeline.append({"attempt": state.get("attempts", 0) + 1,
                         "diagnosis": pending.get("diagnosis", "Initial implementation"),
                         "change_basis": pending.get("change_basis", []),
                         "diff": pending.get("diff", "Initial generated version"),
                         "execution": result})
        return {"execution": result, "attempts": state.get("attempts", 0) + 1,
                "timeline": timeline, "pending_repair": {},
                "previous_hash": code_hash,
                "previous_error": error,
                "stop_reason": "repeated_failure" if repeated and not result.get("passed") else ""}

    def next_step(state: PaperState):
        if state["execution"].get("passed") or state.get("stop_reason"):
            return "finish"
        if state["attempts"] >= MAX_PAPER_EXECUTIONS:
            return "finish"
        return "repair"

    def repair(state: PaperState):
        services.check_cancelled()
        draft = services.repair_paper(state)
        _check_size(draft["code"])
        diff = "".join(difflib.unified_diff(state["code"].splitlines(True),
                      draft["code"].splitlines(True), fromfile="previous.py", tofile="repaired.py"))
        return {"code": draft["code"], "explanation": draft.get("explanation", state.get("explanation", "")),
                "pending_repair": {"diagnosis": draft.get("diagnosis", "No diagnosis supplied"),
                                   "change_basis": draft.get("change_basis", []), "diff": diff[:12000]}}

    def finish(state: PaperState):
        reason = state.get("stop_reason") or (
            "generated_unexecuted" if not state.get("execution_enabled", True)
            else "success" if state.get("execution", {}).get("passed") else "max_attempts")
        result: PaperState = {"stop_reason": reason}
        _emit("finished", stop_reason=reason)
        if state.get("code"):
            try:
                result["faithfulness"] = services.assess_faithfulness(state)
            except Exception:
                result["faithfulness"] = {"status": "UNCERTAIN", "reason": "Evidence review unavailable"}
            try:
                services.check_cancelled()
                result["questions"] = services.program_questions(state)
            except Exception:
                # A failed optional teaching prompt must not erase the generated artifact.
                result["questions"] = []
        return result

    graph.add_node("evidence", evidence)
    graph.add_node("gather", gather)
    graph.add_node("confirm", confirm)
    graph.add_node("generate", generate)
    graph.add_node("review", review)
    graph.add_node("fix", fix)
    graph.add_node("execute", execute)
    graph.add_node("repair", repair)
    graph.add_node("finish", finish)
    graph.add_edge(START, "evidence")
    graph.add_conditional_edges("evidence", after_evidence, ["gather", "confirm", "finish"])
    graph.add_conditional_edges("gather", after_evidence, ["gather", "confirm", "finish"])
    graph.add_edge("confirm", "generate")
    graph.add_conditional_edges("generate", lambda s: "finish" if s.get("stop_reason") else "review")
    graph.add_conditional_edges("review", after_review, ["fix", "execute", "finish"])
    graph.add_conditional_edges("fix", after_fix, ["review", "execute", "finish"])
    graph.add_conditional_edges("execute", next_step)
    graph.add_edge("repair", "execute")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def build_repo_graph(services, checkpointer=None):
    """Pin a public repository snapshot and explain its structure from read-only evidence."""
    graph = StateGraph(RepoState)

    def prepare(state: RepoState):
        services.check_cancelled()
        _emit("pinning", repo_url=state["repo_url"])
        source = services.repo_source(state["repo_url"])
        overview = source.get("overview") or {}
        _emit("snapshot", commit=source.get("commit_sha", "")[:12], files=overview.get("total_files", len(source.get("file_tree", []))),
              searchable=overview.get("text_files", 0), focus=source.get("focus", ""))
        _emit("overview", entry_points=[e["path"] for e in overview.get("entry_points", [])][:5],
              core_modules=[c["path"] for c in overview.get("central_modules", [])][:5])
        return source

    def explain(state: RepoState):
        services.check_cancelled()
        _emit("explaining", files=[item["path"] for item in state.get("files", [])])
        result = services.explain_repo(state)
        _emit("finished", tour_stops=len(result.get("tour", [])))
        return result

    graph.add_node("prepare", prepare)
    graph.add_node("explain", explain)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "explain")
    graph.add_edge("explain", END)
    return graph.compile(checkpointer=checkpointer)
