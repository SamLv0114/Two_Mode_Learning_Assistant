"""Deterministic graph, authorization, and sandbox boundary checks (no API cost)."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from sandbox_worker.app import docker_command
from src.agents.coding_practice import build_paper_graph, build_repo_graph
from src.agents.coding_services import CodingServices, WorkerClient
from src.api.routers.practice import _owned_run, _public_state
from src.agents.coding_runtime import run_comparison_job, run_graph_job
from src.database.models import Base, PracticeRun, User
from src.utils.config import settings


CLEAN_REVIEW = {"checks": [{"detail": "A step", "evidence_ids": ["E1"], "status": "implemented",
                             "code_lines": "1", "note": ""}],
                "core_mechanism": {"implemented": True, "note": ""}, "issue_count": 0,
                "review_type": "model_static_review"}


class PaperServices:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reviews = []      # queued review results; CLEAN_REVIEW once exhausted
        self.review_calls = 0
        self.fixed_code = "print('CHECK:PASS')\nprint('fixed')"

    def review_draft(self, state):
        self.review_calls += 1
        return self.reviews.pop(0) if self.reviews else CLEAN_REVIEW

    def fix_draft(self, state, review):
        return {"code": self.fixed_code, "explanation": "Fixed to match evidence"}

    def gather_evidence(self, state, on_event=None):
        rounds = state.get("gather_rounds", 0) + 1
        return {"gather_rounds": rounds, "gathered": list(state.get("gathered") or []) + [{"round": rounds, "new_pieces": 0}]}

    def check_cancelled(self):
        return None

    def paper_evidence(self, paper_id, objective=''):
        return {"title": "Test paper", "evidence": [{"id": "E1", "text": "A method", "source": "arxiv"}]}

    def assess_paper_method(self, evidence, objective):
        return {"feasible": True, "method": "Example method", "reason": "Enough detail",
                "known_details": [{"detail": "A step", "evidence_ids": ["E1"]}], "unknown_details": []}

    def generate_paper(self, state):
        return {"code": "print('CHECK:PASS')", "explanation": "Example"}

    def repair_paper(self, state):
        return {"code": f"print('CHECK:PASS')\nprint({self.calls})", "diagnosis": "Missing output",
                "change_basis": [{"change": "Print result", "evidence_ids": [], "kind": "engineering_completion"}]}

    def execute(self, code):
        self.calls += 1
        return {"passed": self.calls == 2, "stdout": "CHECK:PASS" if self.calls == 2 else "",
                "stderr": "boom" if self.calls == 1 else ""}

    def program_questions(self, state):
        return ["Which function implements the method?", "What does the self-check establish?"]

    def assess_faithfulness(self, state):
        return {"status": "UNCERTAIN", "reason": "Mock evidence is insufficient"}


class RepoServices:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def check_cancelled(self):
        return None
    def repo_source(self, url):
        return {"repo_url": url, "commit_sha": "a" * 40, "file_tree": ["README.md", "core.py"],
                "readme": "A small library", "files": [{"path": "core.py", "content": "def twice(x): return x * 2"}]}

    def explain_repo(self, state):
        return {"architecture": "core.py contains one pure function", "entry_points": ["core.py"],
                "file_roles": [{"path": "core.py", "role": "Core logic"}], "limitations": "Only one file read"}


ABS_PAGE = (
    '<meta name="citation_title" content="Low-Rank Adapters for Teaching" />'
    '<meta name="citation_abstract" content="We propose low-rank adapters that freeze W and learn B A updates." />'
)

LATEX_PAPER = r"""
\newcommand{\methodac}{LRA}
\begin{document}
\section{Introduction}
Fine-tuning large models is costly. Prior work uses adapters.
\section{Related Work}
Adapters were studied by many authors [1, 2, 3] et al.
\section{Method}
\methodac{} freezes the pretrained weight $W_0$ and learns a low-rank update.
\begin{equation} h = W_0 x + B A x \end{equation}
where $B \in \mathbb{R}^{d \times r}$ is initialised to zero and $A$ is Gaussian.
\begin{algorithm}
\caption{\methodac{} update} Initialise A, B. For each step: update A and B only.
\end{algorithm}
\section*{Acknowledgments}
We thank the funding agency for low-rank support.
\begin{thebibliography}{9}
\bibitem{x} A. Author. Low-rank adapters. In Proceedings of a Conference, 2020.
\end{thebibliography}
\end{document}
""".replace("\\section*{Acknowledgments}", "\n\n".join(
    "Additional method text about low rank adapters and update rules." * 3 for _ in range(30))
    + "\n\n\\section*{Acknowledgments}")


class PaperEvidenceTests(unittest.TestCase):
    def services(self):
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = None
        services = CodingServices(db, embedding_manager=None, worker=Mock(), client=Mock())
        services._public_page = lambda url, limit=800_000: ABS_PAGE
        return services

    def test_latex_source_is_preferred_and_ranked_toward_the_method(self):
        services = self.services()
        with patch.object(services, '_latex_source', return_value=LATEX_PAPER):
            result = services.paper_evidence('2106.09685', 'understand the update rule')
        kinds = [item['kind'] for item in result['evidence']]
        self.assertEqual(result['evidence_source'], 'latex')
        self.assertEqual(kinds[0], 'abstract')
        self.assertEqual(set(kinds[1:]), {'latex'})
        texts = "\n".join(item['text'] for item in result['evidence'])
        self.assertIn('h = W_0 x + B A x', texts)
        self.assertIn('LRA freezes', texts)            # argument-free macro expanded
        self.assertNotIn('In Proceedings', texts)       # bibliography dropped
        self.assertNotIn('funding agency', texts)       # acknowledgements dropped
        self.assertEqual(result['evidence'][1]['chunk_index'], 'Algorithm')

    def test_known_abstract_is_not_repeated_and_details_get_reserved_pieces(self):
        services = self.services()
        paper = LATEX_PAPER.replace("\\section{Introduction}", "\\title{LRA}\\author{A. Writer}\\maketitle\n\n"
                                    "\\begin{abstract}Our LRA abstract text.\\end{abstract}\n\\section{Introduction}")
        paper = paper.replace("\\section{Related Work}", "\\section{Setup}\nOptimiser: Gaussian initialization, "
                              "zero bias, learning rate 0.01.\n\\section{Related Work}")
        with patch.object(services, '_latex_source', return_value=paper):
            result = services.paper_evidence('2106.09685')
        texts = "\n".join(item['text'] for item in result['evidence'][1:])
        self.assertNotIn('Our LRA abstract text', texts)   # E1 already is the abstract
        self.assertNotIn('A. Writer', texts)
        self.assertIn('learning rate 0.01', texts)         # shares no words with the contribution query

    def test_sentence_after_a_display_equation_stays_with_it(self):
        from src.agents.coding_services import _latex_chunks
        tex = ("\\begin{document}\\section{Method}\n" + "Low rank adapters update weights. " * 40
               + "\n\\begin{equation} h = W_0 x + BAx \\end{equation}\n\n"
               "We initialise $B$ to zero so the update starts at zero.\n\n" + "More text here. " * 60
               + "\\end{document}")
        first = _latex_chunks(tex)[0]["text"]
        self.assertIn("\\end{equation}\n\nWe initialise $B$ to zero", first)

    def test_source_archive_is_read_in_memory_and_includes_are_inlined(self):
        import io, tarfile
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, text in (("main.tex", "\\begin{document}\\input{sections/method}\\end{document}"),
                               ("sections/method.tex", "\\section{Method} The update is $h=W_0x+BAx$.")):
                data = text.encode()
                info = tarfile.TarInfo(name); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        services = self.services()
        with TemporaryDirectory() as folder, patch.object(settings, 'DATA_DIR', Path(folder)), \
             patch.object(services, '_stream_latex', return_value=None), \
             patch.object(services, '_download', return_value=buffer.getvalue()):
            self.assertIn("The update is $h=W_0x+BAx$", services._latex_source('2106.09685'))

    def test_large_tar_sources_are_streamed_keeping_only_tex(self):
        import io, tarfile
        from src.agents.coding_services import _ByteStream, _latex_from_tar_stream
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, data in (("figures/video.mp4", bytes(range(256)) * 20000),
                               ("main.tex", b"\\begin{document}The method.\\end{document}")):
                info = tarfile.TarInfo(name); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        body = buffer.getvalue()
        chunks = [body[i:i + 4096] for i in range(0, len(body), 4096)]
        files = _latex_from_tar_stream(_ByteStream(iter(chunks[1:]), len(body), chunks[0]))
        self.assertEqual(list(files), ["main.tex"])                      # the video is skipped
        with self.assertRaises(OSError):                                 # and the byte cap holds
            _latex_from_tar_stream(_ByteStream(iter(chunks[1:]), len(body) // 2, chunks[0]))

    def test_parsed_latex_is_cached_on_disk_across_runs(self):
        import os, time
        from src.agents import coding_services
        with TemporaryDirectory() as folder, patch.object(settings, 'DATA_DIR', Path(folder)), \
             patch.object(CodingServices, '_stream_latex', return_value=None):
            first, second = self.services(), self.services()
            source = b"\\begin{document}Body\\end{document}"
            import gzip
            with patch.object(first, '_download', return_value=gzip.compress(source)) as fetch:
                self.assertIn("Body", first._latex_source('2106.09685v2'))
            with patch.object(second, '_download') as refetch:
                self.assertIn("Body", second._latex_source('2106.09685v2'))   # a new run reads the cache
            refetch.assert_not_called()
            fetch.assert_called_once()
            with patch.object(second, '_download', return_value=b"%PDF-1.5 ..."):
                self.assertEqual(second._latex_source('2401.00001'), "")       # PDF-only is remembered too
            with patch.object(second, '_download') as refetch:
                self.assertEqual(second._latex_source('2401.00001'), "")
            refetch.assert_not_called()
            with patch.object(second, '_download', return_value=b""):
                self.assertEqual(second._latex_source('2401.00002'), "")       # a failed download is not cached
            self.assertFalse((Path(folder) / "arxiv_sources" / "2401.00002.tex").exists())
            stale = Path(folder) / "arxiv_sources" / "2401.00001.tex"
            os.utime(stale, (time.time() - coding_services.LATEX_CACHE_TTL - 60,) * 2)
            with patch.object(second, '_download', return_value=gzip.compress(source)) as refetch:
                self.assertIn("Body", second._latex_source('2401.00001'))     # unversioned IDs expire
            refetch.assert_called_once()

    def test_pdf_wrapper_source_falls_back_to_pdf_without_references(self):
        services = self.services()
        services.embedding_manager = Mock()
        services.embedding_manager.collection.get.return_value = {'documents': [], 'metadatas': []}
        wrapper = "\\begin{document}\\includepdf[pages=1-last]{paper.pdf}\\end{document}"
        references = "[1] A. B. et al. In Proceedings of NeurIPS, 2020. [2] C. D. arXiv preprint 2021."
        with patch.object(services, '_latex_source', return_value=wrapper), \
             patch('src.agents.tools._exec_fetch_full_paper', return_value={'results': [
                 {'content': references, 'chunk_index': 90},
                 {'content': 'The update rule is theta = theta - lr * m / (sqrt(v) + eps).', 'chunk_index': 12}]}):
            result = services.paper_evidence('1412.6980')
        self.assertEqual(result['evidence_source'], 'pdf')
        self.assertEqual([item['chunk_index'] for item in result['evidence']], ['abstract', '12'])

    def test_no_body_evidence_means_no_evidence_even_with_an_abstract(self):
        services = self.services()
        with patch.object(services, '_latex_source', return_value=''):
            result = services.paper_evidence('2106.09685')  # no embedding manager -> no PDF path
        self.assertEqual(result['evidence'], [])
        self.assertIn('unavailable', result['error'])

    def test_queries_come_from_the_papers_own_contribution(self):
        from src.agents.coding_services import _evidence_queries
        queries = _evidence_queries("LoRA", "Fine-tuning is costly. We propose LoRA, which freezes weights. "
                                            "Results are strong.", "learn the update")
        self.assertEqual(queries, ["LoRA", "We propose LoRA, which freezes weights.", "learn the update"])

    def test_preflight_keeps_simplifications_and_uses_the_coding_model(self):
        services = self.services()
        calls = []

        def fake_json(system, user, max_tokens=2200, model=None):
            calls.append((system, model))
            return {"feasible": True, "method": "LoRA", "reason": "Toy-scale demo",
                    "known_details": [{"detail": "h = W0 x + BAx", "evidence_ids": ["E2"]},
                                      {"detail": "B starts at zero", "evidence_ids": ["E2"]}],
                    "unknown_details": [], "simplifications": ["Tiny random linear layer instead of GPT-3"]}

        services._json = fake_json
        manifest = services.assess_paper_method(
            {"title": "LoRA", "evidence": [{"id": "E1", "text": "abstract"}, {"id": "E2", "text": "method"}]}, "")
        self.assertTrue(manifest["feasible"])
        self.assertEqual(manifest["simplifications"], ["Tiny random linear layer instead of GPT-3"])
        self.assertIn("toy", calls[0][0])
        self.assertEqual(calls[0][1], settings.CODING_AGENT_MODEL)

    def test_preflight_keeps_only_quotes_that_occur_in_the_cited_excerpt(self):
        services = self.services()
        services._json = lambda system, user, max_tokens=2200, model=None: {
            "feasible": True, "method": "LoRA", "reason": "",
            "known_details": [
                {"detail": "B starts at zero", "evidence_ids": ["E2"], "quote": "zero for $B$,  so the update"},
                {"detail": "A is Gaussian", "evidence_ids": ["E2"], "quote": "A is drawn from a Gaussian"},
                {"detail": "Abstract claim", "evidence_ids": ["E1"], "quote": "zero for $B$, so the update"}],
            "unknown_details": [], "simplifications": []}
        evidence = [{"id": "E1", "text": "We freeze weights."},
                    {"id": "E2", "text": "We use a random Gaussian initialization for $A$ and zero for $B$,\nso the update starts at zero."}]
        details = services.assess_paper_method({"title": "LoRA", "evidence": evidence}, "")["known_details"]
        self.assertEqual(details[0]["quote"], "zero for $B$, so the update")  # whitespace-insensitive match
        self.assertNotIn("quote", details[1])  # invented wording
        self.assertNotIn("quote", details[2])  # real text, but not in the cited excerpt

    def test_cut_off_structured_output_is_retried_with_more_room(self):
        services = self.services()
        cut = Mock(); cut.choices = [Mock(finish_reason="length", message=Mock(content='{"code": "pri'))]
        done = Mock(); done.choices = [Mock(finish_reason="stop", message=Mock(content='{"code": "print(1)"}'))]
        cut.usage = done.usage = None
        services.client.chat.completions.create.side_effect = [cut, done]
        self.assertEqual(services._json("system", "user", max_tokens=1000), {"code": "print(1)"})
        limits = [call.kwargs["max_completion_tokens"] for call in services.client.chat.completions.create.call_args_list]
        self.assertEqual(limits, [1000, 2000])

    def test_a_refusal_is_rechecked_once_against_the_missing_steps(self):
        services = self.services()
        calls = []
        answers = [{"feasible": False, "method": "RAG", "reason": "Objective not specified",
                    "known_details": [], "unknown_details": ["Training loss formula"]},
                   {"feasible": True, "method": "RAG", "reason": "",
                    "known_details": [{"detail": "Minimise negative marginal log-likelihood", "evidence_ids": ["E2"]},
                                      {"detail": "Top-k marginalisation", "evidence_ids": ["E2"]}],
                    "unknown_details": [], "simplifications": []}]

        def fake_json(system, user, max_tokens=2200, model=None):
            calls.append(user)
            return answers[len(calls) - 1]

        services._json = fake_json
        evidence = {"title": "RAG", "evidence": [{"id": "E1", "text": "abstract"},
                                                 {"id": "E2", "text": "we minimize the negative marginal log-likelihood"}]}
        self.assertTrue(services.assess_paper_method(evidence, "")["feasible"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("Training loss formula", calls[0])
        self.assertIn("Training loss formula", calls[1])

    def test_a_confirmed_refusal_still_refuses_and_feasible_answers_are_not_rechecked(self):
        services = self.services()
        calls = []
        refusal = {"feasible": False, "method": "", "reason": "Survey paper", "known_details": [],
                   "unknown_details": ["No proposed mechanism"]}
        services._json = lambda system, user, max_tokens=2200, model=None: calls.append(user) or refusal
        evidence = {"title": "Survey", "evidence": [{"id": "E1", "text": "abstract"}]}
        self.assertFalse(services.assess_paper_method(evidence, "")["feasible"])
        self.assertEqual(len(calls), 2)

    def test_a_feasible_answer_with_too_few_cited_details_is_rechecked(self):
        services = self.services()
        calls = []
        thin = {"feasible": True, "method": "Attention", "reason": "Can be implemented",
                "known_details": [{"detail": "softmax(QK^T/sqrt(d))V", "evidence_ids": ["E1"]}],
                "unknown_details": [], "simplifications": []}
        services._json = lambda system, user, max_tokens=2200, model=None: calls.append(user) or thin
        self.assertFalse(services.assess_paper_method({"title": "T", "evidence": [{"id": "E1", "text": "x"}]}, "")["feasible"])
        self.assertEqual(len(calls), 2)
        self.assertIn("Only 1 detail(s)", calls[1])

    def test_numpy_is_allowed_but_other_third_party_imports_are_not(self):
        from src.agents.coding_practice import _draft_issue
        self.assertEqual(_draft_issue("import numpy as np\nprint(np.zeros(2))"), "")
        self.assertIn("torch", _draft_issue("import torch"))

    def test_static_check_catches_names_that_only_exist_inside_a_function(self):
        from src.agents.coding_practice import _draft_issue
        self.assertIn("'delta' on line 4", _draft_issue("def f(x):\n    delta = x * 2\n    return delta\nprint(delta)"))
        for fine in ("import numpy as np\nxs = [i * i for i in range(3)]\nprint(np.sum(xs), len(xs))",
                     "class A:\n    pass\ntry:\n    A()\nexcept Exception as e:\n    print(e)",
                     "for i in range(2):\n    total = i\nprint(total, __name__)",
                     "def g():\n    return later\nlater = 1\nprint(g())"):
            self.assertEqual(_draft_issue(fine), "", fine)

    def test_runner_image_installs_every_allowed_package(self):
        from src.agents.coding_practice import ALLOWED_THIRD_PARTY
        dockerfile = (Path(__file__).parents[1] / "sandbox_worker" / "runner.Dockerfile").read_text()
        for package in ALLOWED_THIRD_PARTY:
            self.assertRegex(dockerfile, rf"pip install[^\n]*\b{package}==")


CONTRADICTED_REVIEW = {
    "checks": [{"detail": "B is initialised to zero", "evidence_ids": ["E1"], "status": "contradicted",
                "code_lines": "3", "note": "random init"}],
    "core_mechanism": {"implemented": True, "note": ""}, "issue_count": 1, "review_type": "model_static_review"}


class DraftReviewTests(unittest.TestCase):
    def run_graph(self, services, execution_enabled=False):
        return build_paper_graph(services, InMemorySaver()).invoke(
            {"paper_id": "2106.09685", "attempts": 0, "execution_enabled": execution_enabled},
            {"configurable": {"thread_id": "review"}})

    def test_contradiction_is_fixed_once_and_the_revision_is_reviewed(self):
        services = PaperServices()
        services.reviews = [CONTRADICTED_REVIEW]
        state = self.run_graph(services)
        self.assertEqual(services.review_calls, 2)
        self.assertEqual(state["stop_reason"], "generated_unexecuted")
        self.assertIn("fixed", state["code"])
        self.assertTrue(state["draft_revision"]["accepted"])
        self.assertEqual(state["draft_revision"]["issues"], ["B is initialised to zero"])
        self.assertIn("+print('fixed')", state["draft_revision"]["diff"])
        self.assertEqual((state["detail_review"]["round"], state["detail_review"]["issue_count"]), (2, 0))
        self.assertEqual(services.calls, 0)  # still a draft: nothing executed

    def test_clean_review_needs_no_revision(self):
        services = PaperServices()
        state = self.run_graph(services)
        self.assertEqual(services.review_calls, 1)
        self.assertNotIn("draft_revision", state)
        self.assertEqual(state["detail_review"]["round"], 1)

    def test_revision_that_breaks_the_static_check_is_rejected(self):
        services = PaperServices()
        services.reviews = [CONTRADICTED_REVIEW]
        services.fixed_code = "import torch\nprint('CHECK:PASS')"
        state = self.run_graph(services)
        self.assertEqual(state["code"], "print('CHECK:PASS')")
        self.assertFalse(state["draft_revision"]["accepted"])
        self.assertIn("torch", state["draft_revision"]["reason"])
        self.assertEqual(services.review_calls, 1)  # nothing new to review

    def test_revision_with_a_syntax_error_gets_one_static_repair(self):
        services = PaperServices()
        services.reviews = [CONTRADICTED_REVIEW]
        services.fixed_code = "def f():\nreturn 1"
        services.revise_paper_draft = lambda state, draft, issue: {"code": "def f():\n    return 1\nprint('CHECK:PASS')"}
        state = self.run_graph(services)
        self.assertTrue(state["draft_revision"]["accepted"])
        self.assertIn("    return 1", state["code"])

    def test_only_one_fix_even_if_issues_remain(self):
        services = PaperServices()
        services.reviews = [CONTRADICTED_REVIEW, CONTRADICTED_REVIEW]
        state = self.run_graph(services)
        self.assertEqual(services.review_calls, 2)
        self.assertEqual(state["detail_review"]["issue_count"], 1)
        self.assertEqual(state["stop_reason"], "generated_unexecuted")

    def test_review_runs_before_execution_when_a_worker_exists(self):
        services = PaperServices()
        services.reviews = [CONTRADICTED_REVIEW]
        state = self.run_graph(services, execution_enabled=True)
        self.assertEqual(services.review_calls, 2)
        self.assertEqual(state["stop_reason"], "success")
        self.assertTrue(state["draft_revision"]["accepted"])
        # the repair diff starts from the reviewed revision, so execution began from it
        self.assertIn("-print('fixed')", state["timeline"][1]["diff"])

    def test_review_keeps_only_cited_checks_and_flags_placeholder_mechanisms(self):
        services = CodingServices(Mock(), client=Mock())
        calls = []

        def fake_json(system, user, max_tokens=2200, model=None):
            calls.append((user, model))
            return {"core_mechanism": {"implemented": False, "note": "seq2seq is one matrix product"},
                    "checks": [
                        {"detail": "B starts at zero", "evidence_ids": ["E2"], "status": "contradicted",
                         "code_lines": "3", "note": "random"},
                        {"detail": "Invented requirement", "evidence_ids": ["E9"], "status": "missing"},
                        {"detail": "Odd status", "evidence_ids": ["E2"], "status": "maybe"},
                        {"detail": "Scaling by alpha/r", "evidence_ids": ["E2"], "status": "implemented",
                         "code_quote": "   2| A = 1"},
                        {"detail": "Random B", "evidence_ids": ["E2"], "status": "implemented",
                         "code_quote": "B = np.random.randn(2)"},
                        {"detail": "W0 frozen", "evidence_ids": ["E2"], "status": "implemented",
                         "code_quote": "# W0 is frozen here"}]}

        services._json = fake_json
        review = services.review_draft({"manifest": {"method": "LoRA"},
                                        "evidence": [{"id": "E1", "text": "abs"}, {"id": "E2", "text": "B = 0"}],
                                        "code": "import numpy as np\nA = 1\nB = np.random.randn(2)  # [E2]"})
        self.assertEqual([c["detail"] for c in review["checks"]],
                         ["B starts at zero", "Scaling by alpha/r", "Random B", "W0 frozen"])
        self.assertEqual(review["checks"][1]["status"], "implemented")
        self.assertEqual(review["checks"][2]["status"], "implemented")  # quote without the line's trailing comment
        # a comment is not an implementation, whatever the reviewer claims
        self.assertEqual(review["checks"][3]["status"], "missing")
        self.assertEqual(review["issue_count"], 3)  # contradiction + comment-only claim + placeholder mechanism
        self.assertFalse(review["core_mechanism"]["implemented"])
        self.assertIn("   3| B = np.random.randn(2)", calls[0][0])  # reviewer sees line numbers
        self.assertEqual(calls[0][1], settings.CODING_AGENT_MODEL)


class CodingPracticeTests(unittest.TestCase):
    def test_public_arxiv_paper_outside_catalog_can_supply_evidence(self):
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = None
        services = CodingServices(db, embedding_manager=Mock(), worker=Mock(), client=Mock())
        with patch('src.agents.tools._exec_fetch_full_paper', return_value={
            'title': '2608.14407', 'results': [{'content': 'A method with implementation details', 'chunk_index': 0}],
        }), patch.object(services, 'paper_linked_repo', return_value={}), \
             patch.object(services, '_public_page', return_value=ABS_PAGE), \
             patch.object(services, '_latex_source', return_value=''):
            result = services.paper_evidence('2608.14407')
        self.assertEqual(result['source_url'], 'https://arxiv.org/abs/2608.14407')
        self.assertEqual(result['title'], 'Low-Rank Adapters for Teaching')
        self.assertEqual([item['kind'] for item in result['evidence']], ['abstract', 'pdf'])
        self.assertEqual(result['evidence_source'], 'pdf')

    def test_paper_evidence_includes_scoped_method_anchor(self):
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = None
        manager = Mock()
        manager.collection.get.return_value = {
            'documents': ['Algorithm 1: default settings are alpha = 0.001'],
            'metadatas': [{'chunk_index': 9}],
        }
        services = CodingServices(db, embedding_manager=manager, worker=Mock(), client=Mock())
        with patch('src.agents.tools._exec_fetch_full_paper', return_value={
            'results': [{'content': 'A general introduction', 'chunk_index': 1}],
        }), patch.object(services, 'paper_linked_repo', return_value={}), \
             patch.object(services, '_public_page', return_value=ABS_PAGE), \
             patch.object(services, '_latex_source', return_value=''):
            result = services.paper_evidence('1412.6980')
        self.assertEqual([item['chunk_index'] for item in result['evidence']], ['abstract', '9', '1'])
        self.assertEqual([item['id'] for item in result['evidence']], ['E1', 'E2', 'E3'])
        self.assertEqual(manager.collection.get.call_args.kwargs['where']['$and'][1],
                         {'arxiv_id': '1412.6980'})

    def test_public_arxiv_paper_can_start_without_catalog_row(self):
        from src.api.routers import practice
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with Session(engine) as db, patch.object(practice.settings, 'PRACTICE_ENABLED', True), \
             patch.object(practice.settings, 'OPENAI_API_KEY', 'test-key'), \
             patch.object(practice.settings, 'PRACTICE_WORKER_URL', None):
            user = User(id=1, email="hot@example.com", hashed_password="x")
            db.add(user)
            db.commit()
            result = practice.create_paper(practice.PaperRequest(arxiv_id="2608.14407"), user, db)
            saved = db.get(PracticeRun, result['run_id'])
            self.assertEqual(saved.status, 'queued')
            self.assertFalse(json.loads(saved.request_json)['execution_enabled'])

    def test_paper_repairs_once_and_stops(self):
        services = PaperServices()
        graph = build_paper_graph(services, InMemorySaver())
        state = graph.invoke({"paper_id": "2608.14407", "attempts": 0},
                             {"configurable": {"thread_id": "paper-1"}})
        self.assertEqual(services.calls, 2)
        self.assertEqual(state["stop_reason"], "success")
        self.assertEqual(state["attempts"], 2)
        self.assertEqual(len(state["questions"]), 2)
        self.assertEqual(len(state["timeline"]), 2)
        self.assertEqual(state["timeline"][1]["diagnosis"], "Missing output")
        self.assertIn("+print(1)", state["timeline"][1]["diff"])
        self.assertEqual(state["faithfulness"]["status"], "UNCERTAIN")

    def test_paper_draft_without_worker_never_executes(self):
        services = PaperServices()
        state = build_paper_graph(services, InMemorySaver()).invoke(
            {"paper_id": "2608.14407", "attempts": 0, "execution_enabled": False},
            {"configurable": {"thread_id": "paper-draft"}})
        self.assertEqual(services.calls, 0)
        self.assertEqual(state["stop_reason"], "generated_unexecuted")
        self.assertNotIn("execution", state)
        self.assertEqual(state["attempts"], 0)
        self.assertEqual(len(state["questions"]), 2)

    def test_paper_draft_rewrites_third_party_import_without_execution(self):
        services = PaperServices()
        services.generate_paper = lambda state: {"code": "import torch\nprint('CHECK:PASS')",
                                                 "explanation": "First draft"}
        revisions = []
        def revise(state, draft, issue):
            revisions.append(issue)
            return {"code": "import math\nprint(math.sqrt(4))\nprint('CHECK:PASS')",
                    "explanation": "Standard-library draft"}
        services.revise_paper_draft = revise
        state = build_paper_graph(services, InMemorySaver()).invoke(
            {"paper_id": "2608.14407", "attempts": 0, "execution_enabled": False},
            {"configurable": {"thread_id": "paper-draft-revised"}})
        self.assertEqual(services.calls, 0)
        self.assertEqual(state["stop_reason"], "generated_unexecuted")
        self.assertIn("torch", revisions[0])
        self.assertNotIn("torch", state["code"])

    def test_paper_method_preflight_refuses_before_generation(self):
        services = PaperServices()
        services.assess_paper_method = lambda evidence, objective: {
            "feasible": False, "method": "", "reason": "Essential steps are absent",
            "known_details": [], "unknown_details": ["update equation"]}
        state = build_paper_graph(services, InMemorySaver()).invoke(
            {"paper_id": "2608.14407", "attempts": 0},
            {"configurable": {"thread_id": "refuse"}})
        self.assertEqual(state["stop_reason"], "method_detail_insufficient")
        self.assertEqual(state["attempts"], 0)
        self.assertNotIn("code", state)

    def test_preflight_rejects_uncited_model_details(self):
        services = CodingServices(Mock(), client=Mock())
        with patch.object(services, '_json', return_value={
            "feasible": True, "method": "Some method", "reason": "Model says yes",
            "known_details": [{"detail": "Invented step", "evidence_ids": ["E99"]}],
            "unknown_details": []}):
            manifest = services.assess_paper_method(
                {"title": "Paper", "evidence": [{"id": "E1", "text": "A method"}]}, "")
        self.assertFalse(manifest["feasible"])
        self.assertEqual(manifest["known_details"], [])

    def test_faithfulness_pass_needs_reason_and_valid_evidence(self):
        services = CodingServices(Mock(), client=Mock())
        state = {"execution": {"passed": True}, "evidence": [{"id": "E1", "text": "Update rule"}],
                 "code": "print('CHECK:PASS')", "manifest": {"method": "Method"}}
        with patch.object(services, '_json', return_value={"status": "PASS", "reason": "",
                                                         "evidence_ids": ["E1"], "unsupported_choices": []}):
            self.assertEqual(services.assess_faithfulness(state)["status"], "UNCERTAIN")
        with patch.object(services, '_json', return_value={"status": "PASS", "reason": "Matches the cited update rule",
                                                         "evidence_ids": ["E99"], "unsupported_choices": []}):
            self.assertEqual(services.assess_faithfulness(state)["status"], "UNCERTAIN")
        with patch.object(services, '_json', return_value={"status": "PASS", "reason": "Matches the cited update rule",
                                                         "evidence_ids": ["E1"], "unsupported_choices": []}):
            self.assertEqual(services.assess_faithfulness(state)["status"], "PASS")

    def test_repair_basis_only_certifies_preflight_manifest_refs(self):
        services = CodingServices(Mock(), client=Mock())
        state = {"code": "print('old')", "execution": {"passed": False, "stderr": "error"},
                 "evidence": [{"id": "E1", "text": "Known step"}, {"id": "E2", "text": "Other text"}],
                 "manifest": {"known_details": [{"detail": "Known step", "evidence_ids": ["E1"]}]}}
        with patch.object(services, '_json', return_value={"code": "print('new')", "diagnosis": "Fix",
            "change_basis": [{"change": "Known update", "evidence_ids": ["E1"]},
                             {"change": "Other update", "evidence_ids": ["E2"]}]}):
            result = services.repair_paper(state)
        self.assertEqual(result["change_basis"][0]["kind"], "paper_supported")
        self.assertEqual(result["change_basis"][1]["kind"], "engineering_completion")
        self.assertEqual(result["change_basis"][1]["evidence_ids"], [])

    def test_arxiv_link_only_from_abstract_or_comments(self):
        services = CodingServices(Mock(), client=Mock())
        html = ('<nav><a href="https://github.com/site/navigation">Site</a></nav>'
                '<blockquote class="abstract mathjax">Code: '
                '<a href="https://github.com/team/method">repository</a></blockquote>')
        with patch.object(services, '_public_page', return_value=html):
            link = services.paper_linked_repo('2608.14407')
        self.assertEqual(link['url'], 'https://github.com/team/method')
        self.assertEqual(link['relationship'], 'linked_from_paper')
        self.assertEqual(link['source_section'], 'abstract')
        self.assertEqual(link['match_type'], 'exact_github_url')

    def test_paper_link_from_comments_is_labeled_separately(self):
        services = CodingServices(Mock(), client=Mock())
        html = ('<blockquote class="abstract">No code link here.</blockquote>'
                '<td class="comments">Code: <a href="https://github.com/team/method">GitHub</a></td>')
        with patch.object(services, '_public_page', return_value=html):
            link = services.paper_linked_repo('2608.14407')
        self.assertEqual(link['source_section'], 'comments')
        self.assertEqual(link['match_type'], 'exact_github_url')

    def test_readme_link_does_not_match_prefix_of_longer_arxiv_id(self):
        from src.agents.coding_services import ARXIV_LINK
        text = 'Other paper arxiv:2608.144079 and target arxiv:2608.14407v2'
        self.assertEqual([match.group(1) for match in ARXIV_LINK.finditer(text)], ['2608.14407'])

    def test_comparison_requires_reciprocal_reference_and_real_module(self):
        services = CodingServices(Mock(), client=Mock())
        paper = {"paper_id": "2608.14407", "linked_repo": {"url": "https://github.com/team/method"},
                 "manifest": {"method": "An algorithm"}, "evidence": [{"id": "E1", "text": "Step one"}],
                 "code": "print('teaching')"}
        repo = {"repo_url": "https://github.com/team/method", "commit_sha": "a" * 40,
                "file_tree": ["src/method.py"], "readme": "Paper arXiv:2608.14407", "files": [],
                "linked_paper": {"arxiv_id": "2608.14407", "relationship": "readme_reference"}}
        with patch.object(services, 'repo_source', return_value=repo), \
             patch.object(services, 'paper_linked_repo', return_value={}):
            with self.assertRaisesRegex(Exception, "reciprocally"):
                services.compare_paper_repo(paper)
        repo["linked_paper"]["relationship"] = "reciprocal"
        responses = iter([
            {"path": "not-in-tree.py", "reason": "guess"},
            {"path": "src/method.py", "reason": "README points here"},
            {"summary": "Two visible differences", "differences": [
                {"teaching": "simple loop", "repository": "vectorized loop", "paper_evidence_ids": ["E1"]},
                {"teaching": "unsupported", "repository": "claim", "paper_evidence_ids": ["E99"]}],
             "limitations": "Only one module inspected"},
        ])
        with patch.object(services, 'repo_source', return_value=repo), \
             patch.object(services, 'paper_linked_repo', return_value={'url': 'https://github.com/team/method'}), \
             patch.object(services, '_json', side_effect=lambda *args, **kwargs: next(responses)), \
             patch.object(services, '_raw_file', return_value="def implementation(): pass"):
            with self.assertRaisesRegex(Exception, "identify a corresponding module"):
                services.compare_paper_repo(paper)
            compared = services.compare_paper_repo(paper)
        self.assertEqual(compared["module_path"], "src/method.py")
        self.assertEqual(compared["selection_reason"], "README points here")
        self.assertEqual(len(compared["differences"]), 1)
        self.assertEqual(compared["differences"][0]["paper_evidence_ids"], ["E1"])
        self.assertEqual(compared["differences"][0]["reason"], "")

    def test_repo_exploration_finishes_with_pinned_structure(self):
        graph = build_repo_graph(RepoServices(), InMemorySaver())
        config = {"configurable": {"thread_id": "repo-1"}}
        state = graph.invoke({"repo_url": "https://github.com/example/project"}, config)
        self.assertEqual(state["commit_sha"], "a" * 40)
        self.assertEqual(state["entry_points"], ["core.py"])
        self.assertNotIn("exercises", _public_state("repo", state))

    def test_repository_source_excerpts_are_not_exposed_in_snapshot(self):
        state = {"file_tree": ["core.py"], "files": [{"path": "core.py", "content": "secret source"}]}
        exposed = _public_state("repo", state)
        self.assertNotIn("secret source", str(exposed))

    def test_run_ownership(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            db.add(User(id=1, email="first@example.com", hashed_password="x"))
            db.add(User(id=2, email="second@example.com", hashed_password="x"))
            db.add(PracticeRun(run_id="run-1", user_id=1, kind="paper", source="2608.14407"))
            db.commit()
            self.assertEqual(_owned_run(db, "run-1", 1).user_id, 1)
            with self.assertRaises(HTTPException) as raised:
                _owned_run(db, "run-1", 2)
            self.assertEqual(raised.exception.status_code, 404)

    def test_worker_requires_https_and_hardened_command(self):
        with self.assertRaises(Exception):
            WorkerClient(url="http://localhost:8090", token="x")
        with TemporaryDirectory() as temporary:
            command = docker_command(Path(temporary), "practice-test")
        for flag in ("--network", "none", "--read-only", "--cap-drop", "ALL",
                     "--pids-limit", "--memory", "--cpus", "--user"):
            self.assertIn(flag, command)
        self.assertNotIn("--privileged", command)

    def test_repo_url_is_public_github_only(self):
        self.assertEqual(CodingServices._repo_parts("https://github.com/example/repo"), ("example", "repo"))
        for url in ("https://github.com.evil.test/example/repo", "http://github.com/example/repo",
                    "https://github.com/example/repo?token=x"):
            with self.assertRaises(Exception):
                CodingServices._repo_parts(url)

    def test_repo_snapshot_pins_commit_and_reads_structure_without_worker(self):
        from src.agents.repo_snapshot import RepoSnapshot
        services = CodingServices(Mock(), worker=None, client=Mock())
        snapshot = RepoSnapshot("example", "project", "a" * 40, ["README.md", "src/core.py"],
                                {"README.md": "Overview", "src/core.py": "def run():\n    pass\n"})
        with patch.object(services, 'paper_linked_repo', return_value={}), \
             patch.object(services, '_github_json', side_effect=[
                 {'private': False, 'default_branch': 'main'}, {'sha': 'a' * 40}]), \
             patch('src.agents.coding_services.load_snapshot', return_value=snapshot) as load:
            result = services.repo_source('https://github.com/example/project')
        load.assert_called_once_with('example', 'project', 'a' * 40, '')
        self.assertEqual(result['commit_sha'], 'a' * 40)
        self.assertEqual(result['file_tree'], ['README.md', 'src/core.py'])
        self.assertEqual(result['files'][0]['path'], 'src/core.py')

    def test_repo_readme_paper_link_requires_reciprocal_arxiv_reference(self):
        from src.agents.repo_snapshot import RepoSnapshot
        services = CodingServices(Mock(), worker=None, client=Mock())
        snapshot = RepoSnapshot("example", "project", "a" * 40, ["README.md", "src/core.py"],
                                {"README.md": "See arXiv:2608.14407", "src/core.py": "def run(): pass\n"})
        with patch.object(services, '_github_json', side_effect=[
                 {'private': False, 'default_branch': 'main'}, {'sha': 'a' * 40}]), \
             patch('src.agents.coding_services.load_snapshot', return_value=snapshot), \
             patch.object(services, 'paper_linked_repo', return_value={
                 'url': 'https://github.com/example/project'}) as paper_link:
            result = services.repo_source('https://github.com/example/project')
        self.assertEqual(result['linked_paper']['relationship'], 'reciprocal')
        paper_link.assert_called_once_with('2608.14407')

    def test_repo_explanation_does_not_certify_unread_file_roles(self):
        services = CodingServices(Mock(), client=Mock())
        state = {'repo_url': 'https://github.com/example/project', 'commit_sha': 'a' * 40,
                 'file_tree': ['README.md', 'src/core.py', 'tests/test_core.py'],
                 'readme': 'Project overview',
                 'files': [{'path': 'src/core.py', 'content': 'def run(): pass'}]}
        with patch.object(services, '_json', return_value={
            'architecture': 'src/core.py defines run',
            'entry_points': ['src/core.py', 'tests/test_core.py'],
            'file_roles': [{'path': 'src/core.py', 'role': 'Core logic'},
                           {'path': 'tests/test_core.py', 'role': 'Full coverage'}],
            'limitations': 'The tests are insufficient'}):
            result = services.explain_repo(state)
        self.assertEqual(result['entry_points'], ['src/core.py'])
        self.assertEqual([item['path'] for item in result['file_roles']], ['src/core.py'])
        self.assertNotIn('insufficient', result['limitations'])

    def test_draft_and_repo_modes_do_not_need_execution_worker(self):
        from src.api.routers import practice
        with patch.object(practice.settings, 'PRACTICE_ENABLED', True), \
             patch.object(practice.settings, 'OPENAI_API_KEY', 'test-key'), \
             patch.object(practice.settings, 'PRACTICE_WORKER_URL', None):
            self.assertTrue(practice._available())
            self.assertFalse(practice._worker_available())
            self.assertFalse(practice.capabilities(User(id=1, email='test@example.com'))['paper_execution_enabled'])
            with self.assertRaises(HTTPException) as raised:
                practice._require_execution_available()
            self.assertEqual(raised.exception.status_code, 503)

    def test_code_lab_message_is_scoped_and_idempotent(self):
        from src.api.routers import practice
        from src.database.models import PracticeMessage
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        saver = InMemorySaver()
        graph = build_repo_graph(RepoServices(), saver)
        graph.invoke({'repo_url': 'https://github.com/example/project'},
                     {'configurable': {'thread_id': 'repo-run'}})
        with Session(engine) as db:
            user = User(id=1, email='repo@example.com', hashed_password='x')
            db.add(user)
            db.add(PracticeRun(run_id='repo-run', user_id=1, kind='repo',
                               source='https://github.com/example/project', status='completed'))
            db.commit()
            fake = Mock(input_tokens=3, output_tokens=4)
            fake.answer_repo_question.return_value = {
                'answer': 'Read [core.py:1-1] first.', 'tool_calls': 1,
                'trace': [{'tool': 'read_file', 'args': {'path': 'core.py'}, 'summary': 'core.py:1-1'}],
                'citations': [{'text': '[core.py:1-1]', 'path': 'core.py', 'start': 1, 'end': 1,
                               'verified': True, 'url': 'https://github.com/example/project/blob/x/core.py#L1'}]}
            with patch.object(practice, 'graph_state', return_value=graph.get_state(
                    {'configurable': {'thread_id': 'repo-run'}}).values), \
                 patch.object(practice, '_require_available'), \
                 patch('src.agents.coding_services.CodingServices', return_value=fake):
                    request = practice.MessageRequest(idempotency_key='message-123', message='Where should I start?')
                    first = practice.send_message('repo-run', request, user, db)
                    second = practice.send_message('repo-run', request, user, db)
            self.assertEqual(first, second)
            self.assertEqual(db.query(PracticeMessage).count(), 1)
            self.assertEqual(first['trace'][0]['tool'], 'read_file')     # stored and returned with the answer
            self.assertTrue(first['citations'][0]['verified'])
            fake.answer_repo_question.assert_called_once()
            fake.answer_about_run.assert_not_called()
            self.assertEqual(db.get(PracticeRun, 'repo-run').input_tokens, 3)

    def test_comparison_is_owner_scoped_and_cached(self):
        from src.api.routers import practice
        from src.database.models import PracticeComparison
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        saver = InMemorySaver()
        services = PaperServices()
        services.paper_evidence = lambda paper_id, objective='': {
            'title': 'Test paper', 'source_url': 'https://arxiv.org/abs/2608.14407',
            'evidence': [{'id': 'E1', 'text': 'A method', 'source': 'arxiv'}],
            'linked_repo': {'url': 'https://github.com/team/method',
                            'provenance': 'https://arxiv.org/abs/2608.14407'}}
        build_paper_graph(services, saver).invoke(
            {'paper_id': '2608.14407', 'attempts': 0},
            {'configurable': {'thread_id': 'compare-run'}})
        with Session(engine) as db:
            owner = User(id=1, email='owner@example.com', hashed_password='x')
            stranger = User(id=2, email='stranger@example.com', hashed_password='x')
            db.add_all([owner, stranger, PracticeRun(run_id='compare-run', user_id=1,
                kind='paper', source='2608.14407', status='completed')])
            db.commit()
            fake = Mock(input_tokens=5, output_tokens=6)
            fake.compare_paper_repo.return_value = {'module_path': 'src/method.py', 'differences': []}
            state = build_paper_graph(services, saver).get_state(
                {'configurable': {'thread_id': 'compare-run'}}).values
            with patch.object(practice, 'graph_state', return_value=state), \
                 patch.object(practice, '_require_available'), \
                 patch('src.agents.coding_runtime.SessionLocal', lambda: Session(engine)), \
                 patch('src.agents.coding_runtime.graph_state', return_value=state), \
                 patch('src.agents.coding_services.CodingServices', return_value=fake):
                    with self.assertRaises(HTTPException) as raised:
                        practice.compare_with_repository('compare-run', stranger, db)
                    self.assertEqual(raised.exception.status_code, 404)
                    first = practice.compare_with_repository('compare-run', owner, db)
                    self.assertEqual(first['status'], 'queued')
                    db.get(PracticeComparison, 'compare-run').status = 'running'
                    db.commit()
                    run_comparison_job('compare-run')
                    second = practice.compare_with_repository('compare-run', owner, db)
            self.assertEqual(second['status'], 'completed')
            self.assertEqual(second['result']['module_path'], 'src/method.py')
            fake.compare_paper_repo.assert_called_once()
            self.assertEqual(db.get(PracticeComparison, 'compare-run').user_id, 1)

    def test_edited_execution_is_marked_unverified_in_response_and_history(self):
        from src.api.routers import practice
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        saver = InMemorySaver()
        build_paper_graph(PaperServices(), saver).invoke(
            {'paper_id': '2608.14407', 'attempts': 0},
            {'configurable': {'thread_id': 'edited-run'}})
        with Session(engine) as db:
            owner = User(id=1, email='editor@example.com', hashed_password='x')
            db.add_all([owner, PracticeRun(run_id='edited-run', user_id=1, kind='paper',
                source='2608.14407', status='completed')])
            db.commit()
            state = build_paper_graph(PaperServices(), saver).get_state(
                {'configurable': {'thread_id': 'edited-run'}}).values
            with patch.object(practice, 'graph_state', return_value=state), \
                 patch.object(practice, '_require_execution_available'), \
                 patch('src.agents.coding_services.WorkerClient') as worker:
                    worker.return_value.execute.return_value = {'passed': True, 'stdout': 'CHECK:PASS'}
                    code = "print('CHECK:PASS')\nprint('user edit')"
                    request = practice.ActionRequest(idempotency_key='edited-123', code=code)
                    result = practice.execute_edited_paper('edited-run', request, owner, db)
                    saved = practice.get_run('edited-run', owner, db)
            self.assertEqual(result['evidence_status'], 'user_modified_unverified')
            self.assertEqual(saved['user_executions'][0]['evidence_status'], 'user_modified_unverified')

    def test_task_worker_persists_graph_snapshot(self):
        with TemporaryDirectory() as directory:
            db_url = f"sqlite:///{directory}/practice.db"
            engine = create_engine(db_url)
            Base.metadata.create_all(engine)
            sessions = sessionmaker(bind=engine)
            with sessions() as db:
                db.add(User(id=1, email="paper@example.com", hashed_password="x"))
                db.add(PracticeRun(run_id="paper-run", user_id=1, kind="paper",
                                   source="2608.14407", status="queued", stage="queued"))
                db.commit()

            def fake_services(db, embedding_manager=None, run_id=None, user_id=None):
                return PaperServices()

            from src.agents import coding_runtime
            with patch.object(coding_runtime, "SessionLocal", sessions), \
                 patch("src.api.deps.get_embedding_manager", return_value=object()), \
                 patch.object(coding_runtime, "get_checkpointer", return_value=InMemorySaver()), \
                 patch("src.agents.coding_services.CodingServices", side_effect=fake_services):
                run_graph_job("paper-run")
                with sessions() as db:
                    row = db.get(PracticeRun, "paper-run")
                    self.assertEqual(row.status, "completed")

    def test_task_worker_recovers_interrupted_database_jobs(self):
        from src.database.models import PracticeComparison
        from src.jobs import practice_worker
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with sessions() as db:
            db.add(User(id=1, email="recover@example.com", hashed_password="x"))
            db.add(PracticeRun(run_id="recover-run", user_id=1, kind="paper",
                               source="2608.14407", status="running", stage="generate"))
            db.add(PracticeComparison(run_id="recover-run", user_id=1, status="running"))
            db.commit()
        with patch.object(practice_worker, "SessionLocal", sessions):
            practice_worker.recover_interrupted_jobs()
        with sessions() as db:
            self.assertEqual(db.get(PracticeRun, "recover-run").status, "queued")
            self.assertEqual(db.get(PracticeRun, "recover-run").stage, "resuming")
            self.assertEqual(db.get(PracticeComparison, "recover-run").status, "queued")

    def test_task_worker_resumes_existing_graph_checkpoint(self):
        from src.agents import coding_runtime
        with TemporaryDirectory() as directory:
            db_url = f"sqlite:///{directory}/resume.db"
            engine = create_engine(db_url)
            Base.metadata.create_all(engine)
            sessions = sessionmaker(bind=engine)
            saver = InMemorySaver()
            original = PaperServices()
            config = {"configurable": {"thread_id": "resume-run"}}
            build_paper_graph(original, saver).invoke(
                {"paper_id": "2608.14407", "attempts": 0, "execution_enabled": True},
                config, interrupt_after=["generate"])
            with sessions() as db:
                db.add(User(id=1, email="resume@example.com", hashed_password="x"))
                db.add(PracticeRun(run_id="resume-run", user_id=1, kind="paper",
                                   source="2608.14407", status="running", stage="generate"))
                db.commit()
            resumed = PaperServices()
            with patch.object(coding_runtime, "SessionLocal", sessions), \
                 patch.object(coding_runtime, "get_checkpointer", return_value=saver), \
                 patch("src.api.deps.get_embedding_manager", return_value=object()), \
                 patch("src.agents.coding_services.CodingServices", return_value=resumed):
                run_graph_job("resume-run")
            with sessions() as db:
                self.assertEqual(db.get(PracticeRun, "resume-run").status, "completed")
            final = build_paper_graph(None, saver).get_state(config).values
            self.assertEqual(final["stop_reason"], "success")
            self.assertEqual(original.calls, 0)
            self.assertEqual(resumed.calls, 2)

if __name__ == "__main__":
    unittest.main()
