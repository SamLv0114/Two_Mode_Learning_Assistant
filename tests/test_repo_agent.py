"""Repository snapshot, deterministic overview, and the tool-using repository agent."""
import io
import json
import tarfile
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents import repo_agent, repo_snapshot
from src.agents.coding_practice import PracticeError
from src.agents.coding_services import CodingServices
from src.agents.repo_agent import RepoAgent
from src.agents.repo_snapshot import RepoSnapshot, build_overview, code_terms, file_brief, parse_archive

SHA = "a" * 40
FILES = {
    "README.md": "# Demo\nA tiny adapter library. See arXiv:2608.14407\n",
    "pyproject.toml": '[project]\nname = "demo-lib"\n[project.scripts]\ndemo = "demo.cli:main"\n',
    "src/demo/__init__.py": '"""Public API."""\nfrom .core import Engine\n__all__ = ["Engine"]\n',
    "src/demo/core.py": "class Engine:\n    def __init__(self, scale):\n        self.scale = scale\n\n"
                        "    def run(self, x):\n        return x * self.scale\n",
    "src/demo/layers.py": "from .core import Engine\n\nclass LoraLayer:\n    def update_layer(self, r, lora_alpha):\n"
                          "        self.scaling = lora_alpha / r\n        return Engine(self.scaling)\n",
    "src/demo/cli.py": "from demo.core import Engine\n\ndef main():\n    print(Engine(2).run(3))\n\n"
                       "if __name__ == '__main__':\n    main()\n",
    "src/demo/utils.py": "from demo import core\n",
    "tests/test_core.py": "from demo.core import Engine\nfrom demo.layers import LoraLayer\n\ndef test_run():\n"
                          "    assert Engine(2).run(1) == 2\n",
    "train.py": "import sys\nfrom model import Net\nnet = Net()\nfor step in range(3):\n    print(step)\n"
                "print('a')\nprint('b')\nprint('c')\nsys.exit(0)\n",
    "model.py": "class Net:\n    pass\n",
    "web/package.json": json.dumps({"name": "web", "main": "src/index.ts", "scripts": {"dev": "next dev"}}),
    "web/tsconfig.json": '{\n  // comments are allowed in tsconfig\n  "compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["./src/*"]}}\n}',
    "web/src/index.ts": 'import { api } from "@/lib/api";\nimport util from "./util";\nexport const start = () => api(util);\n',
    "web/src/lib/api.ts": "export function api(x: unknown) {\n  return x;\n}\n",
    "web/src/util.ts": "export default function util() {}\n",
    "node_modules/pkg/index.js": "module.exports = 1;\n",
    "assets/logo.png": b"\x89PNG\r\n\x1a\n\x00\x00binary",
}


def make_archive(folder: Path, files=FILES, prefix=f"project-{SHA}", symlink=True) -> Path:
    path = folder / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            data = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if symlink:
            link = tarfile.TarInfo(f"{prefix}/passwd")
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(link)
    return path


def fixture_snapshot(files=FILES) -> RepoSnapshot:
    with TemporaryDirectory() as folder:
        return parse_archive(make_archive(Path(folder), files), "example", "project", SHA)


class ArchiveTests(unittest.TestCase):
    def test_archive_is_read_in_memory_without_links_vendored_or_binary_text(self):
        snapshot = fixture_snapshot()
        self.assertIn("src/demo/core.py", snapshot.text)
        self.assertIn("assets/logo.png", snapshot.paths)        # listed ...
        self.assertNotIn("assets/logo.png", snapshot.text)      # ... but not treated as text
        self.assertFalse(any(p.startswith("node_modules/") for p in snapshot.paths))
        self.assertNotIn("passwd", snapshot.paths)
        self.assertFalse(snapshot.truncated)

    def test_a_cut_off_archive_keeps_what_was_downloaded(self):
        big = {f"src/module_{i}.py": f"value = {i}\n" + "x = 'padding text'\n" * 400 for i in range(40)}
        with TemporaryDirectory() as folder:
            path = make_archive(Path(folder), big, symlink=False)
            data = path.read_bytes()
            path.write_bytes(data[: len(data) // 2])
            snapshot = parse_archive(path, "example", "project", SHA)
        self.assertTrue(snapshot.truncated)
        self.assertTrue(0 < len(snapshot.text) < 40)

    def test_text_budget_goes_to_metadata_then_focus_then_core_source(self):
        files = {"README.md": "# r\n", "pyproject.toml": "[project]\nname='x'\n",
                 "a_docs/guide.md": "d" * 300, "lib/core.py": "c = 1\n" * 50, "pkg/focus.py": "f = 1\n" * 50,
                 "tests/test_x.py": "t = 1\n" * 50}
        with TemporaryDirectory() as folder, patch.object(repo_snapshot, "TEXT_MAX_BYTES", 640):
            path = make_archive(Path(folder), files, symlink=False)
            kept = set(parse_archive(path, "example", "project", SHA, focus="pkg").text)
        self.assertEqual(kept, {"README.md", "pyproject.toml", "pkg/focus.py", "lib/core.py"})

    def test_snapshot_requires_a_pinned_sha(self):
        with self.assertRaises(repo_snapshot.SnapshotError):
            repo_snapshot.load_snapshot("example", "project", "main")


class OverviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = fixture_snapshot()
        cls.overview = build_overview(cls.snapshot)

    def test_entry_points_come_from_packaging_metadata_and_code(self):
        entries = {(e["kind"], e["path"]) for e in self.overview["entry_points"]}
        self.assertIn(("console_script", "src/demo/cli.py"), entries)
        self.assertIn(("package_api", "src/demo/__init__.py"), entries)
        self.assertIn(("script", "train.py"), entries)           # top-level script without a main guard
        self.assertIn(("package_main", "web/src/index.ts"), entries)
        self.assertIn(("npm_script", "web/package.json"), entries)
        self.assertNotIn(("script", "src/demo/cli.py"), entries)  # already listed as the console script

    def test_core_modules_are_ranked_by_importers_excluding_tests_and_reexports(self):
        central = {item["path"]: item["imported_by"] for item in self.overview["central_modules"]}
        # __init__, layers, cli and utils import core; the test file does not count
        self.assertEqual(central["src/demo/core.py"], 4)
        self.assertNotIn("src/demo/__init__.py", central)        # re-export only
        self.assertEqual(central["model.py"], 1)                  # script-style sibling import
        self.assertEqual(central["web/src/lib/api.ts"], 1)        # tsconfig '@/*' alias
        self.assertEqual(central["web/src/util.ts"], 1)           # relative import

    def test_focus_limits_the_overview(self):
        overview = build_overview(self.snapshot, "web")
        self.assertTrue(all(item["path"].startswith("web/") for item in overview["central_modules"]))
        self.assertTrue(all(e["path"].startswith("web/") for e in overview["entry_points"]))

    def test_file_brief_has_numbered_opening_lines_and_an_outline(self):
        brief = file_brief(self.snapshot, "src/demo/layers.py")
        self.assertIn("    1| from .core import Engine", brief)
        self.assertIn("L4-6 method LoraLayer.update_layer", brief)


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = fixture_snapshot()

    def test_identifiers_are_split_for_search(self):
        self.assertEqual(code_terms("self.loraAlpha_value"), "self loraalpha_value lora alpha value")

    def test_search_finds_code_from_a_natural_query(self):
        results = self.snapshot.search("lora scaling")
        self.assertEqual(results[0][0], "src/demo/layers.py")

    def test_find_symbol_and_grep(self):
        symbol = self.snapshot.find_symbol("LoraLayer.update_layer")[0]
        self.assertEqual((symbol.path, symbol.start, symbol.end), ("src/demo/layers.py", 4, 6))
        self.assertEqual(self.snapshot.find_symbol("update_layer")[0].qualname, "LoraLayer.update_layer")
        hits, total = self.snapshot.grep(r"self\.scal")
        self.assertEqual(total, 4)
        self.assertEqual(self.snapshot.grep("(unbalanced")[1], 0)  # invalid regex falls back to a literal

    def test_read_file_is_bounded_and_records_what_was_seen(self):
        agent = RepoAgent(Mock(), "gpt-4o-mini", self.snapshot)
        content, summary = agent._read_file("src/demo/core.py", 2, 3)
        self.assertIn("    2|     def __init__(self, scale):", content)
        self.assertEqual(summary, "src/demo/core.py:2-3")
        self.assertEqual(agent.seen["src/demo/core.py"], [(2, 3)])
        with patch.object(repo_agent, "READ_MAX_LINES", 2):
            self.assertEqual(agent._read_file("src/demo/core.py", 3, 99)[1], "src/demo/core.py:3-4")
            # starting where a definition begins may cover all of it
            self.assertEqual(agent._read_file("src/demo/core.py", 1, 99)[1], "src/demo/core.py:1-6")
        self.assertIn("Similar paths: src/demo/core.py", agent._read_file("src/demo/cor.py")[0])
        self.assertIn("binary", agent._read_file("assets/logo.png")[0])


class LargeRepoTests(unittest.TestCase):
    def test_retention_keeps_core_code_over_earlier_files_when_the_budget_is_full(self):
        files = {"aaa_assets/notes.txt": "n" * 300, "core/engine.py": "x = 1\n" * 50, "tests/test_engine.py": "t\n" * 150}
        with TemporaryDirectory() as folder, patch.object(repo_snapshot, "TEXT_MAX_BYTES", 320):
            snapshot = parse_archive(make_archive(Path(folder), files, symlink=False), "example", "project", SHA)
        self.assertEqual(set(snapshot.text), {"core/engine.py"})   # the earlier, less important file was evicted
        self.assertEqual(len(snapshot.paths), 3)                     # but every file is still listed

    def test_files_outside_the_budget_are_fetched_on_demand_once(self):
        snapshot = fixture_snapshot()
        missing = snapshot.text.pop("src/demo/core.py")
        with patch.object(repo_snapshot, "fetch_raw_file", return_value=missing) as fetch:
            agent = RepoAgent(Mock(), "gpt-4o-mini", snapshot)
            self.assertIn("class Engine", agent._read_file("src/demo/core.py", 1, 6)[0])
            agent._read_file("src/demo/core.py", 1, 6)
        fetch.assert_called_once_with("example", "project", SHA, "src/demo/core.py")
        self.assertIn("src/demo/core.py", snapshot.on_demand)
        with patch.object(repo_snapshot, "fetch_raw_file") as fetch:
            self.assertFalse(snapshot.ensure_text("assets/logo.png"))  # known binary: no request
            self.assertFalse(snapshot.ensure_text("not/listed.py"))
        fetch.assert_not_called()

    def test_long_files_show_their_outline_first_and_outline_tool_lists_definitions(self):
        long_module = "\n".join([f"def f{i}():\n    return {i}\n" for i in range(200)])
        snapshot = fixture_snapshot({**FILES, "src/demo/long.py": long_module})
        agent = RepoAgent(Mock(), "gpt-4o-mini", snapshot)
        content, summary = agent._read_file("src/demo/long.py")
        self.assertIn("here is its outline", content)
        self.assertIn("L4-5 function f1", content)
        self.assertNotIn("L600", content)
        self.assertEqual(agent._outline("src/demo/layers.py")[1], "src/demo/layers.py: 2 definitions in 6 lines")

    def test_disk_cache_stores_kept_text_not_the_archive(self):
        snapshot = fixture_snapshot()
        with TemporaryDirectory() as folder, patch.object(repo_snapshot.settings, "DATA_DIR", Path(folder)), \
             patch.object(repo_snapshot, "_download_snapshot", return_value=snapshot) as download:
            repo_snapshot.clear_memory_cache()
            repo_snapshot.load_snapshot("example", "project", SHA)
            repo_snapshot.clear_memory_cache()
            again = repo_snapshot.load_snapshot("example", "project", SHA)
            cached = list((Path(folder) / "repo_snapshots").glob("*.json.gz"))
            repo_snapshot.clear_memory_cache()
        download.assert_called_once()
        self.assertEqual(again.text, snapshot.text)
        self.assertEqual(again._unreadable, {"assets/logo.png"})
        self.assertEqual(len(cached), 1)

    def test_huge_repositories_get_folder_suggestions(self):
        files = {f"pkg/{area}/m{i}.py": "x = 1\n" * 20 for area in ("alpha", "beta") for i in range(20)}
        files.update({f"pkg/gamma/big{i}.py": "y = 2\n" * 170 for i in range(40)})  # too big to fit
        with TemporaryDirectory() as folder, patch.object(repo_snapshot, "TEXT_MAX_BYTES", 3000):
            snapshot = parse_archive(make_archive(Path(folder), files, symlink=False), "example", "project", SHA)
            suggestions = build_overview(snapshot)["focus_suggestions"]
        self.assertEqual({s["path"] for s in suggestions}, {"pkg/alpha", "pkg/beta"})
        self.assertEqual(build_overview(fixture_snapshot())["focus_suggestions"], [])  # small repos need none


def call(call_id, name, args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content=None, calls=None):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10), choices=[
        SimpleNamespace(finish_reason="tool_calls" if calls else "stop",
                        message=SimpleNamespace(content=content, tool_calls=calls))])


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = fixture_snapshot()

    def test_answer_cites_lines_and_only_seen_lines_count_as_verified(self):
        client = FakeClient([
            reply(calls=[call("c1", "find_symbol", {"name": "LoraLayer"}),
                         call("c2", "search_code", {"query": "lora scaling"})]),
            reply(calls=[call("c3", "read_file", {"path": "src/demo/layers.py", "start_line": 1, "end_line": 6})]),
            reply("`update_layer` sets the scale [src/demo/layers.py:4-5] and builds an Engine "
                  "[src/demo/core.py:5-6]; see also [src/demo/missing.py:1-2]."),
        ])
        usage = []
        agent = RepoAgent(client, "gpt-4o-mini", self.snapshot, on_usage=lambda p, c: usage.append((p, c)))
        result = agent.answer("Where is the LoRA scaling set?", [], build_overview(self.snapshot))
        citations = {c["path"]: c for c in result["citations"]}
        self.assertTrue(citations["src/demo/layers.py"]["verified"])
        self.assertEqual(citations["src/demo/layers.py"]["url"],
                         f"https://github.com/example/project/blob/{SHA}/src/demo/layers.py#L4-L5")
        self.assertFalse(citations["src/demo/core.py"]["verified"])   # exists, but never shown to the model
        self.assertFalse(citations["src/demo/missing.py"]["verified"])
        self.assertEqual(citations["src/demo/missing.py"]["url"], "")
        self.assertEqual([step["tool"] for step in result["trace"]], ["find_symbol", "search_code", "read_file"])
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(usage, [(100, 10)] * 3)
        tool_messages = [m for m in client.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_messages], ["c1", "c2", "c3"])
        self.assertTrue(all(m["content"].startswith('<untrusted_source origin="repository_tool_result">')
                            for m in tool_messages))

    def test_tool_budget_is_enforced_and_every_call_is_answered(self):
        greedy = reply(calls=[call(f"c{i}", "grep", {"pattern": "Engine"}) for i in range(3)])
        client = FakeClient([greedy, greedy, greedy, reply("Engine is defined in [src/demo/core.py:1-1].")])
        result = RepoAgent(client, "gpt-4o-mini", self.snapshot).answer("Where is Engine?", [], {})
        self.assertEqual(result["tool_calls"], repo_agent.MAX_TOOL_CALLS)
        self.assertEqual(len(result["trace"]), repo_agent.MAX_TOOL_CALLS)
        self.assertEqual(client.requests[-1]["tool_choice"], "none")
        last = client.requests[-1]["messages"]
        asked = [c["id"] for m in last if m["role"] == "assistant" for c in m.get("tool_calls", [])]
        answered = [m["tool_call_id"] for m in last if m["role"] == "tool"]
        self.assertEqual(asked, answered)
        self.assertTrue(result["citations"][0]["verified"])  # grep showed line 1

    def test_an_answer_without_citations_is_rewritten_once_with_them(self):
        client = FakeClient([
            reply(calls=[call("c1", "read_file", {"path": "src/demo/core.py", "start_line": 1, "end_line": 6})]),
            reply("Engine multiplies its input by the scale."),
            reply("Engine multiplies its input by the scale [src/demo/core.py:5-6]."),
        ])
        result = RepoAgent(client, "gpt-4o-mini", self.snapshot).answer("What does Engine do?", [], {})
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(client.requests[-1]["tool_choice"], "none")
        self.assertTrue(result["citations"][0]["verified"])

    def test_bad_tool_arguments_do_not_crash_the_loop(self):
        client = FakeClient([reply(calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="read_file", arguments="{oops"))]),
                             reply("I could not read it.")])
        result = RepoAgent(client, "gpt-4o-mini", self.snapshot).answer("Read it", [], {})
        self.assertEqual(result["trace"][0]["summary"], "invalid arguments")
        self.assertEqual(result["answer"], "I could not read it.")


class RepoServiceTests(unittest.TestCase):
    def test_repository_url_may_name_a_branch_or_commit_and_a_folder(self):
        target = CodingServices._repo_target
        self.assertEqual(target("https://github.com/o/r"), ("o", "r", "", ""))
        self.assertEqual(target("https://github.com/o/r.git"), ("o", "r", "", ""))
        self.assertEqual(target("https://github.com/o/r/tree/dev/src/pkg"), ("o", "r", "dev", "src/pkg"))
        self.assertEqual(target(f"https://github.com/o/r/commit/{SHA}"), ("o", "r", SHA, ""))
        for bad in ("https://github.com/o/r/blob/main/x.py", "https://github.com/o/r/tree/main/../x",
                    f"https://github.com/o/r/commit/{SHA}/x", "https://github.com/o/r/tree",
                    "http://github.com/o/r", "https://github.com/o/r?x=1"):
            with self.assertRaises(PracticeError, msg=bad):
                target(bad)

    def test_repo_source_pins_the_ref_and_reads_files_chosen_by_structure(self):
        services = CodingServices(Mock(), client=Mock())
        snapshot = fixture_snapshot()
        with patch.object(services, "_github_json", side_effect=[{"private": False, "default_branch": "main"},
                                                                  {"sha": SHA}]) as api, \
             patch("src.agents.coding_services.load_snapshot", return_value=snapshot) as load, \
             patch.object(services, "paper_linked_repo", return_value={}):
            result = services.repo_source("https://github.com/example/project/tree/dev/src")
        self.assertEqual(api.call_args_list[1].args[0], "/repos/example/project/commits/dev")
        load.assert_called_once_with("example", "project", SHA, "src")
        self.assertEqual((result["commit_sha"], result["focus"]), (SHA, "src"))
        inspected = [item["path"] for item in result["files"]]
        self.assertEqual(inspected[0], "src/demo/cli.py")                 # console script first
        self.assertIn("src/demo/core.py", inspected)                      # most imported module
        self.assertIn("Outline:", result["files"][0]["content"])
        self.assertEqual(result["linked_paper"]["relationship"], "readme_reference")

    def test_missing_focus_folder_is_rejected(self):
        services = CodingServices(Mock(), client=Mock())
        with patch.object(services, "_github_json", side_effect=[{"private": False, "default_branch": "main"},
                                                                  {"sha": SHA}]), \
             patch("src.agents.coding_services.load_snapshot", return_value=fixture_snapshot()):
            with self.assertRaisesRegex(PracticeError, "does not exist"):
                services.repo_source("https://github.com/example/project/tree/main/nope")

    def test_repo_question_runs_the_agent_and_counts_tokens(self):
        services = CodingServices(Mock(), client=FakeClient([reply("Start at [src/demo/cli.py:3-4].")]))
        state = {"repo_url": "https://github.com/example/project", "commit_sha": SHA, "focus": "",
                 "overview": {}, "readme": ""}
        with patch("src.agents.coding_services.load_snapshot", return_value=fixture_snapshot()):
            result = services.answer_repo_question(state, "Where do I start?", [])
        self.assertEqual(result["citations"][0]["verified"], False)  # no tool showed those lines
        self.assertEqual((services.input_tokens, services.output_tokens), (100, 10))


if __name__ == "__main__":
    unittest.main()
