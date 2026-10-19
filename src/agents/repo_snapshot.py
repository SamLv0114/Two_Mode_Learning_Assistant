"""Pinned, read-only text snapshot of a public GitHub repository.

One commit's archive is downloaded from codeload.github.com (no REST API
quota), cached on disk by commit SHA (immutable), and held in memory as text.
Nothing in it is executed, imported or extracted to the filesystem.

On top of the text the snapshot offers what a code-reading agent needs:
directory listing, line-ranged reads, regex search, BM25 search over code
chunks, a symbol index, and a deterministic overview (entry points from
packaging metadata and import-graph centrality).
"""
import ast
import configparser
import difflib
import gzip
import heapq
import io
import json
import logging
import os
import posixpath
import re
import tarfile
import threading
import tomllib
import zlib
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from src.utils.config import settings

logger = logging.getLogger(__name__)

STREAM_MAX_BYTES = 400_000_000   # compressed bytes read from codeload; only the kept text stays in memory
FILE_MAX_BYTES = 400_000         # a larger single file is almost always generated or data
TEXT_MAX_BYTES = 25_000_000      # decoded text kept in memory per snapshot
MAX_FILES = 80_000               # paths listed; text kept is bounded by TEXT_MAX_BYTES
ON_DEMAND_FILES = 200
BUCKET_SHARE = 0.15              # one area (e.g. tensorflow/core) may hold at most this share before it yields            # files outside the budget fetched individually when someone reads them
CACHE_MAX_ARCHIVES = 8           # cached snapshots (gzip-compressed kept text, a few MB each)
CACHE_MAX_BYTES = 300_000_000
MEMORY_SNAPSHOTS = 2             # the API host has 2 GB; keep few parsed snapshots alive
CHUNK_LINES, CHUNK_STEP = 50, 40
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache"}
SOURCE_SUFFIXES = (".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs", ".java", ".kt",
                   ".scala", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".rb", ".swift",
                   ".cs", ".php", ".sh", ".lua", ".jl", ".r", ".m")
CONFIG_SUFFIXES = (".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml")
NON_CORE = re.compile(r"(^|/)(tests?|testing|docs?|examples?|benchmarks?|third_party|vendor)(/|$)"
                      r"|(^|/)test_[^/]*$|_test\.\w+$|\.(test|spec)\.(ts|tsx|js|jsx)$")
README_NAMES = ("readme.md", "readme.rst", "readme.txt", "readme")
# Small files that say how a project is built and started; always kept first.
METADATA_NAMES = {"pyproject.toml", "setup.py", "setup.cfg", "package.json", "tsconfig.json", "jsconfig.json",
                  "cargo.toml", "go.mod", "requirements.txt", "makefile", "dockerfile"} | set(README_NAMES)
TOOLING = re.compile(r"(^|/)(\.[^/]+|scripts?|tools?|ci|cmake|build|utils/ci)(/|$)")
INDEX_SUFFIXES = SOURCE_SUFFIXES + (".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml")
INDEX_MAX_BYTES = 200_000


class SnapshotError(Exception):
    pass


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str
    path: str
    start: int
    end: int
    signature: str
    calls: List[Tuple[str, int]] = field(default_factory=list)  # (called name, line), Python only


@dataclass
class _PythonFacts:
    symbols: List[Symbol]
    imports: List[Tuple[str, int, List[str]]]   # (module, relative level, imported names)
    main_guard: bool
    reexport_only: bool
    top_level_code: int = 0                     # statements that run on import (loops, calls, ...)


@dataclass
class RepoSnapshot:
    owner: str
    repo: str
    sha: str
    paths: List[str]                 # every regular file in the archive
    text: Dict[str, str]             # readable text files that were kept
    skipped: int = 0                 # files present but not kept (binary, oversize, budget)
    truncated: bool = False          # the archive itself was cut short
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _python: Optional[Dict[str, _PythonFacts]] = field(default=None, repr=False)
    _python_rest: Optional[Dict[str, _PythonFacts]] = field(default=None, repr=False)
    _symbols: Optional[List[Symbol]] = field(default=None, repr=False)
    _by_name: Optional[Dict[str, List[Symbol]]] = field(default=None, repr=False)
    sizes: Dict[str, int] = field(default_factory=dict)      # byte size of every listed file
    on_demand: set = field(default_factory=set, repr=False)  # files fetched after the snapshot was built
    _unreadable: set = field(default_factory=set, repr=False)
    _path_set: Optional[set] = field(default=None, repr=False)
    _fetch_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _index: Any = field(default=None, repr=False)
    _chunks: Optional[List[Tuple[str, int, int]]] = field(default=None, repr=False)

    # ── reading ───────────────────────────────────────────────────────────

    def url(self, path: str, start: int = 0, end: int = 0) -> str:
        anchor = f"#L{start}-L{end}" if start and end and end != start else (f"#L{start}" if start else "")
        return f"https://github.com/{self.owner}/{self.repo}/blob/{self.sha}/{path}{anchor}"

    def lines(self, path: str) -> List[str]:
        return self.text[path].splitlines()

    def ensure_text(self, path: str) -> bool:
        """
        True when `path` can be read. A listed file that did not fit the text budget is fetched
        once from raw.githubusercontent.com at the pinned commit (no API quota). It is readable
        afterwards but not part of the search index, which was built from the snapshot.
        """
        if path in self.text:
            return True
        if self._path_set is None:
            self._path_set = set(self.paths)
        if path not in self._path_set or path in self._unreadable or self.sizes.get(path, 0) > FILE_MAX_BYTES:
            return False
        with self._fetch_lock:
            if path in self.text:
                return True
            if len(self.on_demand) >= ON_DEMAND_FILES:
                return False
            content = fetch_raw_file(self.owner, self.repo, self.sha, path)
            if content is None:
                self._unreadable.add(path)
                return False
            self.text[path] = content
            self.on_demand.add(path)
            return True

    def suggest(self, path: str) -> List[str]:
        return difflib.get_close_matches(path, list(self.text), n=3, cutoff=0.5)

    def list_dir(self, path: str = "") -> Dict[str, Any]:
        prefix = path.strip("/")
        prefix = f"{prefix}/" if prefix else ""
        dirs: Counter = Counter()
        files = []
        for name in self.paths:
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            if "/" in rest:
                dirs[rest.split("/", 1)[0]] += 1
            else:
                files.append(name)
        return {"dirs": sorted(dirs.items()), "files": sorted(files)}

    # ── search ────────────────────────────────────────────────────────────

    def grep(self, pattern: str, prefix: str = "", ignore_case: bool = True, limit: int = 40,
             code_only: bool = False) -> Tuple[List[Tuple[str, int, str]], int]:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern[:200], flags)
        except re.error:
            regex = re.compile(re.escape(pattern[:200]), flags)
        hits, total = [], 0
        for path in sorted(list(self.text)):
            if (prefix and not path.startswith(prefix)) or (code_only and not path.endswith(SOURCE_SUFFIXES)):
                continue
            for number, line in enumerate(self.text[path].splitlines(), 1):
                if len(line) <= 2000 and regex.search(line):
                    total += 1
                    if len(hits) < limit:
                        hits.append((path, number, line.strip()[:240]))
        return hits, total

    def search(self, query: str, prefix: str = "", limit: int = 8,
               boost: Optional[Dict[str, float]] = None) -> List[Tuple[str, int, int, float]]:
        """
        BM25 over overlapping line windows; identifiers are split so 'lora scaling' meets self.scaling.
        boost multiplies the score of given files (the agent passes import centrality), so a module the
        codebase depends on outranks a helper that merely mentions the same words more often.
        """
        index, chunks = self._search_index()
        terms = code_terms(query)
        if not terms or index is None:
            return []
        ranked = []
        for row, score in index.top(terms, limit * 6):
            path, start, end = chunks[row]
            if prefix and not path.startswith(prefix):
                continue
            weight = (0.6 if NON_CORE.search(path) else 1.0) * (boost or {}).get(path, 1.0)
            ranked.append((path, start, end, score * weight))
        ranked.sort(key=lambda item: -item[3])
        best, seen_files = [], Counter()
        for item in ranked:
            if seen_files[item[0]] < 2:  # at most two windows per file keeps results diverse
                seen_files[item[0]] += 1
                best.append(item)
            if len(best) == limit:
                break
        return best

    def _search_index(self):
        with self._lock:
            if self._index is None:
                from src.rag.retriever import _build_index
                chunks, records = [], []
                for path, content in list(self.text.items()):
                    if not path.endswith(INDEX_SUFFIXES) or len(content) > INDEX_MAX_BYTES:
                        continue  # notebooks, JSON and data files stay reachable through grep
                    lines = content.splitlines()
                    path_terms = code_terms(path.replace("/", " "))
                    for start in range(0, max(len(lines), 1), CHUNK_STEP):
                        window = lines[start:start + CHUNK_LINES]
                        if not window:
                            break
                        chunks.append((path, start + 1, start + len(window)))
                        records.append((str(len(records)), f"{path_terms} {path_terms} {code_terms(chr(10).join(window))}", None))
                self._chunks = chunks
                self._index = _build_index(records, okapi_idf=False) if records else None
            return self._index, self._chunks

    # ── symbols ───────────────────────────────────────────────────────────

    def python_facts(self, include_non_core: bool = False) -> Dict[str, _PythonFacts]:
        """Parsed Python files. Tests, docs and examples are parsed only when asked for: in
        repositories that vendor whole model zoos under examples/ they dominate parse time."""
        with self._lock:
            if self._python is None:
                self._python = {path: _python_facts(path, content) for path, content in list(self.text.items())
                                if path.endswith((".py", ".pyi")) and not NON_CORE.search(path)}
            if include_non_core and self._python_rest is None:
                self._python_rest = {path: _python_facts(path, content) for path, content in list(self.text.items())
                                     if path.endswith((".py", ".pyi")) and NON_CORE.search(path)}
            return {**self._python, **(self._python_rest or {})} if include_non_core else self._python

    def symbols(self) -> List[Symbol]:
        facts = self.python_facts(include_non_core=True)
        with self._lock:
            if self._symbols is None:
                found = [symbol for fact in facts.values() for symbol in fact.symbols]
                for path, content in list(self.text.items()):
                    if not path.endswith((".py", ".pyi")) and path.endswith(SOURCE_SUFFIXES):
                        found.extend(_regex_symbols(path, content))
                self._symbols = found
                self._by_name = defaultdict(list)
                for symbol in found:
                    self._by_name[symbol.name].append(symbol)
            return self._symbols

    # ── call graph (name-based: Python has no static types to resolve receivers) ──

    def references(self, name: str, prefix: str = "", limit: int = 40) -> Tuple[List[Tuple[str, int, str]], int]:
        word = name.strip().split(".")[-1]
        if not re.fullmatch(r"[A-Za-z_]\w*", word):
            return [], 0
        hits = self.grep(rf"\b{word}\b", prefix=prefix, ignore_case=False, limit=limit, code_only=True)
        # a mis-cased name (LoRALayer for LoraLayer) still finds its uses
        return hits if hits[1] else self.grep(rf"\b{word}\b", prefix=prefix, ignore_case=True, limit=limit, code_only=True)

    def callers(self, name: str, limit: int = 25) -> List[Tuple[Symbol, int]]:
        target = name.strip().split(".")[-1]
        hits = [(symbol, line) for symbol in self.symbols() for callee, line in symbol.calls if callee == target]
        hits.sort(key=lambda h: (bool(NON_CORE.search(h[0].path)), bool(TOOLING.search(h[0].path)), h[0].path, h[1]))
        return hits[:limit]

    def callees(self, name: str, limit: int = 30) -> Tuple[Optional[Symbol], List[Tuple[str, int, List[Symbol]]]]:
        """Calls made inside one function, each matched to same-named definitions (same class and file first)."""
        matches = self.find_symbol(name)
        functions = [m for m in matches if m.kind in ("function", "method")]
        if not (functions or matches):
            return None, []
        symbol = (functions or matches)[0]
        if symbol.kind == "class":  # calling a class runs its constructor
            symbol = next((s for s in self.symbols() if s.qualname == f"{symbol.qualname}.__init__"
                           and s.path == symbol.path), symbol)
        by_name = self._by_name or {}
        owner = symbol.qualname.rsplit(".", 1)[0] + "." if "." in symbol.qualname else ""
        result, seen = [], set()
        for callee, line in symbol.calls:
            if callee in seen:
                continue
            seen.add(callee)
            candidates = sorted(by_name.get(callee, []), key=lambda c: (
                not (owner and c.path == symbol.path and c.qualname.startswith(owner)), c.path != symbol.path,
                bool(NON_CORE.search(c.path)), c.path.count("/"), c.path))
            result.append((callee, line, candidates[:3]))  # same-named overrides in subclasses stay visible
        return symbol, result[:limit]

    def find_symbol(self, name: str, limit: int = 15) -> List[Symbol]:
        name = name.strip()
        if not name:
            return []
        symbols = self.symbols()
        exact = [s for s in symbols if s.name == name or s.qualname == name or s.qualname.endswith("." + name)]
        if not exact:
            lowered = name.lower()
            exact = [s for s in symbols if s.name.lower() == lowered]
        if not exact:
            lowered = name.lower()
            exact = [s for s in symbols if lowered in s.qualname.lower()]
        exact.sort(key=lambda s: (bool(NON_CORE.search(s.path)), bool(TOOLING.search(s.path)),
                                  s.path.count("/"), s.path, s.start))
        return exact[:limit]

    def outline(self, path: str, limit: int = 60) -> List[Symbol]:
        if path.endswith((".py", ".pyi")):
            facts = self.python_facts(include_non_core=bool(NON_CORE.search(path)))
            symbols = facts.get(path, _PythonFacts([], [], False, False)).symbols
        else:
            symbols = _regex_symbols(path, self.text.get(path, ""))
        return symbols[:limit]


def fetch_raw_file(owner: str, repo: str, sha: str, path: str) -> Optional[str]:
    """One text file at a pinned commit, or None when it is missing, binary, too large or unreachable."""
    if path.startswith("/") or ".." in path.split("/") or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return None
    try:
        with httpx.stream("GET", f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{quote(path, safe='/')}",
                          timeout=20, follow_redirects=False) as response:
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > FILE_MAX_BYTES:
                    return None
    except httpx.HTTPError:
        return None
    if b"\0" in raw[:8192]:
        return None
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return None


# ── tokenizing ────────────────────────────────────────────────────────────

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def code_terms(text: str) -> str:
    """Space-joined lowercase terms: each identifier plus its snake/camel parts."""
    terms = []
    for word in _IDENTIFIER.findall(text or ""):
        if len(word) > 40:
            continue  # hashes, base64 and generated names only bloat the index
        lower = word.lower()
        terms.append(lower)
        parts = [part.lower() for piece in word.split("_") for part in _CAMEL.findall(piece)]
        if len(parts) > 1:
            terms.extend(parts)
    return " ".join(terms)


# ── language facts ────────────────────────────────────────────────────────

def _python_facts(path: str, content: str) -> _PythonFacts:
    try:
        tree = ast.parse(content, filename=path)
    except (SyntaxError, ValueError, RecursionError):
        return _PythonFacts([], [], False, False)
    lines = content.splitlines()
    top_level_code = sum(1 for node in tree.body if isinstance(node, (ast.Expr, ast.For, ast.While, ast.With, ast.If))
                         and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)))
    symbols: List[Symbol] = []

    def visit(nodes, prefix: str, inside_class: bool):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else ("method" if inside_class else "function")
                qualname = f"{prefix}{node.name}"
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                signature = lines[node.lineno - 1].strip()[:200] if node.lineno <= len(lines) else node.name
                symbols.append(Symbol(node.name, qualname, kind, path, start, node.end_lineno or node.lineno,
                                      signature, [] if kind == "class" else _calls(node)))
                visit(node.body, qualname + ".", isinstance(node, ast.ClassDef))

    visit(tree.body, "", False)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, 0, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", node.level, [alias.name for alias in node.names]))
    main_guard = any(isinstance(node, ast.If) and _is_main_guard(node.test) for node in tree.body)
    reexport_only = path.endswith("__init__.py") and all(_is_reexport_statement(node) for node in tree.body)
    return _PythonFacts(symbols, imports, main_guard, reexport_only, top_level_code)


def _calls(function: ast.AST) -> List[Tuple[str, int]]:
    """Names called in a function body, excluding nested definitions (they are their own symbols)."""
    found, stack = set(), list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name:
                found.add((name, node.lineno))
        stack.extend(ast.iter_child_nodes(node))
    return sorted(found, key=lambda call: (call[1], call[0]))


def _is_main_guard(test: ast.AST) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators))


def _is_reexport_statement(node: ast.AST) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return True  # docstring
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return all(isinstance(t, ast.Name) and t.id.startswith("__") for t in targets)
    if isinstance(node, ast.Try):
        return all(_is_reexport_statement(child) for child in node.body + node.orelse
                   + [stmt for handler in node.handlers for stmt in handler.body]) or not node.body
    if isinstance(node, ast.If):
        return all(_is_reexport_statement(child) for child in node.body + node.orelse)
    return isinstance(node, ast.Pass)


_REGEX_SYMBOLS = [
    ((".ts", ".tsx", ".js", ".jsx", ".mjs"), re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)"), "function"),
    ((".ts", ".tsx", ".js", ".jsx", ".mjs"), re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
    ((".ts", ".tsx", ".js", ".jsx", ".mjs"), re.compile(
        r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),
     "function"),
    ((".ts", ".tsx"), re.compile(r"^\s*(?:export\s+)?(?:interface|type)\s+([A-Za-z_$][\w$]*)"), "type"),
    ((".go",), re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"), "function"),
    ((".go",), re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)"), "type"),
    ((".rs",), re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)"), "function"),
    ((".rs",), re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)"), "type"),
    ((".java", ".kt", ".scala", ".cs", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".swift"),
     re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+|static\s+|data\s+|open\s+)*"
                r"(?:class|struct|interface|enum|object)\s+([A-Za-z_]\w*)"), "type"),
]


def _regex_symbols(path: str, content: str) -> List[Symbol]:
    """Best-effort definitions for non-Python sources; end lines are unknown, so one line is recorded."""
    rules = [(regex, kind) for suffixes, regex, kind in _REGEX_SYMBOLS if path.endswith(suffixes)]
    if not rules:
        return []
    found = []
    for number, line in enumerate(content.splitlines(), 1):
        if len(line) > 400:
            continue
        for regex, kind in rules:
            match = regex.match(line)
            if match:
                found.append(Symbol(match.group(1), match.group(1), kind, path, number, number, line.strip()[:200]))
                break
    return found


# ── overview ──────────────────────────────────────────────────────────────

def _python_module_map(py_paths: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """dotted name -> path, and path -> dotted name, using __init__.py to find each import root."""
    py_paths = sorted(py_paths, key=lambda p: (p.count("/"), p))
    package_dirs = {posixpath.dirname(p) for p in py_paths if p.endswith("/__init__.py")}
    by_name, by_path = {}, {}
    for path in py_paths:
        dirs = path.split("/")[:-1]
        root = len(dirs)
        while root > 0 and "/".join(dirs[:root]) in package_dirs:
            root -= 1
        parts = path.rsplit(".", 1)[0].split("/")[root:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name:
            by_path[path] = name
            by_name.setdefault(name, path)
    return by_name, by_path


def _resolve_module(dotted: str, by_name: Dict[str, str]) -> Optional[str]:
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        path = by_name.get(".".join(parts[:end]))
        if path:
            return path
    return None


def _python_edges(snapshot: RepoSnapshot) -> List[Tuple[str, str]]:
    facts = snapshot.python_facts()
    by_name, by_path = _python_module_map(facts)
    edges = set()
    for path, fact in facts.items():
        module = by_path.get(path, "")
        package = module if path.endswith("__init__.py") else module.rpartition(".")[0]
        folder = posixpath.dirname(path)
        for target_module, level, names in fact.imports:
            if level:
                base = package.split(".") if package else []
                base = base[:len(base) - (level - 1)] if level > 1 else base
                target_module = ".".join(base + ([target_module] if target_module else []))
            candidates = [c for c in [f"{target_module}.{name}" for name in names if name != "*"] + [target_module] if c]
            # Script-style imports ("from model import GPT" next to model.py) resolve in the
            # importing file's own folder first; package imports resolve through the module map.
            target = next((t for c in candidates if level == 0 for t in [_resolve_sibling(c, folder, facts)] if t), None) \
                or next((t for c in candidates for t in [_resolve_module(c, by_name)] if t), None)
            if target and target != path:
                edges.add((path, target))
    return sorted(edges)


def _resolve_sibling(dotted: str, folder: str, files: Dict[str, Any]) -> Optional[str]:
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        stem = posixpath.join(folder, *parts[:end])
        for candidate in (stem + ".py", stem + "/__init__.py"):
            if candidate in files:
                return candidate
    return None


_JS_IMPORT = re.compile(r"""(?:import|export)\s[^'";]*?from\s*['"]([^'"]+)['"]|import\s*['"]([^'"]+)['"]"""
                        r"""|require\(\s*['"]([^'"]+)['"]\s*\)|import\(\s*['"]([^'"]+)['"]\s*\)""")
_JS_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs")


def _js_aliases(snapshot: RepoSnapshot) -> List[Tuple[str, str, str]]:
    """(config folder, alias prefix, target folder) from tsconfig/jsconfig 'paths' entries like '@/*'."""
    aliases = []
    for path in list(snapshot.text):
        if posixpath.basename(path) not in ("tsconfig.json", "jsconfig.json") or path.count("/") > 2:
            continue
        raw = snapshot.text[path]
        try:
            config = json.loads(raw)
        except ValueError:
            try:
                config = json.loads(re.sub(r"(?m)^\s*//[^\n]*$|/\*.*?\*/|,(\s*[}\]])", r"\1", raw, flags=re.S))
            except ValueError:
                continue
        options = config.get("compilerOptions") or {}
        folder = posixpath.dirname(path)
        base = posixpath.normpath(posixpath.join(folder, options.get("baseUrl") or "."))
        for key, targets in (options.get("paths") or {}).items():
            if key.endswith("/*") and isinstance(targets, list) and targets and str(targets[0]).endswith("/*"):
                target = posixpath.normpath(posixpath.join(base, str(targets[0])[:-2]))
                aliases.append((folder, key[:-1], "" if target == "." else target))
    return aliases


def _js_edges(snapshot: RepoSnapshot) -> List[Tuple[str, str]]:
    sources = {p for p in list(snapshot.text) if p.endswith(_JS_SUFFIXES)}
    aliases = _js_aliases(snapshot)
    edges = set()
    for path in sources:
        folder = posixpath.dirname(path)
        for match in _JS_IMPORT.finditer(snapshot.text[path]):
            spec = next(group for group in match.groups() if group)
            if spec.startswith("."):
                base = posixpath.normpath(posixpath.join(folder, spec))
            else:
                base = next((posixpath.join(target, spec[len(prefix):]) for config, prefix, target in aliases
                             if spec.startswith(prefix) and (not config or path.startswith(config + "/"))), None)
                if base is None:
                    continue  # a package import, not a file in this repository
            for candidate in [base] + [base + s for s in _JS_SUFFIXES] + [f"{base}/index{s}" for s in _JS_SUFFIXES]:
                if candidate in sources and candidate != path:
                    edges.add((path, candidate))
                    break
    return sorted(edges)


def _entry_points(snapshot: RepoSnapshot, focus: str, imported: set) -> List[Dict[str, str]]:
    prefix = f"{focus}/" if focus else ""
    facts = snapshot.python_facts()
    by_name, _ = _python_module_map(facts)
    found: List[Dict[str, str]] = []

    def script(name: str, target: str, origin: str):
        module, _, func = target.strip().partition(":")
        path = _resolve_module(module.strip(), by_name)
        if path:
            found.append({"path": path, "kind": "console_script",
                          "detail": f"`{name.strip()}` runs {module.strip()}:{func.strip()} (declared in {origin})"})

    # Packaging metadata near the focus folder, and at the root, which often declares a sub-folder's commands.
    metadata = [p for p in list(snapshot.text) if (p.startswith(prefix) and p.count("/") - prefix.count("/") <= 1)
                or (prefix and "/" not in p)]
    for path in sorted(metadata):
        base = posixpath.basename(path)
        content = snapshot.text[path]
        if base == "pyproject.toml":
            try:
                data = tomllib.loads(content)
            except tomllib.TOMLDecodeError:
                continue
            project = data.get("project") or {}
            for table in (project.get("scripts"), project.get("gui-scripts"),
                          ((data.get("tool") or {}).get("poetry") or {}).get("scripts")):
                for name, target in (table or {}).items():
                    if isinstance(target, str):
                        script(name, target, path)
        elif base == "setup.cfg":
            parser = configparser.ConfigParser()
            try:
                parser.read_string(content)
                block = parser.get("options.entry_points", "console_scripts", fallback="")
            except configparser.Error:
                continue
            for line in block.splitlines():
                if "=" in line:
                    script(*line.split("=", 1), path)
        elif base == "setup.py":
            block = re.search(r"console_scripts['\"]?\s*[:=]\s*\[([^\]]*)\]", content)
            for name, target in re.findall(r"['\"]\s*([\w.-]+)\s*=\s*([\w.]+:[\w.]+)\s*['\"]", block.group(1) if block else ""):
                script(name, target, path)
        elif base == "package.json":
            try:
                data = json.loads(content)
            except ValueError:
                continue
            folder = posixpath.dirname(path)
            bins = data.get("bin")
            bins = {data.get("name", "bin"): bins} if isinstance(bins, str) else (bins if isinstance(bins, dict) else {})
            for name, target in bins.items():
                found.append({"path": posixpath.normpath(posixpath.join(folder, str(target))), "kind": "npm_bin",
                              "detail": f"`{name}` command (package.json bin)"})
            for key in ("main", "module"):
                if isinstance(data.get(key), str):
                    found.append({"path": posixpath.normpath(posixpath.join(folder, data[key])), "kind": "package_main",
                                  "detail": f"package.json {key}"})
            for name, command in list((data.get("scripts") or {}).items())[:6]:
                if isinstance(command, str):
                    found.append({"path": path, "kind": "npm_script", "detail": f"npm run {name}: {command[:80]}"})
    declared = {snapshot.repo.lower().replace("-", "_")}
    if "pyproject.toml" in snapshot.text:
        try:
            declared.add(str((tomllib.loads(snapshot.text["pyproject.toml"]).get("project") or {})
                             .get("name", "")).lower().replace("-", "_"))
        except tomllib.TOMLDecodeError:
            pass
    packages = [p for p in facts if p.endswith("/__init__.py") and p.startswith(prefix)
                and posixpath.dirname(posixpath.dirname(p)) + "/__init__.py" not in facts
                and p.count("/") - prefix.count("/") <= (3 if "src/" in p else 2) and not TOOLING.search(p)]
    packages.sort(key=lambda p: (posixpath.basename(posixpath.dirname(p)).lower() not in declared,
                                 not p.startswith(prefix + "src/"), p.count("/"), p))
    if focus and f"{focus}/__init__.py" in facts:
        packages = [f"{focus}/__init__.py"]  # studying a sub-package: its own API comes first
    for path in packages[:1 if packages and (focus or posixpath.basename(posixpath.dirname(packages[0])).lower() in declared) else 2]:
        _, by_path = _python_module_map(facts)
        name = by_path.get(path) or posixpath.basename(posixpath.dirname(path))
        found.append({"path": path, "kind": "package_api", "detail": f"`import {name}`: the package's public API"})
    scripts = [p for p, fact in facts.items() if p.startswith(prefix) and not NON_CORE.search(p)
               and posixpath.basename(p) not in ("setup.py", "conftest.py", "noxfile.py", "conf.py")
               and "template" not in posixpath.basename(p).lower()  # code generators' templates, not programs
               and (fact.main_guard or (p.count("/") - prefix.count("/") <= 1 and p not in imported
                                        and fact.top_level_code >= 5))]
    listed = {item["path"] for item in found}
    package_dirs = tuple(posixpath.dirname(item["path"]) + "/" for item in found if item["kind"] == "package_api")
    scripts = sorted((p for p in scripts if p not in listed),
                     key=lambda p: (bool(TOOLING.search(p)), bool(package_dirs) and not p.startswith(package_dirs),
                                    p.count("/"), p))
    # a library is used by importing it; its scattered scripts are secondary
    for path in scripts[:3 if any(item["kind"] in ("package_api", "console_script") for item in found) else 6]:
        found.append({"path": path, "kind": "script",
                      "detail": "runs as a script" + (" (if __name__ == '__main__')" if facts[path].main_guard else "")})
    for path in sorted(p for p in list(snapshot.text) if p.startswith(prefix) and not NON_CORE.search(p)):
        if path.endswith(".go") and re.search(r"(?m)^package main\b", snapshot.text[path]) \
                and re.search(r"(?m)^func main\(", snapshot.text[path]):
            found.append({"path": path, "kind": "go_main", "detail": "Go main package"})
        elif path.endswith(".rs") and (path.endswith("src/main.rs") or "/src/bin/" in path):
            found.append({"path": path, "kind": "rust_main", "detail": "Rust binary"})
    unique, seen = [], set()
    for item in found:
        key = (item["path"], item["detail"])
        if key not in seen and item["path"].startswith(prefix) \
                and (item["path"] in snapshot.text or item["kind"] == "npm_script"):
            seen.add(key)
            unique.append(item)
    return unique[:12]


def build_overview(snapshot: RepoSnapshot, focus: str = "") -> Dict[str, Any]:
    """Deterministic structure: layout, entry points, and modules ranked by how many files import them."""
    prefix = f"{focus}/" if focus else ""
    in_focus = [p for p in snapshot.paths if p.startswith(prefix)]
    groups: Dict[str, Counter] = defaultdict(Counter)
    for path in in_focus:
        rest = path[len(prefix):]
        top = rest.split("/", 1)[0] + "/" if "/" in rest else "(root files)"
        groups[top][posixpath.splitext(path)[1] or posixpath.basename(path)] += 1
    layout = [{"path": f"{prefix}{name}" if name != "(root files)" else (focus or "(root files)"),
               "files": sum(counts.values()),
               "kinds": ", ".join(f"{ext} {n}" for ext, n in counts.most_common(3))}
              for name, counts in sorted(groups.items(), key=lambda item: -sum(item[1].values()))][:14]

    all_edges = _python_edges(snapshot) + _js_edges(snapshot)
    edges = [(a, b) for a, b in all_edges if a.startswith(prefix) and b.startswith(prefix)]
    facts = snapshot.python_facts()
    importers: Dict[str, set] = defaultdict(set)
    for source, target in edges:
        if not NON_CORE.search(source):
            importers[target].add(source)
    candidates = [path for path in importers if not NON_CORE.search(path)
                  and not (facts.get(path) and facts[path].reexport_only)]
    candidates.sort(key=lambda p: (-len(importers[p]), -len(snapshot.text.get(p, "")), p))
    central = [{"path": p, "imported_by": len(importers[p])} for p in candidates[:8]]
    if len(central) < 3:  # no usable import graph (e.g. C++/Go): fall back to definition counts
        counts = Counter({path: len(fact.symbols) for path, fact in facts.items() if path.startswith(prefix)
                          and fact.symbols and posixpath.basename(path) not in ("setup.py", "conftest.py")})
        for path, _ in counts.most_common(8):
            if all(path != item["path"] for item in central) and len(central) < 8:
                central.append({"path": path, "imported_by": len(importers.get(path, ()))})
    central_paths = {item["path"] for item in central}
    entry_points = _entry_points(snapshot, focus, {target for _, target in all_edges})
    shown = central_paths | {e["path"] for e in entry_points}
    return {"focus": focus, "layout": layout, "entry_points": entry_points, "central_modules": central,
            "import_edges": [[a, b] for a, b in edges if a in shown and b in central_paths][:24],
            "text_files": sum(1 for p in list(snapshot.text) if p.startswith(prefix)),
            "focus_suggestions": focus_suggestions(snapshot, focus),
            "total_files": len(in_focus), "skipped_files": snapshot.skipped,
            "archive_truncated": snapshot.truncated}


def focus_suggestions(snapshot: RepoSnapshot, focus: str = "", limit: int = 5) -> List[Dict[str, Any]]:
    """
    When less than half of the files fit the snapshot, folders whose source would fit entirely, so
    the learner can study one part of a huge repository with full coverage. Largest first, no nesting.
    """
    prefix = f"{focus}/" if focus else ""
    listed = [p for p in snapshot.paths if p.startswith(prefix)]
    if not listed or sum(1 for p in listed if p in snapshot.text) * 2 >= len(listed):
        return []
    sources: Dict[str, int] = Counter()
    volume: Dict[str, int] = Counter()
    for path in listed:
        if not path.endswith(SOURCE_SUFFIXES) or NON_CORE.search(path) or TOOLING.search(path):
            continue
        parts = path[len(prefix):].split("/")[:-1]
        for depth in range(1, min(len(parts), 4) + 1):
            folder = prefix + "/".join(parts[:depth])
            sources[folder] += 1
            volume[folder] += snapshot.sizes.get(path, 0)
    candidates = [f for f, n in sources.items() if n >= 15 and volume[f] <= TEXT_MAX_BYTES * 0.8]
    candidates.sort(key=lambda f: (-sources[f], f))
    chosen: List[str] = []
    for folder in candidates:
        if not any(folder.startswith(c + "/") or c.startswith(folder + "/") for c in chosen):
            chosen.append(folder)
        if len(chosen) == limit:
            break
    return [{"path": f, "source_files": sources[f]} for f in chosen]


def file_brief(snapshot: RepoSnapshot, path: str, head_lines: int = 30, max_chars: int = 3500) -> str:
    """The opening lines (module docstring, imports) and an outline of definitions with line ranges."""
    lines = snapshot.lines(path)
    head = "\n".join(f"{n:5d}| {line[:200]}" for n, line in enumerate(lines[:head_lines], 1))
    outline = "\n".join(f"  L{s.start}-{s.end} {s.kind} {s.qualname}: {s.signature}"
                        for s in snapshot.outline(path))
    text = f"{len(lines)} lines\n{head}\nOutline:\n{outline or '  (no definitions found)'}"
    return text[:max_chars]


def readme_for(snapshot: RepoSnapshot, focus: str = "") -> Tuple[str, str]:
    for folder in ([focus] if focus else []) + [""]:
        for path in list(snapshot.text):
            if posixpath.dirname(path) == folder and posixpath.basename(path).lower() in README_NAMES:
                return path, snapshot.text[path]
    return "", ""


# ── loading ───────────────────────────────────────────────────────────────
#
# The archive is read as a stream and never stored. Every file is listed; text files compete for
# TEXT_MAX_BYTES by importance (metadata, focus folder, core source, then tests and docs), so a
# huge repository keeps its core code rather than whichever directories sort first. Files that
# did not fit are fetched one at a time when someone reads them. Only the kept text is cached.

_memory: "OrderedDict[str, RepoSnapshot]" = OrderedDict()
_memory_lock = threading.Lock()
_download_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


def load_snapshot(owner: str, repo: str, sha: str, focus: str = "") -> RepoSnapshot:
    """The focus folder only matters for huge repositories: its files are kept first under the text budget."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SnapshotError("A pinned 40-character commit SHA is required")
    key = f"{owner}/{repo}@{sha}:{focus}"
    with _memory_lock:
        if key in _memory:
            _memory.move_to_end(key)
            return _memory[key]
    with _download_locks[key]:
        with _memory_lock:
            if key in _memory:
                return _memory[key]
        snapshot = _read_cache(owner, repo, sha, focus)
        if snapshot is None:
            snapshot = _download_snapshot(owner, repo, sha, focus)
            _write_cache(snapshot, focus)
        with _memory_lock:
            _memory[key] = snapshot
            while len(_memory) > MEMORY_SNAPSHOTS:
                _memory.popitem(last=False)
        return snapshot


class ChunkStream(io.RawIOBase):
    """File-like view of an HTTP body for tarfile's streaming mode; OSError past `limit` bytes."""

    def __init__(self, chunks, limit: int, head: bytes = b""):
        self._chunks, self._buffer, self._left = chunks, bytearray(head), limit - len(head)

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        while not self._buffer:
            chunk = next(self._chunks, b"")
            if not chunk:
                return 0
            if len(chunk) > self._left:
                raise OSError("archive exceeds the download limit")
            self._left -= len(chunk)
            self._buffer.extend(chunk)
        size = min(len(target), len(self._buffer))
        target[:size] = self._buffer[:size]
        del self._buffer[:size]
        return size


def _download_snapshot(owner: str, repo: str, sha: str, focus: str) -> RepoSnapshot:
    try:
        with httpx.stream("GET", f"https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}",
                          timeout=60, follow_redirects=False) as response:
            response.raise_for_status()
            return snapshot_from_stream(ChunkStream(response.iter_bytes(), STREAM_MAX_BYTES), owner, repo, sha, focus)
    except httpx.HTTPError as exc:
        raise SnapshotError(f"Could not download the repository archive: {exc}") from exc


def _keep_priority(name: str, focus: str = "") -> Tuple[int, int, int, int, int, str]:
    base = posixpath.basename(name).lower()
    metadata = base in METADATA_NAMES and name.count("/") <= 2
    kind = 0 if name.endswith(SOURCE_SUFFIXES) else 1 if name.endswith(CONFIG_SUFFIXES) or "." not in base else 2
    outside = bool(focus) and not name.startswith(focus + "/")
    third_party = bool(re.search(r"(^|/)(third_party|vendor|external)(/|$)", name))
    secondary = bool(NON_CORE.search(name) or TOOLING.search(name) or re.search(r"(^|/)security(/|$)", name))
    return (0 if metadata else 1, int(outside), int(secondary) + int(third_party), kind, name.count("/"), name)


def _score(name: str, focus: str) -> int:
    """_keep_priority as one number (lower is more important), for the retention heap."""
    metadata, outside, non_core, kind, depth, _ = _keep_priority(name, focus)
    return ((((metadata * 2 + outside) * 3 + non_core) * 3 + kind) * 100) + min(depth, 99)


def keep_order(paths: Iterable[str], focus: str = "") -> List[str]:
    return sorted(paths, key=lambda path: _keep_priority(path, focus))


def _bucket(name: str) -> str:
    """The area a file belongs to: its first two folders (tensorflow/core), or its first folder."""
    parts = name.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0] if len(parts) > 1 else ""


def snapshot_from_stream(stream, owner: str, repo: str, sha: str, focus: str = "") -> RepoSnapshot:
    paths: List[str] = []
    sizes: Dict[str, int] = {}
    binary: set = set()
    text: Dict[str, str] = {}
    heap: List[Tuple[int, int, str]] = []   # (-score, -arrival, path): the least important file on top
    kept_bytes, arrival, truncated = 0, 0, False
    bucket_bytes: Counter = Counter()
    # Past its share, an area's further files rank behind every other area's, so a C++-heavy
    # core cannot crowd out the Python package that sorts later in the archive.
    share = TEXT_MAX_BYTES * BUCKET_SHARE
    overflow = 10 ** 6
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue  # directories, symlinks and devices are never followed
                name = member.name.split("/", 1)[1] if "/" in member.name else ""
                if not name or any(part in SKIP_DIRS for part in name.split("/")[:-1]):
                    continue
                paths.append(name)
                sizes[name] = member.size
                if len(paths) >= MAX_FILES:
                    truncated = True
                    break
                if member.size > FILE_MAX_BYTES:
                    continue
                bucket = _bucket(name)
                score = _score(name, focus) + (overflow if bucket_bytes[bucket] >= share else 0)
                if kept_bytes + member.size > TEXT_MAX_BYTES and heap and -heap[0][0] <= score:
                    continue  # the budget is full of files at least as important: skip without reading
                handle = tar.extractfile(member)
                raw = handle.read() if handle else b""
                if b"\0" in raw[:8192]:
                    binary.add(name)
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    binary.add(name)
                    continue
                arrival += 1
                heapq.heappush(heap, (-score, -arrival, name))
                text[name] = content
                kept_bytes += len(raw)
                bucket_bytes[bucket] += len(raw)
                while kept_bytes > TEXT_MAX_BYTES and heap:
                    _, _, worst = heapq.heappop(heap)
                    size = len(text.pop(worst).encode("utf-8"))
                    kept_bytes -= size
                    bucket_bytes[_bucket(worst)] -= size
    except (tarfile.TarError, EOFError, OSError, zlib.error) as exc:
        truncated = True  # cut at the download limit or a broken stream: keep what was read
        logger.warning("Repository archive for %s/%s read only in part: %s", owner, repo, exc)
    if not paths:
        raise SnapshotError("No readable files found in the repository")
    snapshot = RepoSnapshot(owner, repo, sha, sorted(paths), text, skipped=len(paths) - len(text), truncated=truncated)
    snapshot.sizes = sizes
    snapshot._unreadable = binary  # known not to be text: never fetched on demand
    return snapshot


def parse_archive(archive: Path, owner: str, repo: str, sha: str, focus: str = "") -> RepoSnapshot:
    with open(archive, "rb") as handle:
        return snapshot_from_stream(handle, owner, repo, sha, focus)


def _cache_dir() -> Path:
    folder = Path(settings.DATA_DIR) / "repo_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _cache_path(owner: str, repo: str, sha: str, focus: str) -> Path:
    return _cache_dir() / f"{owner}__{repo}__{sha}__{focus.replace('/', '~') or 'root'}.json.gz"


def _read_cache(owner: str, repo: str, sha: str, focus: str) -> Optional["RepoSnapshot"]:
    path = _cache_path(owner, repo, sha, focus)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        os.utime(path)  # recently used snapshots survive eviction
    except (OSError, ValueError):
        return None
    snapshot = RepoSnapshot(owner, repo, sha, data["paths"], data["text"],
                            skipped=len(data["paths"]) - len(data["text"]), truncated=data.get("truncated", False))
    snapshot.sizes = data.get("sizes", {})
    snapshot._unreadable = set(data.get("binary", []))
    return snapshot


def _write_cache(snapshot: "RepoSnapshot", focus: str) -> None:
    path = _cache_path(snapshot.owner, snapshot.repo, snapshot.sha, focus)
    partial = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.part")  # API and worker share it
    try:
        with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump({"paths": snapshot.paths, "text": snapshot.text, "sizes": snapshot.sizes,
                       "binary": sorted(snapshot._unreadable), "truncated": snapshot.truncated}, handle)
        os.replace(partial, path)
    except OSError as exc:
        logger.warning("Could not cache repository snapshot: %s", exc)
    finally:
        partial.unlink(missing_ok=True)
    _evict()


def _evict() -> None:
    folder = _cache_dir()
    for legacy in folder.glob("*.tar.gz"):  # archives cached by the previous loader
        legacy.unlink(missing_ok=True)
    cached = sorted(folder.glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for index, entry in enumerate(cached):
        total += entry.stat().st_size
        if index >= CACHE_MAX_ARCHIVES or total > CACHE_MAX_BYTES:
            entry.unlink(missing_ok=True)


def clear_memory_cache() -> None:
    with _memory_lock:
        _memory.clear()
