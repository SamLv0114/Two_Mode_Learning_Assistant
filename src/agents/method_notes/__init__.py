"""
Method knowledge base for the Coding Agent: implementation notes per method family
(LoRA, attention, diffusion, ...), one Markdown file each in this folder.

A skills-style library with one deliberate difference: notes are chosen by code, not by
the model. A paper's title, its method name and the learner's objective pick at most
MAX_NOTES families by keyword, so which notes reach a prompt is reproducible and can be
evaluated (evaluation/runners/run_method_notes_eval.py).

Notes are background knowledge, never evidence. Every prompt says the paper's excerpts
win where they differ, and the review step already discards any check that does not cite
an excerpt, so a note can direct the reviewer's attention but cannot create a defect.

File format:

    ---
    id: lora                      (must equal the file name)
    name: Low-rank adaptation (LoRA)
    triggers: lora, low-rank adaptation, ...
    excludes: ...                 (optional) phrases in the title or method that rule the note out,
                                  e.g. "vector quantization" for the quantization note
    generic: true                 (optional) a broad family (attention) that is selected like any
                                  other but never ranks above a specific family that also qualifies
    ---
    ## Often left open            choices papers in this family tend not to state
    ## Common mistakes            how implementations of it usually go wrong
    ## Self-checks                properties a toy program can assert
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

NOTES_DIR = Path(__file__).resolve().parent
SECTIONS = ("Often left open", "Common mistakes", "Self-checks")
MAX_NOTES = 2
STRONG_WEIGHT = 3   # a trigger in the title, method name or objective
MIN_SCORE = 3       # one strong hit, or three different triggers in the abstract


@dataclass(frozen=True)
class MethodNote:
    id: str
    name: str
    triggers: Tuple[str, ...]
    sections: Dict[str, str]
    excludes: Tuple[str, ...] = ()
    generic: bool = False

    def matches(self, text: str) -> int:
        """How many different triggers occur in text as whole words."""
        return sum(1 for pattern in _patterns(self.triggers) if pattern.search(text))

    def excluded(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _patterns(self.excludes))


@lru_cache(maxsize=None)
def _patterns(triggers: Tuple[str, ...]) -> Tuple[re.Pattern, ...]:
    # Hyphens and spaces are interchangeable: "low-rank adaptation" == "low rank adaptation".
    # A plural ("attention mechanisms") counts as the trigger.
    return tuple(re.compile(r"(?<![\w-])" + re.escape(t).replace(r"\-", r"[-\s]").replace(r"\ ", r"[-\s]")
                            + r"(?:e?s)?(?![\w-])", re.IGNORECASE) for t in triggers)


def _parse(path: Path) -> MethodNote:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not match:
        raise ValueError(f"{path.name}: missing front matter")
    meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    meta = {key.strip(): value.strip() for key, value in meta.items()}
    sections = {heading.strip(): body.strip()
                for heading, body in re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", match.group(2), re.S | re.M)}
    def phrases(key: str) -> Tuple[str, ...]:
        return tuple(t.strip().lower() for t in meta.get(key, "").split(",") if t.strip())

    note = MethodNote(id=meta.get("id", ""), name=meta.get("name", ""), triggers=phrases("triggers"),
                      sections=sections, excludes=phrases("excludes"),
                      generic=meta.get("generic", "").lower() == "true")
    missing = [s for s in SECTIONS if not sections.get(s)]
    if note.id != path.stem or not note.name or not note.triggers or missing:
        raise ValueError(f"{path.name}: needs id == file name, a name, triggers and sections {missing or SECTIONS}")
    return note


@lru_cache(maxsize=1)
def all_notes() -> Tuple[MethodNote, ...]:
    return tuple(_parse(path) for path in sorted(NOTES_DIR.glob("*.md")))


def select(title: str, abstract: str = "", method: str = "", objective: str = "",
           limit: int = MAX_NOTES) -> List[MethodNote]:
    """The method families a paper belongs to, best match first (deterministic)."""
    strong = " ".join([title or "", method or "", objective or ""])
    scored = []
    for note in all_notes():
        if note.excluded(strong):
            continue
        score = STRONG_WEIGHT * note.matches(strong) + note.matches(abstract or "")
        if note.generic:  # capped at the threshold, so it qualifies but cannot outrank a specific family
            score = min(score, MIN_SCORE)
        if score >= MIN_SCORE:
            scored.append((-score, note.generic, note.id, note))
    return [item[-1] for item in sorted(scored, key=lambda item: item[:3])[:limit]]


def by_ids(ids: Iterable[str]) -> List[MethodNote]:
    known = {note.id: note for note in all_notes()}
    return [known[i] for i in ids if i in known]


def render(notes: Sequence[MethodNote], sections: Sequence[str]) -> str:
    """A prompt block with the chosen sections of each note; empty when there are none."""
    if not notes:
        return ""
    parts = ["\nGeneral implementation notes for this kind of method. They are background knowledge, not "
             "evidence from this paper: where an excerpt says otherwise, follow the excerpt, and never cite "
             "these notes as an excerpt."]
    for note in notes:
        parts.append(f"\n### {note.name}")
        for section in sections:
            parts.append(f"{section}:\n{note.sections[section]}")
    return "\n".join(parts) + "\n"
