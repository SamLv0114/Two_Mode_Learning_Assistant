"""
Long-term memory compression: n>1 version.

The original demonstration (see git history) ran UserFactMemory on a single
hand-written 15-turn conversation and reported 73% token reduction as if it
were a stable property of the system. It is not — it is one sample. This
runner builds ~25 conversations across three deliberately different
*conversation profiles* and reports the distribution (mean, min, max,
quartiles), not a single point estimate.

Profiles (the actual source of variance being measured):
  refining  — turns keep narrowing/restating a small set of ~3 topics.
              Fact dedup (duplicate/supersedes/redundant, see memory.py)
              should collapse most of them -> expect HIGH compression.
  diverse   — every turn introduces a genuinely new, unrelated fact
              (different topic, different fact category each time).
              Nothing to dedup -> expect LOW compression.
  mixed_qa  — mostly plain technical Q&A that reveals nothing personal
              (the extraction prompt is instructed to return 0 facts for
              these), with occasional real disclosures mixed in. This is
              the closest analogue to real ResearchMate chat traffic,
              where most turns are research questions, not self-disclosure.

Each profile is run at short (6-8 turns) and long (18-22 turns) length, with
several topic-seed variants, for ~24-27 conversations total — real variance,
not a cherry-picked example.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (  # noqa: E402
    install_usage_patch, markdown_table, mean, pct, result_header, write_result,
)

import tiktoken  # noqa: E402
from src.agents.memory import UserFactMemory  # noqa: E402
from src.utils.config import settings  # noqa: E402

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


TOPICS = [
    "retrieval-augmented generation", "efficient LLM inference", "graph neural networks",
    "reinforcement learning from human feedback", "speech and audio models",
    "vision-language models", "causal inference", "federated learning",
    "knowledge distillation", "continual learning",
]

EXPERTISE_LEVELS = ["a first-year PhD student", "a senior ML engineer",
                     "a masters student new to the field", "a research scientist"]
STYLES = ["step-by-step derivations", "high-level intuition with diagrams",
          "code-first explanations", "citation-heavy academic style"]
DEADLINES = ["a NeurIPS submission", "a qualifying exam", "an internship report", "a thesis defense"]


def turn_interest(topic):
    return (f"I'm mainly interested in {topic} right now.",
            f"Got it, I'll prioritize {topic} in what I surface for you.")


def turn_dismiss(topic):
    return (f"I found most of the recent {topic} papers pretty shallow, not much new insight.",
            f"Noted, I'll be more selective about {topic} papers going forward.")


def turn_refine(topic, sub):
    return (f"Actually for {topic}, I already know {sub} well, no need to re-explain that part.",
            f"Understood, I'll skip re-explaining {sub} in {topic} discussions.")


def turn_expertise(level, topic):
    return (f"I'm {level} working mostly on {topic}, so you can skip the basics.",
            f"Got it, I'll assume familiarity with {topic} fundamentals.")


def turn_deadline(deadline):
    return (f"I have {deadline} coming up, so please prioritize practical, applicable results.",
            f"Understood, I'll weight practical, directly applicable findings higher given {deadline}.")


def turn_style(style):
    return (f"I prefer {style} when you explain new concepts to me.",
            f"Got it, I'll lean on {style} going forward.")


def turn_qa(topic):
    return (f"Can you explain the core idea behind {topic}?",
            f"{topic.capitalize()} generally works by combining learned representations with a "
            f"task-specific objective; the key design choice is usually how information flows "
            f"between the components involved.")


random.seed(7)


def build_refining(topics, n_turns):
    turns = []
    subtopics = ["the basic formulation", "the standard baseline", "the evaluation protocol"]
    for i in range(n_turns):
        topic = topics[i % len(topics)]
        if i < len(topics):
            turns.append(turn_interest(topic))
        elif i % 3 == 0:
            turns.append(turn_dismiss(topic))
        else:
            turns.append(turn_refine(topic, subtopics[i % len(subtopics)]))
    return turns


def build_diverse(topics, n_turns):
    turns = []
    generators = [turn_interest, turn_dismiss]
    for i in range(n_turns):
        topic = topics[i % len(topics)]
        if i % 5 == 0:
            turns.append(turn_expertise(EXPERTISE_LEVELS[i % len(EXPERTISE_LEVELS)], topic))
        elif i % 5 == 1:
            turns.append(turn_deadline(DEADLINES[i % len(DEADLINES)]))
        elif i % 5 == 2:
            turns.append(turn_style(STYLES[i % len(STYLES)]))
        else:
            turns.append(generators[i % len(generators)](topic))
    return turns


def build_mixed_qa(topics, n_turns):
    turns = []
    for i in range(n_turns):
        topic = topics[i % len(topics)]
        if i % 4 == 3:
            if random.random() < 0.5:
                turns.append(turn_interest(topic))
            else:
                turns.append(turn_expertise(EXPERTISE_LEVELS[i % len(EXPERTISE_LEVELS)], topic))
        else:
            turns.append(turn_qa(topic))
    return turns


def make_conversations():
    convs = []
    profiles = {"refining": build_refining, "diverse": build_diverse, "mixed_qa": build_mixed_qa}
    topic_seeds = [
        TOPICS[0:3], TOPICS[3:6], TOPICS[6:9], TOPICS[1:4], TOPICS[5:8],
    ]
    lengths = {"short": (6, 8), "long": (18, 22)}

    cid = 0
    for profile_name, builder in profiles.items():
        for length_name, (lo, hi) in lengths.items():
            for seed_idx, topics in enumerate(topic_seeds):
                n_turns = random.randint(lo, hi)
                turns = builder(list(topics), n_turns)
                cid += 1
                convs.append({
                    "id": f"mc{cid:03d}", "profile": profile_name, "length": length_name,
                    "n_turns": n_turns, "turns": turns,
                })
    return convs


def main() -> None:
    install_usage_patch()
    memory = UserFactMemory()
    conversations = make_conversations()
    print(f"built {len(conversations)} conversations "
          f"({sum(1 for c in conversations if c['profile']=='refining')} refining, "
          f"{sum(1 for c in conversations if c['profile']=='diverse')} diverse, "
          f"{sum(1 for c in conversations if c['profile']=='mixed_qa')} mixed_qa)")

    records = []
    for i, conv in enumerate(conversations, start=1):
        user_id = 10000 + i  # distinct id per conversation, isolates fallback-dict storage
        raw_parts = []
        for user_msg, assistant_msg in conv["turns"]:
            raw_parts.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
            memory.extract_and_store(user_id, user_msg, assistant_msg, redis_client=None)
        raw_text = "\n\n".join(raw_parts)
        raw_tokens = count_tokens(raw_text)
        ctx = memory.get_context_string(user_id, redis_client=None)
        ctx_tokens = count_tokens(ctx)
        n_facts = ctx.count("\n- ")
        reduction = (1 - ctx_tokens / raw_tokens) if raw_tokens else 0.0
        records.append({**conv, "raw_tokens": raw_tokens, "ctx_tokens": ctx_tokens,
                        "n_facts": n_facts, "reduction": reduction})
        print(f"[{i:2d}/{len(conversations)}] {conv['id']} {conv['profile']:<10} "
              f"{conv['length']:<6} turns={conv['n_turns']:<3} "
              f"raw={raw_tokens:<5} ctx={ctx_tokens:<4} facts={n_facts:<3} "
              f"reduction={reduction:.1%}")

    reductions = [r["reduction"] for r in records]
    parts = [result_header("长期记忆压缩率评估（n>1，多风格对话）", {
        "llm_model": settings.LLM_MODEL,
        "n_conversations": len(records),
        "profiles": "refining / diverse / mixed_qa, each at short(6-8) and long(18-22) turns, 5 topic-seed variants",
    })]

    parts.append(f"""
## 总体分布（{len(records)}段对话，不是单一样本）

- 均值 **{mean(reductions):.1%}**
- 最小 {min(reductions):.1%}，最大 {max(reductions):.1%}
- P25 {pct(reductions, 0.25):.1%}，中位数 {pct(reductions, 0.5):.1%}，P75 {pct(reductions, 0.75):.1%}
""")

    by_profile = {}
    for r in records:
        by_profile.setdefault(r["profile"], []).append(r["reduction"])
    parts.append("## 按对话风格分组（这才是方差的真实来源）\n")
    parts.append(markdown_table(
        ["profile", "n", "均值", "最小", "最大"],
        [[p, len(v), f"{mean(v):.1%}", f"{min(v):.1%}", f"{max(v):.1%}"]
         for p, v in by_profile.items()],
    ))

    parts.append("\n## 逐条结果\n")
    parts.append(markdown_table(
        ["id", "profile", "length", "turns", "raw_tokens", "ctx_tokens", "facts", "reduction"],
        [[r["id"], r["profile"], r["length"], r["n_turns"], r["raw_tokens"],
          r["ctx_tokens"], r["n_facts"], f"{r['reduction']:.1%}"] for r in records],
    ))

    write_result("2026-09-20_memory_compression_baseline.md", "\n".join(parts))
    print(f"\nmean={mean(reductions):.1%} min={min(reductions):.1%} max={max(reductions):.1%} "
          f"(n={len(records)})")


if __name__ == "__main__":
    main()
