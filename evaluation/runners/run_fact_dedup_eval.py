"""
Accuracy of UserFactMemory._classify_fact_relation() against hand-labeled
fact pairs.

This function is a pure token-overlap heuristic (see memory.py) — no LLM
call, no embedding, just tokenization + contiguous-sublist containment +
Jaccard overlap on content words. That matters for how to read this eval:
it is not testing "does the LLM understand paraphrases", it is testing
"does this specific heuristic get the right answer", and a token heuristic
has known, findable blind spots (paraphrases with no shared words, reordered
phrases, negation) that this eval is deliberately designed to expose rather
than avoid.

Two pools:
  templated   — ~120 pairs generated from templates across 12 topics,
                5 balanced per relation type (duplicate/supersedes/redundant/
                update/distinct). The "easy" cases the heuristic was
                actually designed to handle.
  hard_cases  — ~24 hand-curated pairs specifically targeting the
                heuristic's structural blind spots: synonym paraphrases with
                zero token overlap, word-order changes that break the
                contiguous-sublist check, negation, and coincidental
                content-word overlap between genuinely different sub-topics.
                Reported separately — collapsing these into the overall
                accuracy would hide exactly the failure modes worth knowing
                about before relying on this function.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import markdown_table, result_header, write_result  # noqa: E402

from src.agents.memory import UserFactMemory  # noqa: E402

TOPICS = [
    "LoRA fine-tuning", "retrieval augmented generation", "graph neural networks",
    "reinforcement learning", "diffusion models", "knowledge distillation",
    "vision language models", "federated learning", "causal inference",
    "speech recognition", "continual learning", "model quantization",
]


def templated_pairs():
    rows = []
    for i, topic in enumerate(TOPICS):
        base = f"user is studying {topic}"
        rows.append({"id": f"tp-dup-{i:02d}", "new": base, "old": base,
                     "gold": "duplicate", "category": "duplicate"})
        rows.append({"id": f"tp-sup-{i:02d}", "new": f"{base} for a production system",
                     "old": base, "gold": "supersedes", "category": "supersedes"})
        rows.append({"id": f"tp-red-{i:02d}", "new": base,
                     "old": f"{base} for a production system",
                     "gold": "redundant", "category": "redundant"})
        rows.append({"id": f"tp-upd-{i:02d}", "new": f"user now prefers gpt-4o for {topic} tasks",
                     "old": f"user prefers claude for {topic} tasks",
                     "gold": "update", "category": "update"})
        other = TOPICS[(i + 6) % len(TOPICS)]
        rows.append({"id": f"tp-dis-{i:02d}", "new": f"user is studying {topic}",
                     "old": f"user has a deadline related to {other}",
                     "gold": "distinct", "category": "distinct"})
    return rows


# Hand-curated: each targets a specific, named weakness of a token-overlap
# heuristic. `gold` is my best-effort human judgement of what SHOULD happen;
# `weakness` names the mechanism being probed, not just "hard".
HARD_CASES = [
    {"id": "hc-01", "new": "user prefers concise explanations",
     "old": "user likes brief answers", "gold": "update",
     "weakness": "同义改写，内容词零重叠（concise/brief, explanations/answers 完全不同的词）"},
    {"id": "hc-02", "new": "user wants short, to-the-point responses",
     "old": "user prefers detailed step-by-step answers", "gold": "update",
     "weakness": "偏好反转但用词完全不同，跟hc-01同类"},
    {"id": "hc-03", "new": "user does not like transformer-based models",
     "old": "user likes transformer-based models", "gold": "update",
     "weakness": "否定句：likes/does not like 不共享词干，heuristic无词形还原"},
    {"id": "hc-04", "new": "user no longer finds RLHF interesting",
     "old": "user finds RLHF interesting", "gold": "update",
     "weakness": "否定短语插入在中间，掐断连续子串匹配"},
    {"id": "hc-05", "new": "user is studying fine-tuning methods, particularly LoRA and QLoRA",
     "old": "user is studying LoRA and QLoRA fine-tuning methods", "gold": "duplicate",
     "weakness": "词序颠倒但内容完全相同，连续子串匹配要求顺序一致所以检测不到"},
    {"id": "hc-06", "new": "user's focus is now on efficient inference rather than training",
     "old": "user is studying efficient inference for LLMs", "gold": "update",
     "weakness": "转折表达（rather than）+ 部分改写"},
    {"id": "hc-07", "new": "user is studying reinforcement learning for robotics control",
     "old": "user is studying reinforcement learning theory and proofs", "gold": "distinct",
     "weakness": "共享'reinforcement learning'但其实是两个不同的子方向，容易被content word重叠误判成update"},
    {"id": "hc-08", "new": "user is interested in graph neural networks for molecule generation",
     "old": "user is interested in graph neural networks for social network analysis",
     "gold": "distinct", "weakness": "同上，同一大主题下两个不相关的应用场景"},
    {"id": "hc-09", "new": "user's deadline is for an ICML submission",
     "old": "user's deadline is for a NeurIPS submission", "gold": "update",
     "weakness": "高重叠但核心信息（会议名）不同，正确答案是update不是duplicate"},
    {"id": "hc-10", "new": "user is a senior engineer, not a student",
     "old": "user is a masters student new to the field", "gold": "update",
     "weakness": "身份描述矛盾，用词几乎不重叠"},
    {"id": "hc-11", "new": "user wants code examples instead of mathematical notation",
     "old": "user prefers mathematical derivations over code", "gold": "update",
     "weakness": "偏好反转，用词部分重叠（code, mathematical）但结构完全不同"},
    {"id": "hc-12", "new": "user already covered the basics of attention mechanisms",
     "old": "user is new to attention mechanisms and needs basics explained", "gold": "update",
     "weakness": "从'需要讲基础'变成'已经懂基础'，语义相反但共享大量词"},
    {"id": "hc-13", "new": "user is studying LoRA",
     "old": "user is exploring LoRA for parameter-efficient fine-tuning", "gold": "redundant",
     "weakness": "new是old的语义子集但不是连续子串（studying vs exploring用词不同）"},
    {"id": "hc-14", "new": "user cares about model interpretability",
     "old": "user cares about model explainability", "gold": "update",
     "weakness": "interpretability/explainability 近义词但字面完全不同"},
    {"id": "hc-15", "new": "user switched from PyTorch to JAX for their experiments",
     "old": "user uses PyTorch for their experiments", "gold": "update",
     "weakness": "工具切换，only部分词重叠"},
    {"id": "hc-16", "new": "user is skeptical of scaling laws claims",
     "old": "user is very enthusiastic about scaling laws research", "gold": "update",
     "weakness": "态度相反但主题词（scaling laws）重叠度高，可能被误判成duplicate倾向"},
    {"id": "hc-17", "new": "user's undergrad thesis was on computer vision",
     "old": "user's current research is on natural language processing", "gold": "distinct",
     "weakness": "两个不同时期/领域的背景信息，都该保留"},
    {"id": "hc-18", "new": "user asked about batch normalization once out of curiosity",
     "old": "user is actively researching normalization techniques for their thesis",
     "gold": "distinct", "weakness": "一次性好奇提问 vs 长期研究方向，词面重叠但重要性完全不同"},
    {"id": "hc-19", "new": "user dismissed a paper on diffusion models for being too theoretical",
     "old": "user is interested in diffusion models", "gold": "update",
     "weakness": "兴趣表达之后追加了具体的负面反馈，该合并成更细致的画像"},
    {"id": "hc-20", "new": "user is now working on speech models after finishing the vision project",
     "old": "user is working on a computer vision project", "gold": "update",
     "weakness": "研究方向随时间变化，'after finishing'暗示旧事实已过时"},
    {"id": "hc-21", "new": "user mentioned their advisor works on federated learning",
     "old": "user is studying federated learning", "gold": "distinct",
     "weakness": "'导师的方向'和'用户自己在学的方向'是两个不同的事实，不该合并"},
    {"id": "hc-22", "new": "user wants papers with released code",
     "old": "user only reads papers that have open-source implementations", "gold": "duplicate",
     "weakness": "同一个偏好，完全不同的措辞，理想情况应识别为重复"},
    {"id": "hc-23", "new": "user is a research scientist at a startup",
     "old": "user works in industry, not academia", "gold": "update",
     "weakness": "新事实是旧事实的具体化，但字面几乎不重叠"},
    {"id": "hc-24", "new": "user's thesis deadline moved from June to September",
     "old": "user's thesis deadline is in June", "gold": "supersedes",
     "weakness": "日期更新，是对旧事实的修正而非全新事实，但字面上'moved from June to'插入打断了连续包含关系"},
]


def main() -> None:
    templated = templated_pairs()
    all_rows = templated + HARD_CASES

    for r in all_rows:
        r["pred"] = UserFactMemory._classify_fact_relation(r["new"], r["old"])
        r["correct"] = r["pred"] == r["gold"]

    templ_rows = [r for r in all_rows if r["id"].startswith("tp-")]
    hard_rows = [r for r in all_rows if r["id"].startswith("hc-")]

    def acc(rows):
        return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0

    by_cat = {}
    for r in templ_rows:
        by_cat.setdefault(r["category"], []).append(r)

    parts = [result_header("长期记忆去重判断准确率（_classify_fact_relation）", {
        "function": "UserFactMemory._classify_fact_relation (纯token启发式，无LLM调用)",
        "n_templated": len(templ_rows),
        "n_hard_cases": len(hard_rows),
    })]

    parts.append(f"""
## 1. 常规样本（{len(templ_rows)}条，5类关系各{len(templ_rows)//5}条，模板覆盖12个话题）

**总体准确率：{acc(templ_rows):.1%}**
""")
    parts.append(markdown_table(
        ["关系类型", "n", "准确率"],
        [[cat, len(rows), f"{acc(rows):.1%}"] for cat, rows in sorted(by_cat.items())],
    ))

    wrong_templ = [r for r in templ_rows if not r["correct"]]
    if wrong_templ:
        parts.append("\n判错的常规样本：\n")
        parts.append(markdown_table(
            ["id", "new", "old", "gold", "pred"],
            [[r["id"], r["new"][:40], r["old"][:40], r["gold"], r["pred"]] for r in wrong_templ],
        ))

    parts.append(f"""
## 2. 困难样本（{len(hard_rows)}条，专门针对token启发式的已知弱点设计，不是随机抽样）

**准确率：{acc(hard_rows):.1%}** —— 这组数字远比常规样本更重要，它衡量的是"面对同义改写、
词序颠倒、否定句这类真实会发生的表达方式，这套纯词汇重叠的规则还能不能判对"。
""")
    parts.append(markdown_table(
        ["id", "gold", "pred", "对/错", "针对的弱点"],
        [[r["id"], r["gold"], r["pred"], "✓" if r["correct"] else "✗", r["weakness"]]
         for r in hard_rows],
    ))

    overall = templ_rows + hard_rows
    parts.append(f"""
## 3. 总体（常规+困难合并，仅供参考，不建议单独引用）

{sum(r['correct'] for r in overall)}/{len(overall)} = **{acc(overall):.1%}**

不建议把这个合并数字当成headline——它会被常规样本（本来就是heuristic设计时
覆盖的场景）稀释，掩盖困难样本里暴露出的真实弱点。分开报告两组数字更诚实。
""")

    write_result("2026-09-20_fact_dedup_baseline.md", "\n".join(parts))
    print(f"templated: {acc(templ_rows):.1%} ({len(templ_rows)}) | "
          f"hard: {acc(hard_rows):.1%} ({len(hard_rows)}) | "
          f"overall: {acc(overall):.1%} ({len(overall)})")
    for r in wrong_templ:
        print(f"  WRONG (templated) {r['id']}: gold={r['gold']} pred={r['pred']}  "
              f"new='{r['new']}' old='{r['old']}'")
    for r in hard_rows:
        if not r["correct"]:
            print(f"  WRONG (hard) {r['id']}: gold={r['gold']} pred={r['pred']}  {r['weakness']}")


if __name__ == "__main__":
    main()
