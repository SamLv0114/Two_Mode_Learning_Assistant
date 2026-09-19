# ResearchMate 评估体系

离线评测集、跑分脚本和历史结果。所有数字都可复现：评测集和 runner 都在仓库里，
结果文件记录了跑分时的 git commit、模型和关键配置。

```bash
python evaluation/run_all.py              # 跑全部
python evaluation/runners/run_retrieval_eval.py   # 只跑检索
```

---

## 为什么要写死指标口径

同一个 "Recall@5"，分母取法不同能差出十几个点。下面的定义是本仓库所有结果文件的
唯一口径，换口径必须同时改这份文档并重跑历史基线，不允许两套口径的数字混在一起比较。

---

## 1. 检索评估（retrieval）

**语料**：生产环境 ChromaDB 快照，29,215 条 `type=paper` 向量（每篇论文一条，
内容为标题 + 摘要）。快照日期与 commit 记录在结果文件头部。

**命中判定**：文档级。检索结果里只要出现 gold 论文的 `arxiv_id` 即算命中，
同一篇论文的多个 chunk 只计一次。

**Recall@k**
- 单 gold 场景（语义改写型、精确术语型）：gold 集合大小为 1，此时
  `Recall@k` 等价于 `Hit@k`，取值只有 0 或 1，报告的是全体 query 的平均值。
- 多 gold 场景（主题多跳型）：`Recall@k = |前 k 条命中的 gold| / |gold 总数|`，
  分母是该 query 的 gold 总数，不是 k。

**MRR**：`1 / rank_of_first_hit`，rank 从 **1** 开始计数；前 k 条内没有任何 gold
时该 query 记 0。报告的是全体 query 的平均值。k 默认取 10。

**延迟**：单 query 端到端 wall-clock（含 embedding、检索、融合、重排），
不含进程启动和模型加载时间。报告 P50 / P95。

**对照的四个配置**
| 配置 | 实现 |
|---|---|
| `vector_only` | `Retriever.retrieve(use_hybrid=False)`，rerank 关闭 |
| `bm25_only` | 直接调 `Retriever._bm25_search` |
| `hybrid_rrf` | `use_hybrid=True`，rerank 关闭（RRF 融合后直接截断） |
| `hybrid_rrf_rerank` | `use_hybrid=True` + Cross-Encoder 重排 |

**query 分档**（分档报告，不只报总分）
| 档位 | 构造方式 | gold 来源 | 检验什么 |
|---|---|---|---|
| 语义改写型 | LLM 把论文摘要改写成自然问句，提示词明确要求不复用标题里的特征词 | 该论文本身 | 稠密向量 |
| 精确术语型 | LLM 从标题/摘要里抽出最具区分度的方法名、模型名或缩写作为短 query | 该论文本身 | BM25 |
| 主题多跳型 | 自动挖掘语料中**文档频次落在 2~8 之间**的三元词组，query 为 `papers about <词组>` | 含该词组的全部论文（精确字符串匹配） | 召回覆盖度 |

主题档的词组是挖出来的、不是人工挑的：在 29k 篇 ML 语料上，人工挑的常见术语
（如 "diffusion model" 命中 1267 篇）会让 gold 集大到 Recall@10 的天花板只有
10/1267，指标失去意义。低频三元词组能得到小而客观的 gold 集，且整个标注过程
没有任何 LLM 判断参与。

**难样本过滤**（只作用于语义档和精确档）：要求 gold 论文处在向量空间的稠密区域，
即用**该论文自身的文本**检索时，存在 ≥3 篇余弦距离 ≤0.55 的其他论文。

这里刻意不用"看 query 在某个配置下的检索结果难不难"来筛：那样会用被测配置之一
去挑选样本，等于拿考题去迁就考生，四个配置的对照就不成立了。改用论文自身的邻域
密度，对四个配置完全中立，只反映"这篇论文所在的区域本身就拥挤、不好区分"。

**评测集构造方式如实披露**：LLM（gpt-4o-mini）批量生成 + 人工抽检 30 条核验。
不是人工逐条标注，读数时应把这一点考虑进去。

---

## 2. 路由评估（routing）

**任务**：`IntentRecognizer.recognize()` 的四分类
（`research_qa` / `recommendation` / `document_management` / `general_chat`），
以及 `research_qa` 内部的二次路由（`ResearchAgent` / `ReflectionAgent` /
`PlanAndSolveAgent` / `DeepResearchAgent`）。

**gold 标注**：人工标注，标注的是"一个真人看到这句话会认为用户想干什么"。
存在合理歧义的 query 单独标 `ambiguous=true`，主指标里剔除，单独统计。

**指标**
- `intent_accuracy`：四分类准确率
- `stage_distribution`：三阶段各自解决的占比（关键词 / 向量相似度 / LLM 兜底）
- `llm_call_rate`：需要走第三阶段 LLM 分类的 query 占比（= 成本）
- `adversarial_accuracy`：对抗样本子集上的准确率

**对抗样本的构造要求**：每一条在写入数据集前，必须先用
`IntentRecognizer._keyword_match()` 实际跑一遍，确认它在 `KEYWORD_RULES`
里真的同时命中了两个不同意图的关键词——不能只是"看起来像陷阱"就收进去。
第一版对抗集里有两条以 `"vs"` 子串为陷阱（如 `"VSA"`、`"vsync"`），但
`"vs"` 根本不在 `KEYWORD_RULES` 里，它只出现在 `router.py` 的
`_ANALYTICAL_KEYWORDS`（第二次路由阶段），测的是另一段代码、而且那段代码
的单词边界修复在这次评估之前就已经做过了。这两条已替换为真实撞上
`KEYWORD_RULES` 的案例，替换前后的判断过程见 git 历史。

**阈值扫描**：关键词阈值 ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}，
向量阈值 ∈ {0.55, 0.60, 0.65, 0.70, 0.75}。报告 `intent_accuracy` 与
`llm_call_rate` 的权衡曲线。当前线上取值为 0.60 / 0.65。

---

## 3. 歧义处理基线（ambiguity）

**任务**：用户 query 缺少关键槽位（没指明哪篇论文 / 哪个作者 / 哪个时间范围）时，
Agent 的行为。

**标注方式**：人工对每条回答打一个标签，四选一：
| 标签 | 含义 |
|---|---|
| `clarify` | 反问用户澄清，或给出候选让用户选 |
| `guess` | 自行选定一个具体对象直接作答（可能答错对象） |
| `generic` | 不指定对象、只给泛泛的通用回答 |
| `refuse` | 明确表示信息不足无法回答 |

**主指标**：`clarify_rate`。这是 clarify / 中断恢复功能的改动前基线。

---

## 4. Token 与延迟基线（cost）

**采集方式**：评测 runner 在进程内包装 OpenAI SDK 的
`chat.completions.create`，累计每次请求的 `usage.prompt_tokens` /
`completion_tokens`。**生产代码不做修改**，埋点只存在于评测进程里。
因此这里的数字是"同样的 query 在同样的代码路径上跑一遍"的测量值，
不是线上真实流量的统计值。

**指标**
- `total_tokens` 每请求 P50 / P95（含该请求触发的全部 LLM 调用：
  意图分类、Agent 主循环、Critic 打分、Plan/Synthesize 等）
- `llm_calls` 每请求 LLM 调用次数
- `latency_ms` 每请求端到端耗时 P50 / P95，按 agent 分组
- `tool_calls` 每请求工具调用次数
- `duplicate_tool_call_rate`：单次请求内 `(工具名, 参数)` 完全相同的重复调用
  占全部工具调用的比例

---

## 结果文件约定

`results/YYYY-MM-DD_<label>.md`，文件头必须记录：

```
date, git_commit, llm_model, embedding_model, corpus_snapshot,
corpus_size, keyword_threshold, embedding_threshold, rerank_enabled
```

改动前后对比时，两份结果文件除被测变量外的配置必须一致，否则不构成对照。
