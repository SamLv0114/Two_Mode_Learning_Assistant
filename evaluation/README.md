# ResearchMate 评估体系

离线评测集、跑分脚本和历史结果。所有数字都可复现：评测集和 runner 都在仓库里，
结果文件记录了跑分时的 git commit、模型和关键配置。

```bash
python evaluation/run_all.py              # 跑默认的五项离线基线
python evaluation/run_all.py generation   # 单独跑真实 Agent 生成评测，需 API 与固定语料
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

## 5. 长期记忆压缩率（memory_compression）

**任务**：`UserFactMemory` 把跨会话对话压缩成结构化事实列表，压缩率不是靠
单次对话演示报告，而是靠覆盖不同对话风格的多个样本报告分布。

**三种对话风格**（方差的真实来源，不是随机噪声）：
| profile | 特征 | 预期压缩率 |
|---|---|---|
| refining | 反复修正/收窄同一小组主题 | 高——重复内容被去重逻辑合并 |
| diverse | 每轮都是全新无关事实，主题不重复 | 低——没有可合并的重复项 |
| mixed_qa | 多数轮次是纯技术问答（不含个人信息，按抽取prompt规则应返回0条事实），少数轮次才真正披露信息 | 高——原文因问答内容变长，但压缩后的事实列表几乎不受影响 |

每种风格各跑 short（6-8轮）和 long（18-22轮）两档、5个话题种子，共30段对话。

**指标**：`reduction = 1 - 压缩后token数 / 原始token数`，逐段计算后报告
均值、最小值、最大值、四分位数，并按profile分组报告——分组数字比总体均值
更有信息量，因为方差主要由对话风格决定，不是随机误差。

---

## 结果文件约定

`results/YYYY-MM-DD_<label>.md`，文件头必须记录：

```
date, git_commit, llm_model, embedding_model, corpus_snapshot,
corpus_size, keyword_threshold, embedding_threshold, rerank_enabled
```

改动前后对比时，两份结果文件除被测变量外的配置必须一致，否则不构成对照。

---

## 6. 离线生成质量与工具轨迹

`rubrics/generation_v1.md` 固定 faithfulness、answer relevancy、无证据断言的
判定口径。`run_generation_eval.py` 使用与生成模型分开的 judge 模型（默认
`gpt-4o`，可由 `EVAL_JUDGE_MODEL` 指定）。忠实度 judge 只看**实际返回给
Agent 的证据片段**，不看参考答案；另一次独立调用把人工确认的参考答案和
标注证据交给正确性 judge，衡量关键事实覆盖与矛盾。具备人工相关性标注时，
另按返回来源的排名计算 returned_context_precision：只对已返回的相关来源
取命中位置 precision 的平均值；当前每题只有一个相关来源时等于其倒数排名，
不是对全部相关来源归一化的 AP。原始回答与逐题结果单独存 JSONL，
报告披露数据集、预测文件及两版 rubric 的 SHA-256。

`run_trajectory_eval.py` 对人工标注的必需工具、禁止工具、关键先后约束打分。
多余但合理的调用不会因不匹配某一条完整序列而自动判错。

`datasets/generation_candidates.jsonl` 是从检索集及指定 ChromaDB 快照
提取的 50 条候选。项目所有者在对话中确认已逐条核验答案后，
`datasets/generation_golden.jsonl` 保存了 50 条 `human_verified` 记录。
其中 47 条为摘录支持的答案，3 条记录原摘录不足；后者不计入答案正确性
均值。草稿出自与生成 Agent 相同的 `gpt-4o-mini`，因此报告需披露相关偏差。
工具期望来自自动候选，尚未经单独人工确认，未纳入 golden 的轨迹标签。

2026-09-26 的 50 题真实 Agent 评测结果保存在
`results/2026-09-26_generation_eval.md`，逐题评分和回答另存 JSONL。
faithfulness 为 0.736（50 题），answer correctness 为 0.500（47 条证据充足题）；
参考论文仅出现在 22/50 条返回引文中。题目没有指明目标论文，可能存在其他合理
答案，所以这组结果是探索性基线，不是线上准确率或任何改动的提升幅度。
judge 尚无人工评分子集校准，生成成本估算不含 judge。复现实验仍需归档固定语料快照。
该历史报告的 `context_precision` 使用上述命中位置口径；新报告将其明确标为
`returned_context_precision`，两者公式相同。

```bash
python evaluation/runners/build_generation_candidates.py --chroma-sqlite /path/to/chroma.sqlite3
VECTOR_DB_DIR=/path/to/fixed/chroma-copy python evaluation/run_all.py generation
```

检索 runner 不再覆盖 `VECTOR_DB_DIR`：需在运行前把它设为同一语料快照
的**可写副本**路径（Chroma 初始化会写数据库文件）。结果中的
`corpus_snapshot` 应与实际路径、样本量和版本一致。
GitHub Actions 的 `ci.yml` 在分支 push 与 PR 上运行确定性回归和 smoke test；
需要 API 密钥、固定语料快照及人工审核集的全量模型评测仍按明确的输入条件单独运行，
不能把未执行的模型评测写成已通过的 CI 门禁。

可选地用 `python evaluation/draft_generation_answers.py` 生成答案草稿；
这会把候选问题和证据摘录发送到模型 API，输出到
`datasets/generation_answer_drafts.jsonl`，状态仅为 `ai_draft`。
当前 50 条草稿中，47 条给出候选答案，`gq011`、`gq013`、`gq030`
因摘录未覆盖问题所需结论而标为证据不足；审核者可确认拒答或改写答案。
人工审核使用 `python evaluation/review_golden.py --reviewer <姓名或稳定ID>`；
工具逐条展示候选问题、证据和匹配的草稿。Enter 可选用草稿答案，但审核者仍须
亲自核验该答案、填写预期行为与标注说明，并单独确认工具期望；未确认的工具
期望不会作为人工轨迹标签保存。完成后才会写入 `generation_golden.jsonl`。
judge 的人工一致性抽样可放在
`datasets/judge_calibration.jsonl`，每行包含 `id`、`human_faithfulness`、
`human_answer_relevancy`（0–1）；报告给出 MAE 和 0.70 阈值的一致率，缺文件时
明确标为未校准。全量模型评测不能由自动生成候选代替人工核验。

新增的可复现实验：`run_context_budget_eval.py --dry-run` 只核对输入 token
与组装路径，不给问答质量分数；去掉 `--dry-run` 才比较长会话精确事实/URL 保留率。
`run_chunking_eval.py` 使用同一全文快照比较固定块、语义子块及父块证据。
本地快照只有一篇论文全文，结果是该论文分段上的弱监督结果，不代表跨论文性能，
也不测表格解析质量。`run_cache_eval.py` 用 12 对人工标记的相似/相反查询测错误复用，
不代表线上命中率或成本节省。`run_prompt_injection_eval.py` 测构造的提示词层攻击，
不代表端到端工具越权率。`run_supervisor_eval.py` 要求上述人工 golden，且不会
自动启用生产 supervisor。

`ci.yml` 每次分支 push/PR 运行确定性测试与可离线执行的上下文、缓存实验。
`evaluation-live.yml` 每周一运行需要 API 的注入评测；仓库出现人工 golden 后，
还需维护者提供名为 `eval-corpus-v1` 的 GitHub Release，其中
`chroma-snapshot.tar.zst` 解压后直接包含 Chroma 数据库及索引文件，才会运行
生成与轨迹全量评测。没有密钥、快照或人工标注时，相应指标必须标为未测。
