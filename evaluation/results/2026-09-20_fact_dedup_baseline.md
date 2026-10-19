# 长期记忆去重判断准确率（_classify_fact_relation）

```
date            2026-09-20
git_commit      0d66ecb
function        UserFactMemory._classify_fact_relation (纯token启发式，无LLM调用)
n_templated     60
n_hard_cases    24
```

## 1. 常规样本（60条，5类关系各12条，模板覆盖12个话题）

**总体准确率：100.0%**

| 关系类型 | n | 准确率 |
|---|---|---|
| distinct | 12 | 100.0% |
| duplicate | 12 | 100.0% |
| redundant | 12 | 100.0% |
| supersedes | 12 | 100.0% |
| update | 12 | 100.0% |

## 2. 困难样本（24条，专门针对token启发式的已知弱点设计，不是随机抽样）

**准确率：33.3%** —— 这组数字远比常规样本更重要，它衡量的是"面对同义改写、
词序颠倒、否定句这类真实会发生的表达方式，这套纯词汇重叠的规则还能不能判对"。

| id | gold | pred | 对/错 | 针对的弱点 |
|---|---|---|---|---|
| hc-01 | update | distinct | ✗ | 同义改写，内容词零重叠（concise/brief, explanations/answers 完全不同的词） |
| hc-02 | update | distinct | ✗ | 偏好反转但用词完全不同，跟hc-01同类 |
| hc-03 | update | distinct | ✗ | 否定句：likes/does not like 不共享词干，heuristic无词形还原 |
| hc-04 | update | update | ✓ | 否定短语插入在中间，掐断连续子串匹配 |
| hc-05 | duplicate | update | ✗ | 词序颠倒但内容完全相同，连续子串匹配要求顺序一致所以检测不到 |
| hc-06 | update | distinct | ✗ | 转折表达（rather than）+ 部分改写 |
| hc-07 | distinct | distinct | ✓ | 共享'reinforcement learning'但其实是两个不同的子方向，容易被content word重叠误判成update |
| hc-08 | distinct | distinct | ✓ | 同上，同一大主题下两个不相关的应用场景 |
| hc-09 | update | update | ✓ | 高重叠但核心信息（会议名）不同，正确答案是update不是duplicate |
| hc-10 | update | distinct | ✗ | 身份描述矛盾，用词几乎不重叠 |
| hc-11 | update | distinct | ✗ | 偏好反转，用词部分重叠（code, mathematical）但结构完全不同 |
| hc-12 | update | distinct | ✗ | 从'需要讲基础'变成'已经懂基础'，语义相反但共享大量词 |
| hc-13 | redundant | distinct | ✗ | new是old的语义子集但不是连续子串（studying vs exploring用词不同） |
| hc-14 | update | update | ✓ | interpretability/explainability 近义词但字面完全不同 |
| hc-15 | update | distinct | ✗ | 工具切换，only部分词重叠 |
| hc-16 | update | distinct | ✗ | 态度相反但主题词（scaling laws）重叠度高，可能被误判成duplicate倾向 |
| hc-17 | distinct | distinct | ✓ | 两个不同时期/领域的背景信息，都该保留 |
| hc-18 | distinct | distinct | ✓ | 一次性好奇提问 vs 长期研究方向，词面重叠但重要性完全不同 |
| hc-19 | update | distinct | ✗ | 兴趣表达之后追加了具体的负面反馈，该合并成更细致的画像 |
| hc-20 | update | distinct | ✗ | 研究方向随时间变化，'after finishing'暗示旧事实已过时 |
| hc-21 | distinct | distinct | ✓ | '导师的方向'和'用户自己在学的方向'是两个不同的事实，不该合并 |
| hc-22 | duplicate | distinct | ✗ | 同一个偏好，完全不同的措辞，理想情况应识别为重复 |
| hc-23 | update | distinct | ✗ | 新事实是旧事实的具体化，但字面几乎不重叠 |
| hc-24 | supersedes | update | ✗ | 日期更新，是对旧事实的修正而非全新事实，但字面上'moved from June to'插入打断了连续包含关系 |

## 3. 总体（常规+困难合并，仅供参考，不建议单独引用）

68/84 = **81.0%**

不建议把这个合并数字当成headline——它会被常规样本（本来就是heuristic设计时
覆盖的场景）稀释，掩盖困难样本里暴露出的真实弱点。分开报告两组数字更诚实。
