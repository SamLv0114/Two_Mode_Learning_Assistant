# 歧义处理基线 + Token/延迟基线

```
date            2026-09-19
git_commit      237fd51
llm_model       gpt-4o-mini
dataset         ambiguity.jsonl (20 underspecified queries)
corpus_snapshot prod 2026-09-19
labelling       gpt-4o-mini, 原始回答全文见附录
```

## 1. 行为分布

| 行为 | 条数 | 占比 |
|---|---|---|
| clarify（反问澄清） | 10 | 50% |
| guess（自行选定对象作答） | 6 | 30% |
| generic（泛泛而谈不指定对象） | 2 | 10% |
| refuse（拒答但不追问） | 2 | 10% |

**clarify_rate = 50%** ← clarify / 中断恢复功能的改动前基线


## 2. Token 与延迟基线

| 指标 | P50 | P95 | 均值 |
|---|---|---|---|
| total_tokens | 2336 | 7381 | 2989 |
| llm_calls | 5 | 12 | 5.2 |
| latency_ms | 4137 | 58444 | 13122 |
| tool_calls | 1 | 5 | 1.1 |

工具调用总数 22，其中参数完全相同的重复调用 0 次，**重复调用率 0.0%**


## 3. 逐条结果

| id | query | 路由到 | 行为 | tokens | 延迟ms |
|---|---|---|---|---|---|
| am01 | summarize that paper for me | ReflectionAgent | clarify | 2157 | 3600 |
| am02 | what did that author conclude | ReflectionAgent | clarify | 2334 | 5465 |
| am03 | is this method better than the baseline | PlanAndSolveAgent | guess | 7817 | 64949 |
| am04 | can you explain the results section | ReflectionAgent | generic | 4782 | 9953 |
| am05 | how many citations does it have | ReflectionAgent | clarify | 2336 | 3507 |
| am06 | compare those two papers | PlanAndSolveAgent | guess | 7363 | 58444 |
| am07 | what's the main contribution | ReflectionAgent | clarify | 2319 | 3946 |
| am08 | did they release the code | GeneralAgent | clarify | 275 | 845 |
| am09 | what dataset did they evaluate on | GeneralAgent | clarify | 342 | 1151 |
| am10 | give me the key takeaways from it | ReflectionAgent | clarify | 2273 | 3318 |
| am11 | is that approach still state of the art | ReflectionAgent | clarify | 2369 | 4137 |
| am12 | what year was it published | DocumentAgent | refuse | 592 | 2320 |
| am13 | who else works on this | GeneralAgent | clarify | 418 | 1701 |
| am14 | summarize the one I saved | ReflectionAgent | guess | 3101 | 8721 |
| am15 | explain the second one | ReflectionAgent | guess | 4562 | 11149 |
| am16 | what are its limitations | ReflectionAgent | generic | 4655 | 10338 |
| am17 | how does it compare to the transformer ver | PlanAndSolveAgent | guess | 7381 | 55114 |
| am18 | can you pull the full text | DocumentAgent | refuse | 591 | 1964 |
| am19 | what did the ablation show | ReflectionAgent | guess | 3565 | 9428 |
| am20 | is it worth reading | RecommendationAgent | clarify | 545 | 2396 |

## 附录：原始回答（供人工核验标注是否正确）


**am01** `summarize that paper for me` → *clarify*

> Please provide me with the title or arXiv ID of the paper you'd like me to summarize, and I will help you with that.


**am02** `what did that author conclude` → *clarify*

> To provide you with the specific conclusions drawn by the author, I'll need the title of the paper or specific details regarding it. Please share more information so I can assist you effectively.


**am03** `is this method better than the baseline` → *guess*

> ## Evaluation of the Method Compared to the Baseline
> 
> To determine if the method is better than the baseline, we evaluated both quantitative and qualitative aspects based on predefined criteria: performance, accuracy, efficiency, and scalability. Below is a detailed analysis of each dimension.
> 
> ## Performance
> 
> ### Quantitative Comparison
> - **Success Rate**: The method achieved an 85% success rate compared to the baseline's 78%. This indicates a higher probability of achieving intended outcomes with the method.
> - **Throughput**: The method processed 150 tasks per minute, outperforming the baseline's 120 tasks per minute. This suggests the method can handle more tasks in the same timeframe, en


**am04** `can you explain the results section` → *generic*

> The "Results" section of a research paper is critical as it presents the findings of the experiments or analyses conducted. Here's what typically comprises a well-structured Results section:
> 
> 1. **Data Presentation**: This includes tables, graphs, and other visual aids that concisely illustrate the results. This visualization helps the reader to grasp trends, comparisons, and significant findings easily without delving into numerical data first.
> 
> 2. **Narrative Description**: Following the data presentation, there is often a narrative that summarizes key findings in words. This narrative helps contextualize the data, highlighting important trends and insights while guiding readers through th


**am05** `how many citations does it have` → *clarify*

> To provide you with the citation count for a specific paper, please provide the title or arXiv ID of the paper in question, and I will check for the information for you.


**am06** `compare those two papers` → *guess*

> ## Comparison of Two Papers on Generative Models and AI in Information Systems
> 
> In this analysis, we compare two papers: "On Unifying Deep Generative Models" and "The Landscape of Generative AI in Information Systems: A Synthesis of Secondary Reviews and Research Agendas." These papers address different aspects of generative AI, with the first focusing on the technical integration of deep generative models and the second providing a broader overview of generative AI's role in information systems.
> 
> ## Scope and Objectives
> 
> ### Paper 1: On Unifying Deep Generative Models
> 
> - **Scope**: This paper is primarily concerned with the technical integration of Generative Adversarial Networks (GANs) and


**am07** `what's the main contribution` → *clarify*

> It seems like there might have been a misunderstanding regarding your question. Could you please clarify which paper or concept you're referring to? This way, I can provide you with the main contributions and relevant insights backed by sources.


**am08** `did they release the code` → *clarify*

> Could you please clarify which specific model or project you are asking about regarding the code release?


**am09** `what dataset did they evaluate on` → *clarify*

> To provide a specific answer regarding which dataset was used for evaluation, I would need more context about the particular study, paper, or model you're referring to. Different research papers and models in deep learning, LLMs, and AI agents can use a variety of datasets depending on their focus.
> 
> If you could share the title or topic of the paper or model you're interested in, I could help you find the relevant information.


**am10** `give me the key takeaways from it` → *clarify*

> Could you please clarify which paper or topic you're referring to? This will help me provide the key takeaways you're looking for.


**am11** `is that approach still state of the art` → *clarify*

> To determine if a specific approach is still state of the art in deep learning, LLMs, or AI agents, I need to know which approach you are referring to. Once you provide that information, I can search for the latest developments and insights related to that approach to confirm its status.


**am12** `what year was it published` → *refuse*

> You currently have no documents uploaded in your knowledge base. Please upload relevant materials, and I can assist you in finding publication years or other information.


**am13** `who else works on this` → *clarify*

> To provide more accurate information, could you clarify who or what "this" refers to? Are you asking about other researchers, organizations, or initiatives that work in deep learning, large language models, AI agents, or related fields?


**am14** `summarize the one I saved` → *guess*

> It looks like you saved the paper titled **"Query Rewriting for Retrieval-Augmented Large Language Models."** Here's a summary based on its abstract:
> 
> This work presents a new framework for retrieval-augmented large language models (LLMs), named **Rewrite-Retrieve-Read**, which shifts the focus from the traditional **retrieve-then-read** approach. Instead of primarily modifying the retriever or reader components, the study emphasizes adapting the search query itself. This is crucial because there is often a significant gap between the input query and the results retrieved, leading to suboptimal performance in knowledge-intensive tasks.
> 
> For more details, you can access the paper [here](https


**am15** `explain the second one` → *guess*

> The second paper titled "Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity?" discusses a process in which multiple large language models (LLMs) are used in a systematic approach to generate and refine text. The pipeline consists of three stages: one model drafts the text, a second critiques it and provides feedback, and the third uses that feedback to revise the draft.
> 
> A key focus of the paper is the issue of deictic ambiguity, where context-dependent expressions (such as "previous") can have different meanings depending on the surrounding text. The authors explore whether LLMs employed at different stages of the pipeline can consistently resolve these ambiguities. This is


**am16** `what are its limitations` → *generic*

> Deep learning has several notable limitations across various applications. Here are some key points drawn from recent literature:
> 
> 1. **Vulnerability to Adversarial Attacks**: Deep learning models can be easily misled by adversarial inputs—carefully crafted examples designed to invoke incorrect classifications. This adversarial susceptibility raises concerns about the reliability of deep learning in security-sensitive applications [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528).
> 
> 2. **Overfitting and Data Dependency**: Deep learning requires large amounts of annotated data for effective training. In scenarios with limited data, models tend to over


**am17** `how does it compare to the transformer version` → *guess*

> ## Comparison of Transformer Architecture with Alternative Models
> 
> In the realm of deep learning, particularly for sequence modeling tasks, the transformer architecture has emerged as a dominant force. However, to fully appreciate its strengths, it is essential to compare it with other architectures such as Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Convolutional Neural Networks (CNNs). This comparison will highlight the distinct characteristics and functionalities that set transformers apart.
> 
> ## Architecture and Processing
> 
> ### Transformer Architecture
> Transformers are built on a foundation of self-attention mechanisms, which allow them to weigh the sign


**am18** `can you pull the full text` → *refuse*

> It looks like you currently have no documents uploaded to your knowledge base. To assist you further, please consider uploading relevant material.


**am19** `what did the ablation show` → *guess*

> Several recent papers mention ablation studies in different contexts. Here are three notable findings:
> 
> 1. **Characterising cardiac tissue properties with graph neural networks**: This study explores how ablation targets in arrhythmia treatment can be precisely localized using graph neural networks. The ablation results demonstrated the effectiveness of this method in enhancing the identification of areas of interest in cardiac ablation through synthetic electrogram signals. You can find the paper [here](https://arxiv.org/abs/2608.15843).
> 
> 2. **Decoy Direction Optimization for LLMs**: In the context of language models, this paper discusses how ablation techniques, such as Refusal Feature Abl


**am20** `is it worth reading` → *clarify*

> I can help you find some relevant papers and articles worth reading! Would you like to see recommendations from your personalized feed, or are you interested in a specific topic related to deep learning, LLMs, or AI agents?
