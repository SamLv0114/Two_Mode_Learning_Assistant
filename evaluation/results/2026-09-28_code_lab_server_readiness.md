# Coding Agent server acceptance: 2026-09-28

## Production readiness check (read only)

- Server checkout and running API OCI revision: `0d24af6bc5c20643cc58a01829b071c28e94a21a`.
- API `/health` returned HTTP 200.
- Running API image does not contain `src/agents/coding_practice.py` or `src/api/routers/practice.py`; `langgraph` is not installed there.
- The API has no `PRACTICE_ENABLED`, `PRACTICE_WORKER_URL`, or `PRACTICE_WORKER_TOKEN` setting in its container environment. Compose services are `db`, `redis`, `api`, `prometheus`, `frontend`, and `nginx`; no separate worker is deployed.
- This is the previously deployed application, not the uncommitted local Coding Agent implementation. Full browser and worker acceptance would test different code if run against it.

## Limited real-model checks (local, not server acceptance)

- Model: local configured `gpt-4o-mini` through OpenAI. One public repository run on [`karpathy/micrograd`](https://github.com/karpathy/micrograd) at commit `7bc720e951fe422b8f8814aa5aa1b64121d26b4c` produced an architecture answer, two inspected entry points, and file roles. The paper smoke output is in `2026-09-28_code_lab_paper_draft_smoke.jsonl`; repository output is in `2026-09-28_code_lab_repo_smoke.jsonl`. Neither has been human judged.
- Public arXiv paper `2608.14407` was refused by the preflight because the retrieved material described a broad survey rather than one implementable algorithm. This is one sample, not an estimated refusal accuracy.
- Public [Adam paper](https://arxiv.org/pdf/1412.6980) was accepted by preflight. Initial vector excerpts missed its published default parameters and the model incorrectly listed them as unknown. Scoped exact-text retrieval now supplies the Algorithm 1 paragraph. A repeat real-model preflight cited the correct values (`0.001`, `0.9`, `0.999`, `10^-8`) in `2026-09-28_code_lab_preflight_anchor_check.json`. This repeat used actual PDF text and Chroma exact-text lookup, with vector hits mocked; it is a component check, not a full live run.
- After adding draft-only mode, one real-model Adam run produced a 1,512-character standard-library Python draft and three learner questions. Its state records `generated_unexecuted`, zero executions, no timeline, and `FAITHFUL:UNCERTAIN`; see `2026-09-28_code_lab_paper_draft_smoke.jsonl`. The first attempt used `numpy` despite the prompt; a static dependency check and one model rewrite removed it. The code was **not executed or human verified**.
- The smoke checks did not execute paper code, validate worker isolation, compare an official repository module, or run a browser journey. No quality, attack resistance, or browser success rate can be inferred from them.

## Blockers for the three requested tracks

1. Real-model paper drafts can now be generated without a worker, but validating execution and repair requires an independently validated HTTPS worker. Paper and repository quality still need curated, human-checked cases and scoring notes; the smoke checks are insufficient.
2. Isolation tests must execute untrusted samples on a dedicated disposable host without production data, keys, Docker volumes, or instance credentials. Do not run those attack tests inside the production API host.
3. Browser end-to-end tests of draft and repository flows require a staging deployment of this local Coding Agent revision and its API configuration. The edit-and-run browser flow additionally requires the isolated worker. The current production image lacks the feature entirely.

The feature remains disabled. Nothing was pushed, deployed, or changed on the production server during this check.
