"""
Run every evaluation in sequence and print where the result files landed.

    python evaluation/run_all.py            # established suites
    python evaluation/run_all.py routing    # one suite
    python evaluation/run_all.py generation trajectory  # human-verified suites

Retrieval and ambiguity read the production ChromaDB snapshot at
VECTOR_DB_DIR (see evaluation/README.md); routing needs no corpus.
Ambiguity makes real agent calls and is the slow one — budget a few minutes.
"""
import subprocess
import sys
import time
from pathlib import Path

RUNNERS = {
    "routing": "runners/run_routing_eval.py",
    "retrieval": "runners/run_retrieval_eval.py",
    "ambiguity": "runners/run_ambiguity_eval.py",
    "fact_dedup": "runners/run_fact_dedup_eval.py",
    "memory_compression": "runners/run_memory_compression_eval.py",
    "generation": "runners/run_generation_eval.py",
    "trajectory": "runners/run_trajectory_eval.py",
    "prompt_injection": "runners/run_prompt_injection_eval.py",
    "supervisor": "runners/run_supervisor_eval.py",
    "chunking": "runners/run_chunking_eval.py",
    "context_budget": "runners/run_context_budget_eval.py",
    "cache": "runners/run_cache_eval.py",
}

# Generation and trajectory require a 50–100 case human-verified golden set.
# They are selectable, but must not silently turn pending candidates into
# published scores when running the existing suites by default.
DEFAULT_RUNNERS = ["routing", "retrieval", "ambiguity", "fact_dedup", "memory_compression"]

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    wanted = sys.argv[1:] or DEFAULT_RUNNERS
    unknown = [w for w in wanted if w not in RUNNERS]
    if unknown:
        sys.exit(f"unknown suite(s): {unknown}. available: {list(RUNNERS)}")

    failures = []
    for name in wanted:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(EVAL_DIR / RUNNERS[name])],
                              cwd=EVAL_DIR.parent)
        took = time.time() - t0
        if proc.returncode != 0:
            failures.append(name)
            print(f"[{name}] FAILED after {took:.0f}s")
        else:
            print(f"[{name}] done in {took:.0f}s")

    print(f"\nresults in {EVAL_DIR / 'results'}")
    if failures:
        sys.exit(f"failed: {failures}")


if __name__ == "__main__":
    main()
