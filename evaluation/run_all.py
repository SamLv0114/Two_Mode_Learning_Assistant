"""
Run every evaluation in sequence and print where the result files landed.

    python evaluation/run_all.py            # everything
    python evaluation/run_all.py routing    # one suite

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
}

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    wanted = sys.argv[1:] or list(RUNNERS)
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
