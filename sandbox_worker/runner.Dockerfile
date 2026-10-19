# Runner image for generated teaching programs. It is not the worker's own image:
# an operator builds and pushes it, then pins it by digest in PRACTICE_RUNNER_IMAGE
# on the worker host (see README.md).
FROM python:3.11-slim

# Must match ALLOWED_THIRD_PARTY in src/agents/coding_practice.py.
RUN pip install --no-cache-dir numpy==2.2.6 \
    && python -I -c "import numpy; assert numpy.__version__ == '2.2.6'"

# The container gets one CPU and a 64-PID limit; OpenBLAS would otherwise start a
# thread per host core and can hit the PID limit on a large host.
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

USER 65534:65534
