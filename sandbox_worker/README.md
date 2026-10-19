# Isolated practice worker

This is **not** a service for the production `docker-compose.yml`. Run it only
on a separate disposable host with no production volumes, database, Chroma,
application keys, or cloud instance role. Restrict inbound traffic to the API
host and terminate TLS before this service. Configure a long random
`PRACTICE_WORKER_TOKEN` on both ends and an immutable, reviewed Python runner
image via `PRACTICE_RUNNER_IMAGE` (pin an image digest).

The worker requires Docker CLI/daemon on its dedicated host. Install only
`sandbox_worker/requirements.txt` into a virtual environment. Set
`PRACTICE_RUNNER_IMAGE` to a digest-pinned Python 3.11 image already pulled onto
the worker host (`repository@sha256:...`). It will not pull an image during an
execution. The image must provide numpy: generated teaching programs may import the
standard library and the packages in `ALLOWED_THIRD_PARTY` (`src/agents/coding_practice.py`).
`runner.Dockerfile` builds that image:

```bash
docker build -f sandbox_worker/runner.Dockerfile -t <registry>/practice-runner:numpy-2.2.6 sandbox_worker
docker push <registry>/practice-runner:numpy-2.2.6      # prints the digest
# on the worker host
docker pull <registry>/practice-runner@sha256:<digest>
export PRACTICE_RUNNER_IMAGE=<registry>/practice-runner@sha256:<digest>
```

Rebuild, push and re-pin whenever `ALLOWED_THIRD_PARTY` changes. Start the worker
with `uvicorn sandbox_worker.app:app --host 127.0.0.1 --port 8090` behind a private
HTTPS proxy. The API uses
`PRACTICE_WORKER_URL=https://...` and never gets a Docker socket.

The API may set `PRACTICE_ENABLED=true` without this worker for unexecuted paper
drafts and public repository exploration. Set `PRACTICE_WORKER_URL` and
`PRACTICE_WORKER_TOKEN` only after the worker passes isolation checks; those
settings enable paper algorithm execution. Repository exploration reads a
pinned GitHub snapshot and does not execute repo code.

The worker returns the paper program's bounded output to the user. A
`CHECK:PASS` marker is a self-check by the generated program, not independent
proof that the paper algorithm or published experiments were reproduced.

Every execution uses a new container with `--network none`, read-only root,
non-root user, no capabilities, a limited tmpfs, and hard CPU/memory/PID/time
limits. Docker container boundaries alone are not a perfect security proof;
perform the attack and host-isolation tests in `docs/design/coding-agents.md`
before enabling the feature for users.
