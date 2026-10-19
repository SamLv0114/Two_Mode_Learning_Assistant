"""Small Docker-backed worker for an isolated host, never the production EC2.

The Docker daemon is powerful. This service must run on a separate host with
no production volumes, credentials or instance role and behind authenticated
TLS/private networking. The API container must not receive a Docker socket.
"""
import hmac
import os
import selectors
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="ResearchMate isolated practice worker", docs_url=None, redoc_url=None)
MAX_OUTPUT = 16_000
MAX_SCRIPT = 20_000
MAX_SECONDS = 8
IMAGE = os.getenv("PRACTICE_RUNNER_IMAGE", "")
SLOTS = threading.BoundedSemaphore(2)


def require_token(authorization: str = Header(default="")):
    expected = os.getenv("PRACTICE_WORKER_TOKEN", "")
    token = authorization.removeprefix("Bearer ")
    if not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


class ExecuteRequest(BaseModel):
    code: str = Field(min_length=1, max_length=MAX_SCRIPT)


def docker_command(directory: Path, container_name: str) -> list[str]:
    """All arguments fixed by operator; user input is only file contents."""
    return [
        "docker", "run", "--rm", "--name", container_name,
        "--pull", "never",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "64",
        "--memory", "256m", "--memory-swap", "256m", "--cpus", "1",
        "--user", "65534:65534", "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m",
        "--mount", f"type=bind,src={directory},dst=/work,readonly",
        "--workdir", "/work", IMAGE, "python", "-I", "/work/main.py",
    ]


def _terminate(container_name: str):
    subprocess.run(["docker", "rm", "-f", container_name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)


def run_script(code: str) -> dict:
    if "@sha256:" not in IMAGE:
        raise HTTPException(status_code=503, detail="A digest-pinned runner image is required")
    if len(code.encode("utf-8")) > MAX_SCRIPT:
        raise HTTPException(status_code=413, detail="Code is too large")
    with tempfile.TemporaryDirectory(prefix="practice-") as tmp:
        path = Path(tmp)
        # TemporaryDirectory is mode 0700; the non-root container user needs
        # traverse permission on this dedicated, read-only mounted directory.
        path.chmod(0o755)
        script = path / "main.py"
        script.write_text(code, encoding="utf-8")
        script.chmod(0o644)
        container_name = "practice-" + uuid.uuid4().hex[:20]
        process = subprocess.Popen(docker_command(path, container_name),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        output = bytearray()
        started = time.monotonic()
        reason = None
        try:
            while process.poll() is None:
                if time.monotonic() - started > MAX_SECONDS:
                    reason = "timeout"
                    break
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 4096)
                    output.extend(chunk)
                    if len(output) > MAX_OUTPUT:
                        reason = "output_limit"
                        break
                if reason:
                    break
            if reason:
                _terminate(container_name)
                process.kill()
            else:
                output.extend(process.stdout.read(MAX_OUTPUT + 1 - len(output)))
        finally:
            selector.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _terminate(container_name)
                process.kill()
                process.wait(timeout=2)
        text = bytes(output[:MAX_OUTPUT]).decode("utf-8", errors="replace")
        return {"returncode": process.returncode, "stdout": text, "stderr": "",
                "timed_out": reason == "timeout", "error": reason,
                "passed": reason is None and process.returncode == 0 and "CHECK:PASS" in text,
                "duration_ms": int((time.monotonic() - started) * 1000)}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/execute", dependencies=[Depends(require_token)])
def execute(request: ExecuteRequest):
    if not SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Sandbox is busy")
    try:
        return run_script(request.code)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Sandbox runtime unavailable") from exc
    finally:
        SLOTS.release()

