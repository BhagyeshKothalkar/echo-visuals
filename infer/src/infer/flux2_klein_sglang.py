"""
FLUX.2 Klein 4B on Modal + SGLang-Diffusion + GPU Memory Snapshots.

Deployment:
    modal deploy flux2_klein_sglang.py

After deployment, Modal will expose a web endpoint for SGLang's OpenAI-compatible API.

Text-to-image:
    POST /v1/images/generations
    {
      "model": "black-forest-labs/FLUX.2-klein-4B",
      "prompt": "a photorealistic red fox in snow",
      "size": "1024x1024",
      "num_inference_steps": 4,
      "response_format": "b64_json",
      "seed": 42
    }

Notes:
- The default model is the official BFL Diffusers-format BF16 checkpoint.
- BFL also publishes FLUX.2-klein-4b-fp8 as a 4.07 GB single safetensor.
  Current SGLang native Klein registration is for the full Diffusers repo,
  while current SGLang ModelOpt-FP8 documentation targets Hopper/SM90+.
  Therefore the L4-safe default here is the official BF16 repo.
- Keep DIT GPU-resident on an L4; CPU offload adds PCIe transfers to every step.
- The snapshot lifecycle starts SGLang, warms a production-shaped request,
  releases GPU occupation, and snapshots the initialized process. On restore,
  it resumes GPU occupation before serving.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "flux2-klein-sglang"

# Reproducible SGLang image: pinned nightly published by the SGLang project.
# Change this intentionally when upgrading SGLang.
SGLANG_IMAGE = "lmsysorg/sglang:nightly-dev-cu13-20260827-20621aa1"

# MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_ID = "feizhai123/flux2-klein-4b-modelopt-fp8"
# Set this to a commit SHA once you have approved a specific model revision.
MODEL_REVISION = None

# Official BFL FP8 repository (single-file checkpoint):
#   black-forest-labs/FLUX.2-klein-4b-fp8
#
# It is intentionally NOT the default here because it is not the same
# Diffusers repo layout as MODEL_ID.
FP8_SINGLE_FILE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4b-fp8"

PORT = 8000
MINUTES = 60

# Start conservatively for an L4: one request per replica.
# Raise only after measuring memory and image correctness under concurrency.
MAX_INPUTS = 1
TARGET_INPUTS = 1

# Snapshot/warmup shape. Keep this equal to your dominant production shape
# so compile/cuda-graph work is captured before snapshotting.
WARMUP_WIDTH = 1024
WARMUP_HEIGHT = 1024
WARMUP_STEPS = 4
WARMUP_SEED = 42
WARMUP_PROMPT = (
    "A photorealistic red fox standing in fresh snow in a pine forest, "
    "soft winter sunlight, natural colors, detailed fur"
)

HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name(
    f"{APP_NAME}-hf-cache",
    create_if_missing=True,
)

app = modal.App(APP_NAME)

# The SGLang image contains the CUDA/PyTorch/SGLang runtime.
# requests is added for the local lifecycle calls.
image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .run_commands("rm -rf /root/.cache/huggingface")
    .uv_pip_install("requests>=2.32,<3")
    .env(
        {
            # Modal specifically recommends this for TorchInductor +
            # GPU-memory-snapshot compatibility.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "HF_HOME": HF_CACHE_PATH,
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

with image.imports():
    import requests


def _wait_for_health(
    process: subprocess.Popen, timeout_s: float = 15 * MINUTES
) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{PORT}/health"
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited during startup with return code {process.returncode}"
            )

        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return
        except Exception as exc:  # health endpoint is expected to fail until ready
            last_error = exc

        time.sleep(1)

    raise TimeoutError(
        f"SGLang did not become healthy within {timeout_s:.0f}s; "
        f"last error={last_error!r}"
    )


def _post(path: str, payload: dict | None = None, timeout_s: float = 120.0) -> dict:
    response = requests.post(
        f"http://127.0.0.1:{PORT}{path}",
        json=payload or {},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json() if response.content else {}


def _warmup() -> None:
    payload = {
        "model": MODEL_ID,
        "prompt": WARMUP_PROMPT,
        "size": f"{WARMUP_WIDTH}x{WARMUP_HEIGHT}",
        "num_inference_steps": WARMUP_STEPS,
        "response_format": "b64_json",
        "seed": WARMUP_SEED,
        "n": 1,
    }

    # Two passes: the first tends to trigger lazy initialization/compilation;
    # the second confirms that the steady-state request path is hot.
    for i in range(2):
        print(f"Warmup request {i + 1}/2...")
        response = requests.post(
            f"http://127.0.0.1:{PORT}/v1/images/generations",
            json=payload,
            timeout=15 * MINUTES,
        )
        response.raise_for_status()


def _release_memory() -> None:
    # SGLang's diffusion sleep/wake API is designed to release model/cache
    # GPU occupation while keeping the serving process alive.
    print("Releasing SGLang GPU memory for Modal snapshot...")
    _post(
        "/release_memory_occupation",
        {
            "tags": ["weights", "cache"],
        },
        timeout_s=5 * MINUTES,
    )


def _resume_memory() -> None:
    print("Resuming SGLang GPU memory after snapshot restore...")
    _post(
        "/resume_memory_occupation",
        {
            "tags": ["weights"],
        },
        timeout_s=5 * MINUTES,
    )


@app.cls(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    timeout=15 * MINUTES,
    scaledown_window=60,
)
@modal.concurrent(
    target_inputs=TARGET_INPUTS,
    max_inputs=MAX_INPUTS,
)
class SGLang:
    @modal.enter(snap=True)
    def startup(self) -> None:
        """Start, fully initialize, warm, then put SGLang to sleep for snapshot."""

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--num-gpus",
            "1",
            "--dit-cpu-offload",
            "false",
            "--batching-mode",
            "dynamic",
            "--batching-max-size",
            "1",
        ]

        if MODEL_REVISION:
            cmd.extend(["--revision", MODEL_REVISION])

        print("Launching SGLang:")
        print(" ".join(cmd))

        self.process = subprocess.Popen(cmd)

        _wait_for_health(self.process)
        _warmup()
        _release_memory()

    @modal.enter(snap=False)
    def restore(self) -> None:
        """Wake the SGLang engine after Modal restores the memory snapshot."""
        _resume_memory()

        # Defensive readiness check after restore.
        _wait_for_health(self.process, timeout_s=60)

    @modal.web_server(
        port=PORT,
        startup_timeout=15 * MINUTES,
    )
    def serve(self) -> None:
        # SGLang itself owns the HTTP listener.
        pass

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


@app.local_entrypoint()
def main() -> None:
    """Smoke test against the deployed snapshot-enabled service."""
    print("Deploy first with:")
    print(f"  modal deploy {Path(__file__).name}")
    print()
    print("Then call:")
    print("  POST /v1/images/generations")
    print(f"  model={MODEL_ID}")
    print(f"  size={WARMUP_WIDTH}x{WARMUP_HEIGHT}")
    print(f"  steps={WARMUP_STEPS}")
