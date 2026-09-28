#!/usr/bin/env python3
"""
RunPod Serverless handler for MiniMax H3 via SGLang.

Real production-pattern handler, same shape as minimax-h3-worker/handler.py
uses for koboldcpp: runpod.serverless.start() is called immediately, at
the very bottom of this file, with NOTHING blocking before it - matching
handler.py exactly, where runpod.serverless.start() is its literal last
line and start_kobold_if_needed() is never called at module level, only
from inside handler()/run_session() once a real job has already been
claimed. RunPod marks a worker "ready" the moment the module finishes
importing, not after a model finishes loading - so any heavy setup done
at import time (an earlier version of this file did exactly that) risks
the worker being killed/recycled before it's even had a chance to accept
a job. start_sglang() is called from inside handler() below for this
reason, not above runpod.serverless.start().

Fifth revision, and a real architectural pivot: earlier versions of this
file installed SGLang's diffusion extras at runtime onto the persistent
volume, to avoid re-pulling SGLang's official 13.8GB lmsysorg/sglang:dev
image on every cold start that landed on an uncached host. This version
instead uses a genuinely custom, minimal Dockerfile (see Dockerfile in
this repo) that installs SGLang + diffusion extras at BUILD time into a
purpose-built image - no runtime install, no venv, no cross-worker lock
needed at all, because the image itself is now small enough that a fresh
host paying the pull cost is a much smaller problem than it was against
the 13.8GB official image.

Sixth revision: HF_HOME is now pointed at the persistent volume (see
HF_CACHE_DIR below) so the ~60-80GB MiniMax-H3 weights are downloaded once
per volume, not once per cold start. --model-path is a bare HF repo id,
and SGLang resolves that through huggingface_hub.snapshot_download() with
no cache_dir override (confirmed by reading SGLang's own
model_loader/weight_utils.py) - so without this, it was falling back to
the CONTAINER's own ephemeral cache, meaning every fresh container
re-downloaded the whole model before the server could even start. This
was very likely the dominant slow part of "SGLang cold start," not
anything about SGLang's own inference speed.

Set as this endpoint's "Container start command": leave it EMPTY. The
Dockerfile's own CMD runs this file directly.

Job input (all optional, defaults match the production koboldcpp
baseline used throughout this investigation - 1280x736/175 frames/20 steps):
    {"input": {
        "prompt": "...",
        "duration_seconds": 7.29,
        "short_edge": 736,
        "aspect_ratio": "16:9",
        "num_inference_steps": 20,
        "seed": 1101
    }}

KNOWN GAP, same one flagged in run_test.py: SGLang's own docs don't
confirm the /v1/videos response shape (sync video vs. an async job id to
poll). If extraction fails, the raw response is returned instead so we
can see it and adjust _extract_video_b64/_poll_job to match reality.
"""
import base64
import os
import subprocess
import sys
import threading
import time

import requests
import runpod

SGLANG_HOST = "127.0.0.1"
SGLANG_PORT = 30010
BASE_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"
MODEL_PATH = "MiniMaxAI/MiniMax-H3"

# Same fix as handler.py's ensure_koboldcpp_engine()/VOLUME_DIR pattern,
# for the one piece this file was still missing: the model weights
# themselves. --model-path above is a bare HF repo id, and SGLang resolves
# that via huggingface_hub.snapshot_download() with no cache_dir override
# (confirmed by reading model_loader/weight_utils.py directly, not
# assumed) - meaning it falls back to HF_HOME/HUGGINGFACE_HUB_CACHE, which
# defaults to the CONTAINER's own ephemeral disk, not the persistent
# volume. Every cold start on a fresh container was therefore very
# plausibly re-downloading the full ~60-80GB model from HuggingFace before
# ever starting the server - the single biggest slow part, and not
# something SGLang-vs-koboldcpp speed has anything to do with. Pointing
# HF_HOME at the volume fixes this the same way KOBOLD_DIR living on
# VOLUME_DIR does for the engine: first cold start on a given volume pays
# the download once, every worker after that (on that volume) finds the
# weights already there - HF's own cache uses content-hash-linked local
# files, so no marker file is needed to detect "already downloaded" the
# way kobold's PyInstaller extraction required.
VOLUME_DIR = "/runpod-volume"
HF_CACHE_DIR = os.path.join(VOLUME_DIR, "hf-cache")


def _find_cuda_home():
    """The community-confirmed workaround for sgl-project/sglang#11333
    (see Dockerfile's own comment for the full story): point CUDA_HOME at
    the pip-installed nvidia-cuda-runtime package's own directory rather
    than relying on a system CUDA install. Belt-and-suspenders even with
    the upstream fix (PR #13089) merged - costs nothing if unneeded.
    Never raises: worst case sglang falls back to its own detection."""
    try:
        import nvidia.cuda_runtime
        return os.path.dirname(nvidia.cuda_runtime.__file__)
    except Exception:
        return None


def _is_ready():
    for path in ("/health", "/v1/models", "/get_model_info"):
        try:
            r = requests.get(f"{BASE_URL}{path}", timeout=3)
            if r.status_code < 500:
                return True
        except requests.exceptions.RequestException:
            continue
    return False


def start_sglang():
    if _is_ready():
        print("sglang already running.", flush=True)
        return None

    load_start = time.time()

    # Same flags as run_test.py - RTX 5090 tier from SGLang's own
    # MiniMax-H3 cookbook (lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3).
    cmd = [
        "sglang", "serve",
        "--model-path", MODEL_PATH,
        "--model-variant", "fl2va",
        "--performance-mode", "memory",
        "--layerwise-offload-components", "dit,text_encoder,vae",
        "--layerwise-resident-layers", "video_vae=36",
        "--dit-layerwise-resident-layers", "14",
        "--host", "0.0.0.0",
        "--port", str(SGLANG_PORT),
    ]
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    env["HF_HOME"] = HF_CACHE_DIR
    env["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
    print(f"Set HF_HOME={HF_CACHE_DIR} (model weights cached on persistent volume)", flush=True)
    cuda_home = _find_cuda_home()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
        print(f"Set CUDA_HOME={cuda_home}", flush=True)
    print("Starting sglang server...", flush=True)
    subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stdout)

    timeout = 1800
    while time.time() - load_start < timeout:
        if _is_ready():
            elapsed = round(time.time() - load_start, 1)
            print(f"sglang ready after {elapsed}s (Load Time)", flush=True)
            return elapsed
        time.sleep(2)
    raise RuntimeError(f"sglang server did not become ready within {timeout}s")


def sample_gpu_stats(stop_event, samples, interval=1.0):
    """Same pattern as minimax-h3-worker/handler.py's sample_gpu_stats and
    run_test.py's copy of it - periodic nvidia-smi snapshot,
    [unix_timestamp, gpu_util_pct, vram_used_mb]. Never raises."""
    while not stop_event.wait(interval):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            util_str, mem_str = out.stdout.strip().split(",")
            samples.append([round(time.time(), 1), int(util_str.strip()), int(mem_str.strip())])
        except Exception:
            pass


def _extract_video_b64(data):
    """Handles a few plausible response shapes: a direct URL (downloaded
    and base64-encoded here so the job result is fully self-contained), or
    already-base64-encoded bytes. Returns None if neither is found."""
    url = data.get("url") or data.get("video_url") or data.get("download_url")
    b64 = data.get("video_base64") or data.get("b64_json")

    if b64:
        return b64
    if url:
        try:
            vr = requests.get(url, timeout=120)
            vr.raise_for_status()
            return base64.b64encode(vr.content).decode()
        except Exception as e:
            print(f"Failed to download video from {url}: {e}", flush=True)
            return None
    return None


def _poll_job(job_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/v1/videos/{job_id}", timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Poll error: {e}", flush=True)
            time.sleep(2)
            continue
        status = data.get("status")
        if status in ("completed", "succeeded"):
            return data
        if status in ("failed", "error"):
            raise RuntimeError(f"SGLang job failed: {data}")
        time.sleep(2)
    raise TimeoutError(f"Polling {job_id} timed out after {timeout}s")


def handler(job):
    # Called only once this job has already been claimed/dispatched to
    # this worker - matches handler.py's own pattern exactly
    # (start_kobold_if_needed() is called from inside run_session(), which
    # handler() calls, never at module import time). runpod.serverless.start()
    # below registers this worker as claimable immediately; loading the
    # actual model only after a real job arrives is what keeps a slow first
    # load from ever counting against whatever "is this worker even alive"
    # window RunPod applies before a job is dispatched.
    load_time = start_sglang()

    inp = job.get("input", {})
    prompt = inp.get("prompt", "")

    request_received_at = time.time()

    stop_event = threading.Event()
    gpu_samples = []
    gpu_thread = threading.Thread(target=sample_gpu_stats, args=(stop_event, gpu_samples), daemon=True)
    gpu_thread.start()

    body = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "task": "t2va",
        "num_inference_steps": inp.get("num_inference_steps", 20),
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": inp.get("seed", 1101),
        "target": {
            "short_edge": inp.get("short_edge", 736),
            "aspect_ratio": inp.get("aspect_ratio", "16:9"),
            "duration_seconds": inp.get("duration_seconds", 7.29),
        },
    }

    dispatch_start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/v1/videos", json=body, timeout=1800)
        r.raise_for_status()
    except Exception as e:
        stop_event.set()
        return {"error": f"Request to sglang failed: {e}"}

    dispatch_lag_seconds = round(time.time() - dispatch_start, 2)
    resp = r.json()

    job_id = resp.get("id") or resp.get("job_id")
    if job_id and resp.get("status") not in ("completed", "succeeded", None):
        try:
            resp = _poll_job(job_id)
        except Exception as e:
            stop_event.set()
            return {"error": str(e), "raw_response": resp}

    video_b64 = _extract_video_b64(resp)

    stop_event.set()
    gpu_thread.join(timeout=5)

    total_seconds = round(time.time() - request_received_at, 2)

    result = {
        "cold_start_load_time_seconds": load_time,
        "dispatch_lag_seconds": dispatch_lag_seconds,
        "total_generation_seconds": total_seconds,
        "gpu_util_samples": gpu_samples,
    }
    if video_b64:
        result["video_base64"] = video_b64
    else:
        result["raw_response"] = resp
        result["note"] = (
            "Could not extract a video url/base64 field from SGLang's "
            "response - see raw_response above and tell me what the real "
            "field is so _extract_video_b64 can be fixed."
        )
    return result


runpod.serverless.start({"handler": handler})
