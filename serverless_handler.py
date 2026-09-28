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

Also matches handler.py's architecture in a second way: the Docker image
stays lean (see Dockerfile) and the one genuinely heavy, rarely-changing
thing - SGLang's diffusion extras (diffusers, nvidia-nccl-cu13, etc.) -
gets installed ONCE onto the persistent network volume (/runpod-volume),
the same way handler.py's ensure_koboldcpp_engine() downloads the engine
binary onto the volume instead of baking it into the image. Any worker,
on any physical host, that mounts this same volume skips straight past
the install and starts fast - it's not a per-host Docker-layer-cache
thing, it's shared network storage.

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

VOLUME_DIR = "/runpod-volume"
VENV_DIR = os.path.join(VOLUME_DIR, "sglang-diffusion-venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")
VENV_SGLANG = os.path.join(VENV_DIR, "bin", "sglang")
INSTALL_MARKER = os.path.join(VENV_DIR, ".install_complete")
INSTALL_LOCK_DIR = os.path.join(VOLUME_DIR, ".sglang_venv_install.lock")


def ensure_sglang_diffusion_installed():
    """One-time (per volume, not per worker) install of SGLang's diffusion
    extras into a venv on the persistent volume - the direct equivalent of
    handler.py's ensure_koboldcpp_engine(): a no-op if it's already there,
    downloaded/installed once and shared by every future worker that
    mounts this volume, not baked into the Docker image."""
    if os.path.exists(INSTALL_MARKER):
        print("sglang diffusion venv already present on volume - skipping install.", flush=True)
        return

    # Simple cross-worker lock: mkdir is atomic, so this is safe against two
    # cold starts racing to build the venv at the same time. If another
    # worker is mid-install, wait for it rather than corrupt a shared venv.
    got_lock = False
    try:
        os.makedirs(INSTALL_LOCK_DIR)
        got_lock = True
    except FileExistsError:
        pass

    if not got_lock:
        print("Another worker is already installing the sglang venv - waiting...", flush=True)
        wait_start = time.time()
        while time.time() - wait_start < 1800:
            if os.path.exists(INSTALL_MARKER):
                print("Install finished by the other worker.", flush=True)
                return
            time.sleep(5)
        raise RuntimeError("Timed out waiting for another worker's sglang venv install.")

    try:
        print("Building sglang diffusion venv on persistent volume (one-time, first worker only)...", flush=True)
        os.makedirs(VENV_DIR, exist_ok=True)
        # --system-site-packages: inherit the base image's already-installed
        # torch/CUDA stack instead of redownloading it - only the diffusion
        # extras' own additional dependencies actually need installing.
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", VENV_DIR],
            check=True,
        )
        subprocess.run(
            [VENV_PYTHON, "-m", "pip", "install", "-e", "/sgl-workspace/sglang/python[diffusion]"],
            check=True,
        )
        with open(INSTALL_MARKER, "w") as f:
            f.write(str(time.time()))
        print("sglang diffusion venv install complete.", flush=True)
    finally:
        try:
            os.rmdir(INSTALL_LOCK_DIR)
        except OSError:
            pass


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
    ensure_sglang_diffusion_installed()

    # Same flags as run_test.py - RTX 5090 tier from SGLang's own
    # MiniMax-H3 cookbook (lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3).
    cmd = [
        VENV_SGLANG, "serve",
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
    print("Starting sglang server...", flush=True)
    subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stdout)

    timeout = 1800
    while time.time() - load_start < timeout:
        if _is_ready():
            elapsed = round(time.time() - load_start, 1)
            print(f"sglang ready after {elapsed}s (Load Time, includes any one-time venv install)", flush=True)
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
