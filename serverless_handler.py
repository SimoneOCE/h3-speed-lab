#!/usr/bin/env python3
"""
RunPod Serverless handler for MiniMax H3 via SGLang.

This is the real handler RunPod's worker SDK invokes per job - not a
manual test script like run_test.py. Mirrors the exact pattern
minimax-h3-worker/handler.py already uses in production for koboldcpp:
start the model server once when the worker cold-starts (module import
time, below), then process each job by calling its HTTP API and
returning the result. runpod.serverless.start() blocks forever, pulling
jobs from RunPod's queue and calling handler() for each one.

Set as this endpoint's "Container start command":
    bash -c "python -m pip install -e '/sgl-workspace/sglang/python[diffusion]' && pip install runpod requests && git clone https://github.com/SimoneOCE/h3-speed-lab.git /tmp/h3-speed-lab && python /tmp/h3-speed-lab/serverless_handler.py"

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

Response includes real timing (Dispatch Lag = request received -> first
GPU activity, Generation Duration = total time), GPU-util samples (same
[unix_ts, gpu_util_pct, vram_used_mb] shape as handler.py's own
sample_gpu_stats), and the video itself as base64 if SGLang's response
shape allows extracting it.

KNOWN GAP, same one flagged in run_test.py: SGLang's own docs don't
confirm the /v1/videos response shape (sync video vs. an async job id to
poll). If extraction fails, the raw response is returned instead so we
can see it and adjust _extract_video/_poll_job to match reality.
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
    print("Starting sglang server (cold start)...", flush=True)
    subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stdout)

    start = time.time()
    timeout = 1800
    while time.time() - start < timeout:
        if _is_ready():
            elapsed = round(time.time() - start, 1)
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


# Runs once per cold start (when RunPod imports this module), not per job -
# this is the Load Time we've been measuring throughout this investigation.
COLD_START_LOAD_TIME = start_sglang()


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
        "cold_start_load_time_seconds": COLD_START_LOAD_TIME,
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
