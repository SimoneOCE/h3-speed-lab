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

KNOWN GAP, same one flagged in run_test.py: SGLang's own docs don't
confirm the /v1/videos response shape (sync video vs. an async job id to
poll). handler() below returns the raw response for the first real job
so we can see it and adjust to a real video-returning shape once we do.
"""
import os
import subprocess
import sys
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
        print("sglang already running.")
        return

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
            print(f"sglang ready after {round(time.time() - start, 1)}s", flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f"sglang server did not become ready within {timeout}s")


# Runs once per cold start (when RunPod imports this module), not per job -
# this is the Load Time we've been measuring throughout this investigation.
start_sglang()


def handler(job):
    inp = job.get("input", {})
    prompt = inp.get("prompt", "")

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

    r = requests.post(f"{BASE_URL}/v1/videos", json=body, timeout=1800)
    r.raise_for_status()
    resp = r.json()

    return {"raw_response": resp}


runpod.serverless.start({"handler": handler})
