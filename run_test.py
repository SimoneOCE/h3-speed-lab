#!/usr/bin/env python3
"""
h3-speed-lab: SGLang serving-stack speed test harness for MiniMax H3.

Companion to the ComfyUI+SageAttention pod test from the same investigation —
same manual, phase-by-phase workflow, run by hand on a RunPod pod:

    python run_test.py start      # launch sglang, wait for ready -> Load Time
    python run_test.py warmup     # throwaway low-res/1-step gen (manual, your call)
    python run_test.py generate   # real full-settings gen, saves an mp4

Every phase appends a timestamped entry (+ nvidia-smi GPU samples, same
[unix_ts, gpu_util_pct, vram_used_mb] shape as handler.py's own
sample_gpu_stats) to session_log.json, so these numbers line up directly
against the koboldcpp/ComfyUI numbers from earlier in the same investigation.

KNOWN GAP, on purpose rather than by accident: SGLang's own MiniMax-H3
cookbook page documents the server launch flags and most request fields,
but does NOT document the /v1/videos response shape (sync bytes vs. an
async job to poll) or the exact health-check endpoint. Rather than guess
and silently get it wrong, `generate`/`warmup` print the raw response the
first time so we can see what's actually there and fix the parsing below
against reality. Before your first real `generate` run, it's worth
checking http://<pod>:30010/docs or /openapi.json in a browser too, if
SGLang exposes it (most FastAPI-based servers do) - that's the fastest way
to confirm the real contract before spending a full timed run on a guess.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time

import requests

SGLANG_HOST = "127.0.0.1"
SGLANG_PORT = 30010
BASE_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "session_log.json")
PID_FILE = os.path.join(SCRIPT_DIR, "sglang_server.pid")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

MODEL_PATH = "MiniMaxAI/MiniMax-H3"

# Matches the production koboldcpp baseline used for the ComfyUI comparison
# earlier in this investigation: 1280x736 (16:9, the site's "SD"/720p
# tier), ~175 frames @ 24fps = 7.29s duration, 20 steps.
REAL_GEN_PARAMS = {
    "duration_seconds": 7.29,
    "short_edge": 736,
    "aspect_ratio": "16:9",
    "num_inference_steps": 20,
}

# Deliberately tiny - mirrors the low-res/1-step throwaway warmup pattern
# used on both koboldcpp and the ComfyUI pod test.
WARMUP_GEN_PARAMS = {
    "duration_seconds": 1,
    "short_edge": 256,
    "aspect_ratio": "16:9",
    "num_inference_steps": 1,
}

DEFAULT_PROMPT = "A single steady shot of calm ocean waves, cinematic lighting."


def log_event(event, **fields):
    entry = {"ts": round(time.time(), 2), "event": event, **fields}
    print(f"[{time.strftime('%H:%M:%S')}] {event}: {fields}")
    data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE) as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return entry


def sample_gpu_stats(stop_event, samples, interval=1.0):
    """Same pattern as minimax-h3-worker/handler.py's sample_gpu_stats:
    periodic nvidia-smi snapshot, [unix_timestamp, gpu_util_pct,
    vram_used_mb]. Never raises - a missing/failing nvidia-smi only skips
    that one sample, never breaks or slows the actual generation."""
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


def is_server_ready():
    """Best-effort readiness probe. SGLang's exact health endpoint for the
    diffusion/video pipeline isn't confirmed in the docs, so this tries a
    few plausible candidates and accepts any non-5xx response as ready."""
    for path in ("/health", "/v1/models", "/get_model_info"):
        try:
            r = requests.get(f"{BASE_URL}{path}", timeout=3)
            if r.status_code < 500:
                return True
        except requests.exceptions.RequestException:
            continue
    return False


def cmd_start(args):
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        if os.path.exists(f"/proc/{pid}"):
            print(f"sglang server already running (pid {pid}).")
            return
        os.remove(PID_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_event("load_start")

    # Flags match the RTX 5090 example from SGLang's own MiniMax-H3
    # cookbook (lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3).
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

    log_path = os.path.join(OUTPUT_DIR, "sglang_server.log")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"sglang server starting (pid {proc.pid}), logging to {log_path}")
    print("Waiting for it to report ready - this is Load Time...")

    start = time.time()
    timeout = args.timeout
    while time.time() - start < timeout:
        if proc.poll() is not None:
            log_event("load_failed", returncode=proc.returncode)
            print(f"Server process exited early (code {proc.returncode}). Check {log_path}.")
            sys.exit(1)
        if is_server_ready():
            elapsed = round(time.time() - start, 1)
            log_event("load_ready", load_time_seconds=elapsed)
            print(f"\n=== Load Time: {elapsed}s ===\n")
            return
        time.sleep(2)

    log_event("load_timeout", timeout=timeout)
    print(
        f"Timed out after {timeout}s waiting for readiness. Check {log_path} - "
        f"the health endpoints this script probes (/health, /v1/models, "
        f"/get_model_info) are a best guess, not confirmed against SGLang's "
        f"actual diffusion API, so a slow-but-healthy server could also trip this."
    )


def _run_generation(label, params, prompt):
    dispatch_start = time.time()
    log_event(f"{label}_dispatch_start")

    stop_event = threading.Event()
    gpu_samples = []
    gpu_thread = threading.Thread(target=sample_gpu_stats, args=(stop_event, gpu_samples), daemon=True)
    gpu_thread.start()

    body = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "task": "t2va",
        "num_inference_steps": params["num_inference_steps"],
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": 1101,
        "target": {
            "short_edge": params["short_edge"],
            "aspect_ratio": params["aspect_ratio"],
            "duration_seconds": params["duration_seconds"],
        },
    }

    try:
        r = requests.post(f"{BASE_URL}/v1/videos", json=body, timeout=1800)
        r.raise_for_status()
    except Exception as e:
        stop_event.set()
        log_event(f"{label}_request_failed", error=str(e))
        print(f"Request failed: {e}")
        return None

    resp = r.json()
    # Response shape (sync video vs. an async job id to poll) is NOT
    # confirmed by SGLang's docs - print it raw so we can see reality and
    # fix the parsing below on the next run if it doesn't match.
    print("Raw /v1/videos response (first 2000 chars):")
    print(json.dumps(resp, indent=2)[:2000])

    job_id = resp.get("id") or resp.get("job_id")
    if job_id and resp.get("status") not in ("completed", "succeeded", None):
        print(f"Response looks like an async job (id={job_id}, status={resp.get('status')}) - polling...")
        video_path = _poll_job(job_id, label)
    else:
        video_path = _extract_video(resp, label)

    stop_event.set()
    gpu_thread.join(timeout=5)

    total = round(time.time() - dispatch_start, 2)
    log_event(f"{label}_done", total_seconds=total, gpu_samples=gpu_samples, video_path=video_path)
    print(f"\n=== {label} total: {total}s ===\n")
    return video_path


def _poll_job(job_id, label):
    for _ in range(900):  # up to 30 min at 2s intervals
        try:
            r = requests.get(f"{BASE_URL}/v1/videos/{job_id}", timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Poll error: {e}")
            time.sleep(2)
            continue
        status = data.get("status")
        log_event(f"{label}_poll", status=status)
        if status in ("completed", "succeeded"):
            return _extract_video(data, label)
        if status in ("failed", "error"):
            print(f"Job failed: {data}")
            return None
        time.sleep(2)
    print("Polling timed out.")
    return None


def _extract_video(data, label):
    """Handles a few plausible response shapes: a direct URL, a local file
    path, or base64-encoded bytes. Unconfirmed against the real API - if
    none of these match, the raw response printed above has what's
    actually there; tell me the real field name and this gets a one-line fix."""
    url = data.get("url") or data.get("video_url") or data.get("download_url")
    b64 = data.get("video_base64") or data.get("b64_json")
    out_path = os.path.join(OUTPUT_DIR, f"{label}_{int(time.time())}.mp4")

    if url:
        try:
            vr = requests.get(url, timeout=120)
            vr.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(vr.content)
            print(f"Saved video to {out_path}")
            return out_path
        except Exception as e:
            print(f"Failed to download video from {url}: {e}")
            return None
    if b64:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"Saved video to {out_path}")
        return out_path

    print(
        "Could not find a url/base64 video field in the response - check "
        "the raw response printed above and tell me what the real field is."
    )
    return None


def cmd_warmup(args):
    _run_generation("warmup", WARMUP_GEN_PARAMS, "test")


def cmd_generate(args):
    prompt = args.prompt or DEFAULT_PROMPT
    _run_generation("generate", REAL_GEN_PARAMS, prompt)


def cmd_status(args):
    print("Server ready:", is_server_ready())
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            data = json.load(f)
        print(f"{len(data)} logged events. Last 5:")
        for e in data[-5:]:
            print(" ", e)
    else:
        print("No session_log.json yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="Launch sglang server, wait for ready, log Load Time")
    p_start.add_argument("--timeout", type=int, default=900, help="seconds to wait for readiness (default 900)")
    p_start.set_defaults(func=cmd_start)

    p_warmup = sub.add_parser("warmup", help="Throwaway low-res/1-step generation")
    p_warmup.set_defaults(func=cmd_warmup)

    p_gen = sub.add_parser("generate", help="Real full-settings generation, saves an mp4")
    p_gen.add_argument("--prompt", type=str, default=None)
    p_gen.set_defaults(func=cmd_generate)

    p_status = sub.add_parser("status", help="Check server readiness + recent log entries")
    p_status.set_defaults(func=cmd_status)

    parsed = parser.parse_args()
    parsed.func(parsed)
