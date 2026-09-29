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

Sixth revision: HF_HOME is pointed at the persistent volume (see
HF_CACHE_DIR below) so the ~60-80GB MiniMax-H3 weights are downloaded once
per volume, not once per cold start. --model-path is a bare HF repo id,
and SGLang resolves that through huggingface_hub.snapshot_download() with
no cache_dir override (confirmed by reading SGLang's own
model_loader/weight_utils.py) - so without this, it was falling back to
the CONTAINER's own ephemeral cache, meaning every fresh container
re-downloaded the whole model before the server could even start.

Seventh revision, and a reversal of the "genuinely custom minimal image"
approach (Fifth revision): that baked SGLang + its diffusion extras into
the Docker image at BUILD time, shrinking the shipped image from SGLang's
official 13.8GB down to 6.1GB. That helped, but a real deploy then showed
one single 4.11GB layer taking ~11.5 minutes to download on a fresh
RunPod host (~6MB/s effective, vs. the ~250MB/s this same image pushed
at). Shrinking the image never actually fixed the real problem: RunPod's
Docker layer cache is PER PHYSICAL HOST, not fleet-wide, so a worker
landing on a host that's never pulled this image before pays the full
cost again no matter how small the image is. The persistent volume has
the opposite property - it follows the worker wherever it lands, it isn't
tied to one host's local disk - so ensure_sglang_installed() below
installs SGLang there instead, once per volume, the same way this file's
own HF_HOME model-weight caching (and handler.py's own
ensure_koboldcpp_engine()/VOLUME_DIR pattern) already works. The Dockerfile
is back to a minimal, kobold-shaped image: just enough to run this file
and call runpod.serverless.start() immediately.

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
import glob
import os
import shutil
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

# --- SGLang install, on the volume, once per volume (Seventh revision) ---
# Pinned to a specific commit, not an unpinned `main` clone - same
# reproducibility discipline the old Dockerfile used, and needed for the
# same reason: refresh deliberately by checking
# https://github.com/sgl-project/sglang/commits/main for a new SHA, not as
# a side effect of an unrelated change.
SGLANG_COMMIT = "2e7e0802f4f3edc94702933675a71d6c7ef4f24f"
SGLANG_SRC_DIR = "/tmp/sglang_src"
SGLANG_VENV_DIR = os.path.join(VOLUME_DIR, "sglang_venv")
SGLANG_VENV_PYTHON = os.path.join(SGLANG_VENV_DIR, "bin", "python3")
SGLANG_VENV_BIN_DIR = os.path.join(SGLANG_VENV_DIR, "bin")
SGLANG_VENV_SGLANG = os.path.join(SGLANG_VENV_BIN_DIR, "sglang")
INSTALL_MARKER = os.path.join(SGLANG_VENV_DIR, ".install_complete")
# mkdir is atomic across processes/workers sharing the same volume - the
# one worker whose mkdir succeeds is the installer, every other worker
# that lands on this volume at the same time waits below instead of
# racing to install into the same venv simultaneously.
INSTALL_LOCK_DIR = os.path.join(VOLUME_DIR, ".sglang_install_lock")
# A lock only cleaned up in a Python finally block never gets removed on
# a hard kill - a dead lock from an earlier failed cold start would
# otherwise block every future worker on this volume indefinitely. Real
# installs (venv create + git clone + pip install of SGLang's full
# dependency tree, including PyTorch) can genuinely take several minutes,
# so this is generous on purpose - self-heals past that without being so
# short it fights a legitimately slow install.
STALE_LOCK_SECONDS = 1800

# handler.py's own COLD_START_MAX_WAIT_SECONDS/WARM_RESTART_MAX_WAIT_SECONDS
# split exists because of a real incident: a network-volume read once
# measured as slow as 19MB/s, blowing straight through a 180s timeout on a
# file that was already on disk and not actually broken - "already
# downloaded" turned out not to mean "loads quickly." MiniMax-H3's weights
# are roughly 60-80GB; at that same bad-case throughput a genuine first
# download alone could take over an hour. These numbers are provisional
# (no real run has confirmed our own network-volume throughput yet),
# worth tightening once we have an actual timing number back. This is for
# the SGLANG SERVER's own readiness wait, separate from - and starting
# only after - ensure_sglang_installed() below finishes.
COLD_START_MAX_WAIT_SECONDS = 3600
WARM_RESTART_MAX_WAIT_SECONDS = 1200

# Where HF's own hub cache lands a fully-downloaded MiniMax-H3 snapshot -
# used only to tell "this is a genuine first-ever download" apart from
# "the weights are already on the volume" BEFORE starting sglang, so the
# right timeout budget above gets picked. Not used to skip anything
# ourselves - HF's own cache already does that internally.
HF_CACHE_DIR = os.path.join(VOLUME_DIR, "hf-cache")
_MODEL_CACHE_MARKER = os.path.join(HF_CACHE_DIR, "hub", "models--" + MODEL_PATH.replace("/", "--"))


def ensure_sglang_installed():
    """Installs SGLang + its diffusion extras into a venv on the
    persistent volume, once per volume - not baked into the Docker image
    at all anymore. See this module's own Seventh-revision docstring note
    for why: a baked-in image, even shrunk to 6.1GB, is still a real,
    per-HOST-cached pull cost (a real deploy showed one layer alone taking
    ~11.5 minutes on a fresh host). The volume has no such per-host cost -
    it's the same volume wherever the worker lands - so paying the
    install cost once here, the same way handler.py's own
    ensure_koboldcpp_engine() already does for koboldcpp's own engine, is
    the actual fix.
    """
    if os.path.exists(INSTALL_MARKER):
        return

    os.makedirs(VOLUME_DIR, exist_ok=True)

    got_lock = False
    try:
        os.mkdir(INSTALL_LOCK_DIR)
        got_lock = True
    except FileExistsError:
        pass

    if not got_lock:
        print("Another worker is already installing sglang onto this volume - waiting...", flush=True)
        # BOTH wait loops below are bounded by STALE_LOCK_SECONDS - an
        # earlier version of the second one had no timeout at all, which
        # meant a worker that died mid-install (without ever writing
        # INSTALL_MARKER or getting a chance to clean up its lock) would
        # leave every future worker on this volume polling forever, never
        # returning from handler() - burning GPU billing indefinitely with
        # no way for RunPod to know the job was ever "done." Real risk,
        # not hypothetical: this exact class of dead-lock already happened
        # once earlier in this investigation, on the old runtime-venv
        # architecture.
        _wait_deadline = time.time() + STALE_LOCK_SECONDS
        while os.path.isdir(INSTALL_LOCK_DIR) and not os.path.exists(INSTALL_MARKER):
            try:
                lock_age = time.time() - os.path.getmtime(INSTALL_LOCK_DIR)
            except FileNotFoundError:
                break
            if lock_age > STALE_LOCK_SECONDS:
                print(f"Install lock is stale ({lock_age:.0f}s old) - assuming the worker that held it died. Self-healing.", flush=True)
                break
            if time.time() > _wait_deadline:
                raise RuntimeError(
                    f"Gave up waiting for another worker's sglang install after {STALE_LOCK_SECONDS}s "
                    "(lock never went stale by its own mtime, but this wait has its own hard ceiling too)."
                )
            time.sleep(3)
        if os.path.exists(INSTALL_MARKER):
            return
        try:
            os.mkdir(INSTALL_LOCK_DIR)
            got_lock = True
        except FileExistsError:
            # Someone else grabbed it first, or a stale one just got
            # cleaned up by another worker at this exact instant - wait
            # for the marker instead of racing further, but bounded the
            # same way, for the same reason.
            _wait_deadline = time.time() + STALE_LOCK_SECONDS
            while not os.path.exists(INSTALL_MARKER):
                if time.time() > _wait_deadline:
                    raise RuntimeError(
                        f"Gave up waiting for another worker's sglang install after {STALE_LOCK_SECONDS}s."
                    )
                time.sleep(3)
            return

    try:
        print("Installing sglang onto the persistent volume (first worker on this volume)...", flush=True)
        install_start = time.time()

        # Every subprocess.run below carries an explicit timeout=. None of
        # them had one originally - if any single network call inside (git
        # fetch, a pip download) genuinely hung rather than just being
        # slow, subprocess.run would block forever with no way out,
        # exactly like the lock-wait bug above. A hung install here would
        # never raise, never let handler() return, and never send RunPod
        # any signal that the job was done - pure wasted billing with no
        # way to notice short of manually checking the dashboard.
        subprocess.run(["python3", "-m", "venv", SGLANG_VENV_DIR], check=True, timeout=300)

        if os.path.exists(SGLANG_SRC_DIR):
            shutil.rmtree(SGLANG_SRC_DIR)
        subprocess.run(["git", "init", SGLANG_SRC_DIR], check=True, timeout=60)
        subprocess.run(["git", "remote", "add", "origin", "https://github.com/sgl-project/sglang.git"], check=True, cwd=SGLANG_SRC_DIR, timeout=60)
        subprocess.run(["git", "fetch", "--depth", "1", "origin", SGLANG_COMMIT], check=True, cwd=SGLANG_SRC_DIR, timeout=600)
        subprocess.run(["git", "checkout", "FETCH_HEAD"], check=True, cwd=SGLANG_SRC_DIR, timeout=120)

        env = os.environ.copy()
        # SGLang's pip build tries to discover and compile Rust extensions
        # from SGLANG_SRC_DIR/rust (crates: sglang-grpc, sglang-mm,
        # sglang-radix-tree, sglang-renderer, sglang-server) - needs a
        # cargo toolchain this image doesn't have. Checked what's actually
        # in there before disabling it, not just guessing: those crates
        # are disaggregated prefill/decode routing and gRPC serving infra
        # for LLM TEXT serving - the CLI entry point (sglang.cli.main:main)
        # and default HTTP server (sglang.srt.entrypoints.http_server) are
        # pure Python, and the gRPC modules are only conditionally
        # imported, never by the plain `sglang serve` diffusion path this
        # handler uses.
        env["SGLANG_BUILD_RUST_EXTS"] = "none"
        # Generous on purpose - a real run of this exact install showed
        # heavy pip dependency-resolver backtracking (flash-attn-4 alone
        # tried 13 candidate versions) plus downloading PyTorch and its
        # full CUDA dependency tree. Bounded, not unbounded, is what
        # matters here - 3600s is the same order of magnitude as this
        # file's own COLD_START_MAX_WAIT_SECONDS for model loading, not a
        # number expected to actually bind in the normal case.
        subprocess.run(
            [SGLANG_VENV_PYTHON, "-m", "pip", "install",
             f"{SGLANG_SRC_DIR}/python[diffusion]", "runpod", "requests"],
            env=env, check=True, timeout=3600,
        )

        elapsed = round(time.time() - install_start, 1)
        print(f"sglang installed onto volume in {elapsed}s.", flush=True)
        with open(INSTALL_MARKER, "w") as f:
            f.write("ok")
    finally:
        if got_lock:
            try:
                os.rmdir(INSTALL_LOCK_DIR)
            except OSError:
                pass


def _find_cuda_home():
    """The community-confirmed workaround for sgl-project/sglang#11333:
    point CUDA_HOME at the pip-installed nvidia-cuda-runtime package's own
    directory rather than relying on a system CUDA install. Belt-and-
    suspenders even with the upstream fix (PR #13089) merged - costs
    nothing if unneeded. Looks inside the VENV's own site-packages via
    glob rather than `import nvidia.cuda_runtime` directly - this handler
    process itself runs under the image's system python, which never has
    nvidia-cuda-runtime installed; only the venv on the volume does.
    Never raises: worst case sglang falls back to its own detection."""
    try:
        matches = glob.glob(os.path.join(SGLANG_VENV_DIR, "lib", "python3.*", "site-packages", "nvidia", "cuda_runtime"))
        return matches[0] if matches else None
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
    ensure_sglang_installed()

    if _is_ready():
        print("sglang already running.", flush=True)
        return None

    # Checked BEFORE start, same as handler.py's own start_kobold_if_needed -
    # reflects whether a real download is actually about to happen, not
    # whether one technically could (HF's cache dir already exists once
    # created below regardless).
    needs_download = not os.path.isdir(_MODEL_CACHE_MARKER)
    max_wait = COLD_START_MAX_WAIT_SECONDS if needs_download else WARM_RESTART_MAX_WAIT_SECONDS
    print(
        f"Model cache {'not found' if needs_download else 'found'} at "
        f"{_MODEL_CACHE_MARKER} - treating this as a "
        f"{'cold download' if needs_download else 'warm restart'}, "
        f"max wait {max_wait}s.",
        flush=True,
    )

    load_start = time.time()

    # Same flags as run_test.py - RTX 5090 tier from SGLang's own
    # MiniMax-H3 cookbook (lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3).
    cmd = [
        SGLANG_VENV_SGLANG, "serve",
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
    # Some console-script entry points shell back out to `python`/other
    # sibling scripts expecting themselves to be resolvable on PATH -
    # prepending the venv's own bin dir covers that, on top of invoking
    # SGLANG_VENV_SGLANG by its full path above.
    env["PATH"] = f"{SGLANG_VENV_BIN_DIR}:{env.get('PATH', '')}"
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

    while time.time() - load_start < max_wait:
        if _is_ready():
            elapsed = round(time.time() - load_start, 1)
            print(f"sglang ready after {elapsed}s (Load Time)", flush=True)
            return elapsed
        time.sleep(2)
    raise RuntimeError(f"sglang server did not become ready within {max_wait}s")


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
