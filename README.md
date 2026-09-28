# h3-speed-lab

SGLang speed test for MiniMax H3 — the third leg of the same investigation
as the koboldcpp production baseline and the ComfyUI+SageAttention pod
test. Manual, phase-by-phase, run by hand on a RunPod pod, same workflow
as the ComfyUI test: start the server, watch how long it takes to be
ready, run a throwaway gen yourself, then a real one.

## What carries over from the ComfyUI test

- **Same production baseline settings**: 1280x736 (16:9), ~175 frames /
  7.29s duration, 20 steps — so the real `generate` run is directly
  comparable to both the koboldcpp number (~21.6 min) and the
  SageAttention ComfyUI number (336.92s).
- **Same GPU-stats logging pattern** as `minimax-h3-worker/handler.py`'s
  `sample_gpu_stats` — periodic `nvidia-smi` snapshots
  `[unix_timestamp, gpu_util_pct, vram_used_mb]`, recorded to
  `session_log.json` for every generation.
- **Same warmup-first workflow**: low-res/1-step throwaway gen before the
  real timed one.
- **Same phase terminology** we settled on this session — Load Time,
  Warmup Lag, Warmup Duration, Dispatch Lag — used in the log labels
  (`load_ready` = Load Time, `*_dispatch_start` = start of Dispatch Lag).

## What's NOT carried over, on purpose

- **No SageAttention patch** — SGLang has its own built-in optimized
  attention kernels; there's no separate "SageAttention for SGLang" node
  or equivalent to add. SGLang's whole pitch *is* the optimized path here.
- **No GGUF hybrid** — ruled out earlier for output-quality risk.

## What I need from you on the RunPod side

1. **A fresh pod**, RTX 5090 (same tier as the ComfyUI test, for a fair
   comparison) — new pod is fine, the terminated one's local state
   (models, installs) is gone either way and needs redoing regardless.
2. **Docker template**: `lmsysorg/sglang:dev` (SGLang's own official
   image, per their MiniMax-H3 cookbook — this is NOT the ComfyUI
   template from before, it's a different image).
3. **A Hugging Face token** (`HF_TOKEN` env var) if `MiniMaxAI/MiniMax-H3`
   is gated — check on the model's HF page; set it as an environment
   variable on the pod if so.
4. Enough disk on the pod's volume for the model download (same
   ballpark as before, tens of GB — exact size TBD once we see what
   the model repo actually contains).

## Setup steps (run these on the pod's terminal)

```bash
# 1. Install SGLang's diffusion extras (per their own cookbook - assumes
#    the sglang source is already present at this path in the lmsysorg/sglang:dev image)
python -m pip install -e "/sgl-workspace/sglang/python[diffusion]"

# 2. Clone this repo onto the pod
git clone https://github.com/SimoneOCE/h3-speed-lab.git
cd h3-speed-lab
pip install requests

# 3. (If MiniMax-H3 is gated) set your HF token
export HF_TOKEN=your_token_here
```

## Running the test

```bash
# Phase 1: launch the server, wait for ready — this measures Load Time
python run_test.py start

# Before running a real timed test, it's worth checking SGLang's actual
# API contract rather than trusting the cookbook's example blindly —
# open http://<pod-ip>:30010/docs (or curl .../openapi.json) if SGLang
# exposes it. This script's response-parsing is a best guess pending
# real evidence — see the big comment at the top of run_test.py.

# Phase 2: your throwaway/warmup gen (low-res, 1 step, deliberately tiny)
python run_test.py warmup

# Phase 3: the real, full-settings timed generation
python run_test.py generate --prompt "your prompt here"

# Check status / recent log entries at any point
python run_test.py status
```

Each run appends to `session_log.json` in this folder (gitignored — it's
per-session data, not code) and saves the resulting `.mp4` to `outputs/`
(also gitignored). Pull `session_log.json` and the `.mp4` off the pod
however you'd normally grab files (the pod's file browser, `scp`, etc.)
when you want to hand results back for analysis.

## Real Serverless deployment (`serverless_handler.py`)

`run_test.py` above is a manual pod-driven test tool. `serverless_handler.py`
is the actual thing — a proper RunPod Serverless handler, same pattern as
`minimax-h3-worker/handler.py` uses in production for koboldcpp: it starts
the SGLang server once when the worker cold-starts, then calls
`runpod.serverless.start()`, RunPod's own worker loop, which pulls jobs
from the queue and calls `handler()` for each one. Not something you SSH
into and drive by hand.

**Setup**: this repo now has a real `Dockerfile`, same style as
`minimax-h3-worker/Dockerfile` — digest-pinned base image, everything
(SGLang's diffusion extras, the `runpod` SDK, the handler itself) installed
at **build time**, not re-installed on every cold start via a
`git clone`-in-start-command hack. Deploy it with RunPod's
**"Deploy from a GitHub repository"** option (not "Deploy from a Docker
image" — that's the old path), pointed at this repo
(`SimoneOCE/h3-speed-lab`). RunPod builds the image from the Dockerfile
itself; no separate CI/CD config needed, same as production. Leave the
**Container start command** field empty — the Dockerfile's own `CMD`
handles that now.

Note: switching an existing endpoint from a Docker-image source to a
GitHub-repo source isn't necessarily an in-place edit in RunPod's UI —
may need a fresh endpoint. Same network volume (for the model cache) and
GPU/region settings carry over regardless of which endpoint you use.

**Sending a job**: use the endpoint's own **Requests** tab in the RunPod
dashboard (not SSH, not curl) and paste:
```json
{"input": {"prompt": "your prompt here"}}
```
All fields are optional and default to the production baseline settings
(`duration_seconds: 7.29`, `short_edge: 736`, `aspect_ratio: "16:9"`,
`num_inference_steps: 20`) for direct comparability with the koboldcpp and
ComfyUI numbers.

**Response** is self-contained — no need to tail logs separately:
- `cold_start_load_time_seconds` — Load Time, only present on the job that
  triggered the cold start (SGLang server startup).
- `dispatch_lag_seconds` — time from request dispatched to SGLang's
  `/v1/videos` responding.
- `total_generation_seconds` — total handler time for this job.
- `gpu_util_samples` — same `[unix_ts, gpu_util_pct, vram_used_mb]` shape
  as `handler.py`'s own `sample_gpu_stats`.
- `video_base64` — the generated video, base64-encoded, if SGLang's
  response shape allowed extracting it. To save it as a real `.mp4`:
  ```bash
  python3 -c "import base64,sys; open('out.mp4','wb').write(base64.b64decode(sys.stdin.read()))" <<< "PASTE_THE_BASE64_HERE"
  ```
- If extraction failed, you'll get `raw_response` + `note` instead —
  paste that back so `_extract_video_b64`/`_poll_job` can be fixed against
  the real response shape.

## Known gap — read before the first real run

SGLang's own MiniMax-H3 cookbook documents the server launch flags and
most request fields clearly, but does **not** document:

- The exact health-check endpoint (this script guesses at `/health`,
  `/v1/models`, `/get_model_info` and accepts whichever responds first).
- Whether `/v1/videos` responds synchronously with the video, or
  asynchronously with a job id to poll.
- The exact field name the final video comes back under (URL? base64?).

Rather than silently guess and risk a wasted timed run, `warmup` and
`generate` both print the **raw JSON response** from the server the first
time. If the script's parsing doesn't find a video, paste that raw
response back and it's a quick fix to `_extract_video`/`_poll_job` in
`run_test.py` — no need to re-derive the whole script.
