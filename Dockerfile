# Genuinely minimal, purpose-built image for MiniMax H3 diffusion serving -
# NOT SGLang's own maintained image (lmsysorg/sglang:dev is 13.80GB compressed,
# their own "-runtime" tag only gets to 12.64GB - confirmed via the registry
# directly, not a guess). That size is inherent to their general-purpose
# image (full CUDA devel toolkit, LLM-serving extras we don't need, docs,
# build tools). This Dockerfile builds only what MiniMax H3 diffusion
# serving actually needs, multi-stage, so the shipped image never carries
# the build-only tooling.
#
# SECURITY: both stages pinned to digests, not floating tags - same
# practice as minimax-h3-worker/Dockerfile and this repo's own prior
# version. Refresh deliberately with:
#   TOKEN=$(curl -sS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:nvidia/cuda:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
#   curl -sS -D - -o /dev/null -H "Authorization: Bearer $TOKEN" \
#     -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
#     "https://registry-1.docker.io/v2/nvidia/cuda/manifests/<tag>" | grep -i docker-content-digest
#
# Known risk this Dockerfile depends on being resolved (it is, confirmed):
# sgl_kernel had a regression (sgl-project/sglang#11333) hard-requiring the
# CUDA devel toolkit just to IMPORT, even without compiling anything -
# would have broken this exact "runtime-only final stage" approach. Fixed
# in sgl-project/sglang#13089, merged into main 2025-12-02 - the check is
# now a graceful fallback, not a crash. serverless_handler.py also sets
# CUDA_HOME to the pip-installed nvidia-cuda-runtime package's own
# directory at process start (the community-confirmed workaround from that
# same issue thread), belt-and-suspenders even with the fix merged.

# ---- Builder stage: has the CUDA devel toolkit, only to compile SGLang's
# extensions during install. Never shipped. ----
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04@sha256:c2621d98e7de80c2aec5eb8403b19c67454c8f5b0c929e8588fd3563c9b6558d AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang

# --prefix collects everything (SGLang, its diffusion extras, PyTorch, and
# the runpod SDK) into one clean directory tree the final stage can copy
# wholesale, instead of guessing which files matter across a full OS install.
RUN pip install --prefix=/install --break-system-packages \
    "/sgl-workspace/sglang/python[diffusion]" runpod requests

# ---- Final stage: runtime-only CUDA image, no compiler, no devel headers. ----
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04@sha256:f2c12914cf4751e61073843724275aa35d4817e01dbdc87eac03905971628c6e

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /sgl-workspace/sglang /sgl-workspace/sglang

WORKDIR /workspace
COPY serverless_handler.py .

ENTRYPOINT []
CMD ["python3", "-u", "serverless_handler.py"]
