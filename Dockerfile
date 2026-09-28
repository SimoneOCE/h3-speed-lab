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
    python3 python3-pip python3-dev build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinned to a specific commit (not `--depth 1` off main, which would make
# every rebuild silently pick up whatever's newest) - same reproducibility
# discipline as the digest-pinned base images above. Refresh deliberately
# by checking https://github.com/sgl-project/sglang/commits/main for a
# new SHA, not as a side effect of an unrelated rebuild.
RUN git init /sgl-workspace/sglang \
    && cd /sgl-workspace/sglang \
    && git remote add origin https://github.com/sgl-project/sglang.git \
    && git fetch --depth 1 origin 2e7e0802f4f3edc94702933675a71d6c7ef4f24f \
    && git checkout FETCH_HEAD

# SGLang's pip build tries to discover and compile Rust extensions from
# /sgl-workspace/sglang/rust (crates: sglang-grpc, sglang-mm,
# sglang-radix-tree, sglang-renderer, sglang-server) - this needs a cargo
# toolchain, which this image intentionally doesn't have. Checked what's
# actually in there before disabling it, not just guessing: those crates
# are disaggregated prefill/decode routing and gRPC serving infra for LLM
# TEXT serving - the CLI entry point (sglang.cli.main:main) and the
# default HTTP server (sglang.srt.entrypoints.http_server) are pure
# Python, and the gRPC modules are only conditionally imported, never by
# the plain `sglang serve` diffusion path this handler uses. The actual
# diffusion-relevant native deps - st_attn/vsa (spatiotemporal/video
# sparse attention) - are separate pip packages built through their own
# torch-based setuptools path, unrelated to this Cargo workspace. Safe to
# skip per the error message's own documented escape hatch.
ENV SGLANG_BUILD_RUST_EXTS=none

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
# Deliberately NOT copying /sgl-workspace/sglang's source tree here - the
# pip install above is non-editable, so the actual importable code already
# lives in /usr/local's site-packages. If something breaks at runtime with
# a missing-file error pointing at a path under /sgl-workspace, that's the
# signal this assumption was wrong and the source tree needs adding back.

WORKDIR /workspace
COPY serverless_handler.py .

ENTRYPOINT []
CMD ["python3", "-u", "serverless_handler.py"]
