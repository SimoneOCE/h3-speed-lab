# Minimal image, matching minimax-h3-worker/Dockerfile's own shape almost
# exactly - just enough to run serverless_handler.py and call
# runpod.serverless.start() immediately. SGLang itself is installed at
# RUNTIME, onto the persistent volume, by ensure_sglang_installed() in
# serverless_handler.py - not baked in here at all.
#
# Seventh revision, and a deliberate reversal of the "genuinely custom
# minimal image" approach (see git history): that shrunk the SHIPPED image
# from SGLang's official 13.8GB down to 6.1GB, which helped, but a real
# deploy then showed one single 4.11GB layer taking ~11.5 minutes to
# download on a fresh RunPod host (~6MB/s effective, vs. the ~250MB/s this
# same image pushed at). Shrinking the image never actually fixed the real
# problem: RunPod's Docker layer cache is PER PHYSICAL HOST, not
# fleet-wide, so every worker landing on a host that's never pulled this
# image before pays the full pull cost again, no matter how small the
# image is. The persistent network volume has the opposite property - the
# same volume follows the worker wherever it lands, it isn't tied to one
# host's local disk - so installing SGLang there instead means the first
# worker on a given volume ever pays the install cost ONCE, and every
# worker after that (on ANY host, not just a host that happens to have
# cached this image) finds it already there. Same principle as this repo's
# own HF_HOME model-weight caching, and the same principle handler.py's
# ensure_koboldcpp_engine()/VOLUME_DIR pattern already uses in production -
# this just extends it to SGLang itself instead of only the model weights.
#
# SECURITY: pinned to a digest, not a floating tag - refresh deliberately
# with:
#   TOKEN=$(curl -sS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:nvidia/cuda:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
#   curl -sS -D - -o /dev/null -H "Authorization: Bearer $TOKEN" \
#     -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
#     "https://registry-1.docker.io/v2/nvidia/cuda/manifests/<tag>" | grep -i docker-content-digest
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04@sha256:f2c12914cf4751e61073843724275aa35d4817e01dbdc87eac03905971628c6e

# build-essential/python3-dev kept despite this being a "runtime" image -
# one of the packages sglang's own dependency tree pulls in (cuda-tile)
# builds its wheel via pyproject.toml at install time, and the only real
# build log we have of this succeeding (the old Docker builder stage) had
# a compiler present throughout. Never actually confirmed it would still
# succeed without one, and this isn't the moment to find out the hard way
# on a live volume install - these add under 200MB to the image, nothing
# close to the multi-GB problem this whole revision is fixing.
# python3-venv is a separate Debian package from python3 itself - bare
# python3 doesn't include the venv module.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential git \
    ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY serverless_handler.py .

# Just enough for this file's own module-level imports and for
# runpod.serverless.start() to run immediately - matches
# minimax-h3-worker/Dockerfile's own "pip install runpod requests boto3
# psutil" line exactly in spirit. SGLang itself is never installed here;
# see ensure_sglang_installed() in serverless_handler.py.
RUN pip install --break-system-packages runpod requests

ENTRYPOINT []
CMD ["python3", "-u", "serverless_handler.py"]
