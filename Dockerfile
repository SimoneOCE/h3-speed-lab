# SECURITY: pinned to a digest, not floating on :dev — same practice as
# minimax-h3-worker/Dockerfile uses for its koboldcpp base image. Floating
# on :dev meant every RunPod rebuild could silently pull a different (or
# simply unavailable) image with no record of what changed. This is the
# digest :dev resolved to as of this fix; refresh it deliberately (not as
# a side effect of an unrelated rebuild) by re-running:
#   TOKEN=$(curl -sS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:lmsysorg/sglang:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
#   curl -sS -D - -o /dev/null -H "Authorization: Bearer $TOKEN" \
#     -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
#     -H "Accept: application/vnd.oci.image.index.v1+json" \
#     "https://registry-1.docker.io/v2/lmsysorg/sglang/manifests/dev" | grep -i docker-content-digest
FROM lmsysorg/sglang@sha256:d9e4917808cfaa4b3be033a0c85a3a73d71eb93c17accbfa2ac3e96f551f3a33

WORKDIR /workspace

# Installed at build time now, not on every cold start via a "Container
# start command" git-clone hack — this is the whole point of switching to
# a real Dockerfile + RunPod's "Deploy from a GitHub repository" build.
RUN python -m pip install -e "/sgl-workspace/sglang/python[diffusion]" \
    && pip install runpod requests

COPY serverless_handler.py .

ENTRYPOINT []
CMD ["python3", "-u", "serverless_handler.py"]
