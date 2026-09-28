# SECURITY: pinned to a digest, not floating on :dev — same practice as
# minimax-h3-worker/Dockerfile uses for its koboldcpp base image. See that
# file's own comment for the refresh command (same pattern, swap the repo).
FROM lmsysorg/sglang@sha256:d9e4917808cfaa4b3be033a0c85a3a73d71eb93c17accbfa2ac3e96f551f3a33

WORKDIR /workspace

# Deliberately lean, matching minimax-h3-worker/Dockerfile's own pattern:
# only the small, fast, pure-Python deps go here. SGLang's own diffusion
# extras are NOT installed here — that's a heavy, real dependency tree
# (diffusers, nvidia-nccl-cu13, etc.), and baking it into the image means
# every worker on a fresh physical host has to re-pull the whole bloated
# image before it can even start (confirmed via RunPod's own docs: they
# explicitly recommend caching over baking large things into images).
# handler.py's ensure_koboldcpp_engine() never bakes the koboldcpp engine
# or model weights into ITS image either, for the exact same reason -
# instead it installs once onto the persistent network volume, shared
# across every worker regardless of which physical host runs it. See
# serverless_handler.py's ensure_sglang_diffusion_installed() for the
# equivalent here.
RUN pip install runpod requests

COPY serverless_handler.py .

ENTRYPOINT []
CMD ["python3", "-u", "serverless_handler.py"]
