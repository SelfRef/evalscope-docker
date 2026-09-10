# EvalScope — model *correctness* evaluation against an OpenAI-compatible
# endpoint.
#
# WHY NOT THE OFFICIAL IMAGE. EvalScope's "official" image is the ModelScope
# monolith (tags like ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-
# modelscope1.31.0-swift3.9.3) which merely *includes* the evalscope library.
# It ships CUDA, vLLM and ms-swift, none of which this image needs: inference
# happens on the server under test, and this container never touches a GPU at
# all — it is a pure HTTP client. It is also hosted on a region-specific
# registry that is slow to pull from much of the world. This image is built
# from python:3.12-slim instead and is roughly an order of magnitude smaller.
#
# WHY IT IS SEPARATE FROM THE INFERENCE IMAGE. EvalScope's graders pull torch
# and a long dependency tail. Keeping them out of the serving image means an
# eval dependency can never break inference, and this image can be rebuilt on
# its own schedule.
#
# WHAT IT IS NOT FOR. Speed. EvalScope's own `perf` module is deliberately
# unused: it cannot see VRAM pressure, speculative-decoding accept rates or
# output determinism, which is what a serving-side benchmark is for.
FROM python:3.12-slim

ARG EVALSCOPE_VERSION=1.11.1

LABEL org.opencontainers.image.title="evalscope-docker" \
      org.opencontainers.image.description="EvalScope correctness harness for OpenAI-compatible endpoints (CPU-only, no GPU)" \
      org.opencontainers.image.source="https://github.com/SelfRef/evalscope-docker"

# libsndfile: bfcl_eval imports qwen_agent, which imports soundfile at module
# scope, which needs the C library. Without it the ENTIRE bfcl_eval import
# chain dies and every BFCL sample fails to score with a misleading
# "cannot import name 'ast_checker'" — which `--ignore-errors` will happily
# swallow, producing a run that looks like it worked and scored nothing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# torch FIRST, from the CPU-only index. The `bfcl` extra depends on torch, and
# the default PyPI wheel drags ~5 GB of CUDA libraries into a container that
# never touches a GPU at all (it is an HTTP client). Installing the CPU wheel
# up front makes the later resolve a no-op.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Extras, picked one by one rather than [all] — [all] adds vlmeval/opencompass
# and the AIGC stacks, none of which is used here. ifeval: langdetect +
# immutabledict for the instruction-following checkers. bfcl: the
# function-calling AST checker. multi-if: multi-turn instruction following.
# rag: MTEB + langchain_openai, for evaluating embedding / reranker models
# through /v1/embeddings and /v1/rerank — evalscope's RAGEval backend ships an
# APIEncoder, so the SERVED endpoint is evaluated rather than a separate local
# copy of the weights.
#
# The last two lines are BUILD CANARIES. Both graders are only imported at
# SCORING time, so without them a broken dependency chain ships as an image
# that runs happily and scores every sample zero.
RUN pip install --no-cache-dir \
      "evalscope[ifeval,bfcl,multi-if,rag]==${EVALSCOPE_VERSION}" \
      soundfile \
    && python -c "import evalscope, langdetect; print(evalscope.__version__)" \
    && python -c "import torch; assert '+cpu' in torch.__version__, torch.__version__; print(torch.__version__)" \
    && python -c "from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker; print('bfcl ast_checker OK')"

# NLTK data, baked in. IFEval's sentence-level instruction checkers tokenise
# with punkt; without the resource those samples score 0 and the run reports a
# depressed accuracy with a single ERROR line as the only clue. Downloading at
# run time would also mean a network call per container.
ENV NLTK_DATA=/usr/local/nltk_data
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt punkt_tab \
    && python -c "import nltk; nltk.data.find('tokenizers/punkt_tab/english/'); print('nltk punkt_tab OK')"

# The wrapper and the example suites are baked IN so a run needs no bind
# mounts. Mount over /work/suites to iterate on a suite without a rebuild.
COPY scripts/quality /usr/local/bin/quality
COPY suites/ /work/suites/
RUN chmod +x /usr/local/bin/quality && quality --list >/dev/null

# Datasets are fetched from the ModelScope hub — cached under /data so a
# re-run is offline and the slow first pull happens once. Mount a volume there.
ENV MODELSCOPE_CACHE=/data/modelscope \
    HF_HOME=/data/hf \
    QUALITY_OUTPUT=/data/outputs \
    QUALITY_SUITES_DIR=/work/suites \
    PYTHONUNBUFFERED=1

WORKDIR /work
# No long-running process: this is a run-on-demand tool. `docker compose run`
# overrides this; the sleep only keeps a `compose up` container alive for exec.
CMD ["sleep", "infinity"]
