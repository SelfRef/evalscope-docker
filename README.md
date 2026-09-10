# evalscope-docker

A small [EvalScope](https://github.com/modelscope/evalscope) image for evaluating models that are
**already being served** over an OpenAI-compatible API. It answers **"is the answer right"** — it
is not a speed benchmark and does not try to be.

```
docker compose run --rm evalscope quality my-model
```

The container never touches a GPU: it is a pure HTTP client of whatever serves your models.

## Why not the official image

EvalScope's "official" image is the ModelScope monolith — tags like
`ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.3` — which merely
*includes* the evalscope library. For this purpose it is wrong on every axis: it ships CUDA, vLLM
and ms-swift when inference happens elsewhere, it never needs the GPU it is built around, and it
lives on a region-specific registry that is slow to pull from much of the world.

This image is `python:3.12-slim` plus the four extras actually used, and is roughly an order of
magnitude smaller. `torch` is pinned to the **CPU** wheel: the `bfcl` extra depends on torch and
the default PyPI resolve drags ~5 GB of CUDA libraries in (7.2 GB → 2.4 GB; ~3.4 GB once the
`rag`/MTEB extra is included).

## Why it is separate from the inference image

EvalScope's graders pull torch and a long dependency tail. Keeping them out of the serving image
means an eval dependency can never break inference, and this image can be rebuilt on its own
schedule.

## `quality` — the wrapper

A run is *just a model id*.

```
quality my-model                     # default suite, default limit
quality model-a model-b              # A/B — evaluated one at a time, never concurrently
quality -s tools model-a             # tool calling (BFCL v3)
quality -s pl my-model --limit 100   # local judged set from the suites dir
quality -s smoke my-model            # ~1 min wiring check
quality -s ifeval my-model           # ONE benchmark — no suite needed
quality -s ifeval,gsm8k my-model     # an ad-hoc combination
quality --list                       # suites + resolved configuration
quality --list-datasets              # every benchmark id evalscope has registered
quality --dry-run -s tools my-model  # print the evalscope command, run nothing
```

`-s` takes either a suite name or benchmark ids, comma-separated: the suites below are
shorthands for combinations worth re-running, not the set of things that can be run. Any of
the 250-odd ids from `--list-datasets` works, and per-dataset wiring (BFCL subsets, the local
`general_qa` set and its judge) follows the dataset either way. An ad-hoc run writes to its own
`outputs/<ids>/` directory, so it never lands in a suite's report tree; a mistyped id is caught
against the registry *before* a model is loaded.

| suite | datasets | notes |
|---|---|---|
| `quality` (default) | ifeval, gsm8k, humaneval | objectively graded, no judge |
| `tools` | bfcl_v3 | 9 curated subsets incl. both irrelevance sets |
| `pl` | general_qa | local `suites/pl_starter.jsonl`, LLM-judged |
| `reasoning` | gsm8k | quick, no code |
| `smoke` | gsm8k (limit 5) | wiring check |

### Configuration

| variable | default | meaning |
|---|---|---|
| `EVALSCOPE_API_BASE` | *(required)* | base URL of the server under test |
| `EVALSCOPE_API_KEY` | `EMPTY` | bearer token |
| `EVALSCOPE_API_PATH` | `/upstream/{model}/v1/chat/completions` | request path; `{model}` is substituted |
| `QUALITY_SUITE` | `quality` | default suite |
| `QUALITY_LIMIT` | `250` | default `--limit` (**per subset**) |
| `QUALITY_BATCH` | `2` | concurrent requests |
| `QUALITY_JUDGE` | *(unset)* | model id used as judge by judged suites |
| `QUALITY_OUTPUT` | `/data/outputs` | run output root |
| `QUALITY_SUITES_DIR` | `/work/suites` | custom dataset dir |

Mount a volume at `/data`: the first dataset pull is slow and belongs in a cache.

`EVALSCOPE_API_PATH` defaults to [llama-swap](https://github.com/mostlygeek/llama-swap)'s
per-model `/upstream/<model>/` route, which bypasses server-side request rewriting so the eval
controls sampling. For a plain single-model server, set it to `/v1/chat/completions`.

### Decisions baked into the wrapper

These are the things that are easy to get wrong and expensive to notice:

- **Sampling is pinned** (temperature 0, top_p 1, seed 42). A quant or build A/B must not be read
  through sampling noise — and if the endpoint you point at can rewrite sampling parameters
  server-side, scores stop being reproducible between runs.
- **One model at a time.** Where a server swaps models in and out of limited VRAM, two concurrent
  runs thrash the weights on every request.
- **`--eval-batch-size` should match the server's parallel-slot count.** Unlike speed
  benchmarking, concurrency here is *free*: correctness does not care that requests share a
  server, so there is no reason to serialise.
- **Per-sample errors are NOT ignored.** `--ignore-errors` is opt-in. A broken grader dependency
  presents as a run that "succeeds" and scores every sample zero — not as a failure — so ignoring
  errors by default converts a crash into a plausible-looking wrong number.
- **A judged suite requires a different model as judge** (`QUALITY_JUDGE`); judging with the model
  under test is self-preference bias. If the server swaps models within one GPU, prefer a judge
  that is not in the same swapping group, or every judged sample costs a model reload.

There is **no exclusivity guard**: another client using the same server mid-run costs wall-clock,
not correctness. Long runs still belong on an idle host.

## Re-scoring without re-running inference

If a *grader* was broken (not the model), re-score the cached predictions:

```
quality my-model --use-cache /data/outputs/quality/<timestamp> --rerun-review
```

Predictions are reused; only scoring re-runs. It rewrites the reviews **in place** in that
directory rather than creating a new one.

## Gotchas found the hard way

- Benchmarks are registered as **`bfcl_v3` / `bfcl_v4`**, not `bfcl`.
- **`--limit` is per subset**, so a 17-subset benchmark at `--limit 25` is 425 samples.
- **No stock benchmark here covers every language.** Checked against 1.11.1: `mmmlu` ships 14
  languages, `multi_if` 11, `poly_math` 18, `mgsm` 11 — plenty of widely-spoken languages appear
  in none of them. MTEB (via the `rag` extra) has far broader language coverage, but only for
  embedding/retrieval tasks. For generation in an uncovered language you need your own set; see
  `suites/`.
- MTEB (`suites/embed_mteb.json`, RAGEval backend): the config key is **`models`** (plural);
  `limits` truncates a split **without stratifying**, which can leave a classification probe with
  a single class; multilingual tasks fail with a `KeyError` on foreign subsets when filtered by
  language; `eval.output_folder` is not honoured — read scores from the run log.
- `docker run` without `-i` gives no stdin, so `python - <<EOF` silently does nothing. Mount a
  script or put the heredoc *inside* `sh -c`.

## Build

```
docker build -t ghcr.io/selfref/evalscope-docker:latest .
```

CI rebuilds weekly (Saturday 21:00 UTC) and on any change to `Dockerfile`, `scripts/` or
`suites/`. The Dockerfile carries **build canaries**: the BFCL grader and the NLTK resource must
import, or the build fails. Both are only reached at *scoring* time, so without them a broken
dependency chain would ship as an image that runs happily and scores everything zero.
