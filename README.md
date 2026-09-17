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
quality -s fast a b c                # ONE benchmark — quickest way to rank a few models
quality -s tools model-a             # tool calling (BFCL v3)
quality -s pl my-model --limit 100   # local judged set from the suites dir
quality -s smoke my-model            # ~1 min wiring check
quality -s ifeval my-model           # ONE benchmark — no suite needed
quality -s ifeval,gsm8k my-model     # an ad-hoc combination
quality --route proxy my-model       # evaluate the model AS SERVED (filters apply)
quality --route proxy my-model:x     # ... which is the only way to evaluate a TIER
quality --list                       # suites + resolved configuration
quality --list-datasets              # every benchmark id evalscope has registered
quality --dry-run -s tools my-model  # print the evalscope command, run nothing
quality -v my-model                  # stream evalscope's raw output instead of scores
```

Output is a **result, not a log**. evalscope's INFO stream, its config dump, the nested tqdm
bars and its perf tables are filtered out; what is left is a progress line, anything that
went wrong, and a score table read back from the report JSON — pivoted model-per-column when
more than one model was named, which is the table an A/B is actually run for. A failure prints
the tail of what evalscope said plus the path to the full log. `-v` turns the filter off.

Model ids are checked against the server's `/v1/models` **before** anything is loaded. Without
that check a typo costs five retries and a full `openai.NotFoundError` traceback *per sample*,
which is a page of Python for what is really a one-line mistake; a suite name typed where a
model belongs (`quality quality my-model`) is called out by name. `--skip-model-check` is there
for a server whose served ids differ from its routes.

`-s` takes either a suite name or benchmark ids, comma-separated: the suites below are
shorthands for combinations worth re-running, not the set of things that can be run. Any of
the 250-odd ids from `--list-datasets` works, and per-dataset wiring (BFCL subsets, the local
`general_qa` set and its judge) follows the dataset either way. An ad-hoc run writes to its own
`outputs/<ids>/` directory, so it never lands in a suite's report tree; a mistyped id is caught
against the registry *before* a model is loaded.

| suite | datasets | notes |
|---|---|---|
| `quality` (default) | ifeval, gsm8k, humaneval | objectively graded, no judge |
| `fast` | humaneval | quick ranking: fixed n=164, no judge, code only |
| `tools` | bfcl_v3 | 9 curated subsets incl. both irrelevance sets |
| `pl` | general_qa | local `suites/pl_starter.jsonl`, LLM-judged |
| `reasoning` | gsm8k | quick, no code |
| `smoke` | gsm8k (limit 5) | wiring check |

**Why `fast` is humaneval and not one of the other two.** It is the one that *moves*: gsm8k is
saturated for any competent mid-size model (96-98 %, no headroom to rank with) and ifeval has
come back statistically identical between builds that differ measurably elsewhere. humaneval
also has a **fixed n of 164**, so there is no `--limit` to choose, defend, or forget to keep
equal across arms, and it is graded by running the tests rather than by a judge model. The
trade is that it measures Python generation in English and nothing else — `tools` is the one
for the agent role, and a result worth publishing still comes from the full `quality` suite.

### Configuration

| variable | default | meaning |
|---|---|---|
| `EVALSCOPE_API_BASE` | *(required)* | base URL of the server under test |
| `EVALSCOPE_API_KEY` | `EMPTY` | bearer token |
| `EVALSCOPE_API_PATH` | *(unset)* | request path; `{model}` is substituted. Overrides `--route`; for servers that are not llama-swap |
| `QUALITY_ROUTE` | `upstream` | default for `--route`: `upstream` \| `proxy` |
| `QUALITY_SUITE` | `quality` | default suite |
| `QUALITY_LIMIT` | `250` | default `--limit` (**per subset**) |
| `QUALITY_BATCH` | `2` | concurrent requests (fallback for models not named in a `--batch-size` map) |
| `QUALITY_JUDGE` | *(unset)* | model id used as judge by judged suites |
| `QUALITY_OUTPUT` | `/data/outputs` | run output root |
| `QUALITY_SUITES_DIR` | `/work/suites` | custom dataset dir |

Mount a volume at `/data`: the first dataset pull is slow and belongs in a cache.

### `--route`: which thing are you measuring?

Two routes reach the same model behind llama-swap, and they answer different questions.

| `--route` | path | sampling | answers |
|---|---|---|---|
| `upstream` *(default)* | `/upstream/<model>/v1/chat/completions` | pinned by this tool | "is this **model** correct?" — for comparing builds, quants, engines |
| `proxy` | `/v1/chat/completions` | **the entry's filters apply** | "is my **production entry** correct?" — what callers actually get |

`upstream` is the default because it bypasses server-side request rewriting, so scores are
reproducible between runs. `proxy` deliberately gives that up, and prints a warning saying so.

**`proxy` is the only way to evaluate a `setParamsByID` tier**, because a tier *is* a filter — on
the upstream route it does not exist. Measured against a llama-swap serving thinking tiers,
reasoning tokens per sample on gsm8k:

```
route=proxy     qwen38-flash    [0, 0, 0, 0, 0]            filters force thinking off
route=proxy     qwen38-flash:x  [140, 155, 70, 116, 314]   the tier applies
route=upstream  qwen38-flash:x  server default only — bare and :x are indistinguishable
```

So an `upstream` run against an entry whose filters disable thinking is measuring a *thinking*
model, which is not what that entry serves. Pick the route that matches the question.

For a server that is not llama-swap at all, set `EVALSCOPE_API_PATH` (e.g. `/v1/chat/completions`);
it overrides `--route`, and setting both is an error rather than a silent precedence rule.

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
- **A degraded judge is reported, not averaged in.** Same failure mode, one level up: when the judge
  model cannot produce a usable verdict, evalscope records it per sample and then reports the
  aggregate as `0.0`, which reads as "the model is bad". A run whose judge failed on any sample now
  prints `! JUDGE DEGRADED: only N/M samples got a usable verdict` above the table. Seen for real
  with a 0.8B model as judge: every verdict came back `parse_error`, coverage 0.0, and the table
  said `0.0000` for answers that were correct.
- **A benchmark that evalscope will only score with a judge is caught before the run.** 38 of the
  registered benchmarks declare `JUDGE_ONLY` or `JUDGE_DEFAULT` (`needle_haystack`, `simple_qa`,
  `arena_hard`, `hle`, …). Without a judge, evalscope raises *after* pulling the datasets; the
  wrapper asks the registry up front and fails with the same early, actionable error every other
  mistake here gets.
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
- **`needle_haystack` needs a judge, and a *capable* one.** Its rule path is not an escape hatch:
  `--judge-strategy rule` (not reachable through this wrapper, but reachable by calling `evalscope`
  directly) scores by `exact_match` against the whole needle sentence, so a model answering
  `Eat a sandwich and sit in Dolores Park on a sunny day.` is scored **0** against a reference of
  `The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.`
  — a perfect retrieval reported as 0 %. With a competent judge the same run scores 1.0.
  It also defaults to `english` only here (the `chinese` corpus reruns the whole context x depth
  grid; add it back explicitly if that is what is being measured), and its grid is set through
  `extra_params`, e.g.
  `--dataset-args '{"needle_haystack": {"extra_params": {"context_lengths_max": 128000}}}'`.
  Two cosmetic evalscope complaints on this benchmark are harmless: `undeclared metric` and
  `Error generating charts: Columns must be same length as key` (the heatmap needs a full grid).
- **The `needle-haystack` extra is required to even construct it.** `matplotlib` and `seaborn` are
  only used for the heatmap, but the adapter's `__init__` calls `check_import` on them, so without
  the extra the benchmark dies before a single sample runs. It is installed and build-canaried here.
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
