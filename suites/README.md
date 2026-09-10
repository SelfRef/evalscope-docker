# Custom eval suites

EvalScope ships ~200 benchmarks, but **no public benchmark covers your traffic**, and for many
languages it ships no generation benchmark at all (checked against evalscope 1.11.1: `mmmlu` has
14 languages, `multi_if` 11, `poly_math` 18, `mgsm` 11 — many widely-spoken languages appear in
none of them). MTEB, reachable through the `rag` extra, has much broader language coverage but
only for embedding/retrieval tasks.

This directory is where you close that gap with your own prompts.

## `pl_starter.jsonl`

An **example** set, not a benchmark: ~24 short Polish items shaped like ordinary assistant work
(extraction to JSON, constrained formatting, inflection, translation, summarising to a length
limit), each with a short unambiguous reference answer so it can be graded either by Rouge-L or by
a judge model.

It is deliberately small and deliberately easy to replace. **Treat a score from it as a smoke
test**: at n=24 the standard error is ~10 pp, which cannot separate two decent models.

To make it decision-grade, replace the items with prompts drawn from your own traffic. The format
is one JSON object per line:

```json
{"question": "...", "answer": "...", "system": "optional system prompt"}
```

Run it with:

```
docker compose run --rm evalscope quality -s pl <model> --limit 100
```

The suite is LLM-judged, so `QUALITY_JUDGE` must be set to a model **other** than the one under
test — judging with the model under test is self-preference bias. If your server swaps models
within one GPU, pick a judge that is not in the same swapping group, or every judged sample costs
a model reload.

## `embed_mteb.json`

An MTEB run against a served embedding endpoint (evalscope's RAGEval backend, `APIEncoder`), so
the *served* model is measured rather than a separate local copy of the weights. Edit
`eval.task_names` for the tasks and languages you care about.

Config gotchas that each cost a run:

- the key is **`models`** (plural), not `model`;
- **`limits` truncates a split without stratifying** — it can leave a classification probe with a
  single class, which dies in sklearn ("needs samples of at least 2 classes");
- multilingual tasks fail with a `KeyError` on a foreign subset when filtered by `languages`; use
  the single-language task variants;
- `eval.output_folder` is not honoured — read the score table from the run log.

It is invoked through evalscope's Python API rather than the `quality` wrapper, because the
RAGEval backend takes a whole task config rather than a model name.
