# Dummy language-context RAG experiment

## Question

Can a language-specific medical-term database provide enough surrounding evidence to improve the
retrieval and ranking of a corrupted controlled term, without rewriting source text or committing
a correction automatically?

## Design

This is a paired, offline synthetic test. Both arms receive the exact same candidate span and
language. The baseline retrieves same-language controlled terms only when normalized character
similarity is at least 0.85. The context arm uses the same baseline and may also retrieve a term
when surface similarity is at least 0.25 and **both** of that record's two language-specific
context cues occur outside the candidate span. It then ranks with a fixed score of 65% surface
similarity and 35% context evidence.

The fixture contains 12 positive and five negative cases across English (`en`), Mandarin Chinese
(`zh`), Japanese (`ja`), Korean (`ko`), and Hindi (`hi`). Terms, errors, and contexts are synthetic;
the records' provenance is `synthetic-language-context-v1`. No patient data or proprietary
terminology is present.

Metrics are:

- **Correction sensitivity@5:** positive cases whose gold concept appears in the first five.
- **Selection accuracy@1:** positive cases whose gold concept ranks first.
- **MRR:** mean reciprocal gold rank.
- **Gold selection margin:** gold score minus the best alternative, with an unretrieved gold term
  scored as -1. This is coverage-aware; the ordinary first-versus-second margin is misleading when
  the baseline abstains on difficult cases.
- **Negative-control selectivity:** negative cases that correctly abstain.

## Result

| Metric | Surface baseline | Context retrieval | Delta |
|---|---:|---:|---:|
| Correction sensitivity@5 | 0.4167 | 1.0000 | +0.5833 |
| Selection accuracy@1 | 0.4167 | 1.0000 | +0.5833 |
| MRR | 0.4167 | 1.0000 | +0.5833 |
| Mean gold selection margin | -0.4390 | 0.3212 | +0.7602 |
| Negative-control selectivity | 1.0000 | 1.0000 | 0.0000 |
| Auto-commits | 0 | 0 | 0 |

The test supports the narrow hypothesis that corroborating language-specific context can rescue
and rank controlled candidates on constructed hard cases. It does **not** establish the effect
size on real transcripts. The perfect context-arm score is expected from a tiny proof-of-concept
fixture designed to exercise retrieval, not from a held-out benchmark.

An initial single-cue rule produced a false suggestion on a Hindi diabetes negative control. The
checked-in experiment therefore requires two independent cues plus minimum surface support. This
is an important safety finding: diagnosis or generic clinical context alone must not be enough to
select a medication candidate.

## Run and inspect

```powershell
uv run python scripts/run_context_rag_experiment.py
uv run python scripts/run_context_rag_experiment.py --output context-rag-report.json
uv run pytest tests/test_context_rag_experiment.py
```

The JSON output includes per-case candidates, surface/context score components, provenance,
decisions, and per-language slices. Generated reports should not be committed.

## Next validation step

Before considering integration, create a versioned, held-out dataset with native-speaker and
clinical review, substantially more negative controls, realistic ASR/mistranslation errors,
code-switching, mixed scripts, high-risk/LASA terms, and context cues shared by competing terms.
Compare against the production matcher end to end, including span detection, and pre-register
thresholds. Any future context score must be added explicitly to `ScoreBreakdown`; automatic
candidate commitment must remain disabled.
