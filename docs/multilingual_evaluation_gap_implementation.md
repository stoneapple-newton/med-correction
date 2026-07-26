# Multilingual evaluation gap implementation

Updated 2026-07-26. This tracks implementation against the sibling project's
`docs/multilingual-evaluation-gap-analysis.md`. It is an engineering evaluation, not clinical
validation.

## Implemented

- The evaluation contract now separates `benchmark_key`, `canonical_term`, `source_vocabulary`,
  `source_release`, and `authority_concept_id`. Legacy fixture fields remain readable.
- Concept-identity scoring is limited to gold IDs whose namespace exists in the evaluated
  dictionary. Incompatible namespaces are explicitly `not_applicable` with an excluded-span count.
- Reports include language-matched terminology and candidate coverage, surface-only candidate
  coverage, oracle reranking accuracy, split slices, terminology inventory, and deterministic error
  classes (`terminology_absence`, `candidate_absence`, `detector_miss`, `boundary_error`,
  `ranking_error`, `policy_abstention`, and `false_positive`).
- `medterm-evaluate --output` writes a machine-readable report with the dataset hash, code commit,
  dirty-worktree state, dictionary and index versions, optional-model availability, and evaluation
  settings.
- Terminology artifacts and candidate evidence preserve source vocabulary/release, licensing, and
  review status. Missing metadata is honestly represented as `unspecified`, `unverified`, or
  `unreviewed`; it is not inferred.
- Span selection prefers the shortest competing span within a 0.04 retrieval-score tolerance,
  anchors explicitly low-confidence ASR tokens to their supplied offsets, and suppresses context
  windows that merely contain a safe exact dictionary match. Dose adjacency is still evaluated
  from bounded surrounding text, so tighter highlights do not remove the safety reason.
- Retrieval has a deterministic, language-filtered bounded Levenshtein mutation channel for terms
  of at least four normalized characters. The channel and score are exposed in candidate evidence.
  It cannot create a term outside the versioned dictionary.

## Frozen benchmark comparison

The baseline is the 770-record result recorded in the gap analysis. The current report is
`data/evaluation_multilingual_v1.report.json`.

| Gate | Baseline | Current | Result |
|---|---:|---:|---|
| Exact-boundary recall | 45.76% | 61.21% | P1 span gate met (+15.45 points) |
| Overlap recall | 73.33% | 72.42% | P1 span gate met (-0.91 points) |
| Negative-control selectivity | 78.18% | 85.45% | Improved, final 95% target not met |
| Correct-term triggers | 24 | 17 | Improved, isolated target of at most 5 not met |
| Automatic commitments | 0 | 0 | Safety invariant met |

These numbers are development results on the same synthetic benchmark, not held-out estimates.
The bounded mutation channel is implemented, but deletion and substitution recall gates are not
claimed: there is no held-out mutation set and most evaluated languages still lack terminology.

> **Important evaluation caveat:** this rerun is still the dictionary baseline. The Term Trace
> evaluator uses its own versioned `DictionaryIndex` terminology artifact; it is not connected to
> a Chroma or other context-RAG retrieval backend. The report therefore still shows zero
> language-matched terminology coverage for the European-language slices. Connecting the
> evaluator to the versioned RAG backend, recording that backend's collection/version provenance,
> and rerunning this same dataset is the next required step before any RAG uplift can be measured
> or claimed.

## Evidence from coverage diagnostics

Language-matched terminology coverage is 100% for Chinese, Japanese, Korean, and Hindi, 80% for
English, and 0% for Spanish, French, German, Italian, Portuguese, and Russian. The report keeps
cross-language surface matches separate so German drug-name overlap cannot masquerade as German
terminology coverage. Overall language-matched terminology coverage is 43.64%.

This confirms terminology absence as the leading correction bottleneck. It also explains why
correct-term suppression cannot safely reach five triggers: the detector cannot identify an exact,
safe term in a language absent from its controlled dictionary.

## Remaining plan and gates

1. Obtain versioned, redistributable terminology for the six missing languages. Record source
   vocabulary, release, license status, stable concept ID, aliases, provenance, and native-speaker
   and clinical review state. Do not promote the benchmark's synthetic labels into production
   terminology.
2. Add local, opt-in adapters for licensed corpora identified in
   `docs/multilingual_dataset_research.md`. Keep source-specific results separate from synthetic
   results and never download data during normal tests.
3. Connect the evaluator to the versioned RAG retrieval backend and expose its backend, collection,
   embedding, and source-artifact versions in report provenance. Keep a selectable
   `DictionaryIndex` baseline so uplift is measured in a paired rerun rather than inferred across
   different datasets or settings.
4. Freeze separate development and held-out partitions before changing score weights or thresholds.
   Re-run terminology and language-matched candidate coverage; require at least 95% per evaluated
   language before ranking work.
5. After coverage exists, repeat the correct-term suppression gate (at most five triggers with
   overlap recall at least 72.33%) and the deletion/substitution gate (at least +5 points each with
   no more than a one-point selectivity loss) on held-out data.
6. Tune ranking only on records where a language-matched gold candidate is already retrieved. Keep
   every score component and decision reason auditable.
7. Require review of terminology and error severity, confidence intervals where sample sizes allow,
   zero automatic commitments, preserved offsets, and passing high-risk/LASA/dose/mixed-script/
   uncertain-language policy tests before any production-readiness claim.

After committing code, regenerate the report so `code_worktree_dirty` is false and the recorded
commit contains the evaluated implementation. Metric JSON is deterministic when the dataset,
dictionary, settings, optional dependencies, and code state are locked.
