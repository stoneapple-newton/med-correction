# Term Trace

Term Trace is a review-first service for finding medical terms that may have
been mistranscribed and matching them to a controlled dictionary. It accepts text or audio plus
an optional transcript, detects suspicious 1–5 token spans, generates pronunciations, retrieves
dictionary candidates through spelling and phonetic channels, and returns an auditable ranked
result. It never silently rewrites the transcript.

This repository is an implementable MVP of the supplied *Practical Plan for Medical Term Span
Detection and Dictionary Matching*. The bundled terminology is deliberately a small development
sample. Production use requires a licensed, validated terminology release and clinical review.

## What is implemented

- Versioned offline dictionary builds from CSV/JSON, including aliases, provenance, language,
  risk tier, priors, and curated pronunciations.
- RxNorm `RXNCONSO.RRF` conversion for building a larger source file from an official release.
- Loss-minimizing Unicode normalization and structured extraction of strength, route, and dosage
  form tokens.
- Script detection and a fastText `lid.176` adapter with an explicit script-based fallback.
- Offset-preserving approximate substring retrieval for Chinese, Japanese, Korean, and Hindi,
  including text whose term boundaries are not represented by spaces. Normal Japanese
  kanji/hiragana/katakana composition is not mislabeled as suspicious mixed-script text.
- Recall-oriented 1–5 token suspicious-span detection using OOV, ASR confidence, N-best
  disagreement, boundary, mixed-script, and phonetic-neighbor signals.
- Tiered pronunciation generation: curated IPA, English CMUdict, Epitran for configured languages,
  then a deterministic fallback.
- Exact, RapidFuzz orthographic, and IPA trigram retrieval channels, followed by phonetic-first
  re-ranking. PanPhon weighted feature edit distance is used when the optional phonetics extra is
  installed.
- Banded decisions: abstain or review. Auto-commit is disabled in the schema, service, UI, and
  health response.
- FastAPI text and audio contracts, optional faster-whisper ASR, SQLite audit/review records, and
  a Streamlit reviewer console with audio, context, pronunciation evidence, score breakdowns,
  safety warnings, and reviewer actions.
- Docker Compose for the API and reviewer console.
- A synthetic-only LangGraph experiment that uses DeepSeek through its OpenAI-compatible API,
  validates model offsets and controlled-dictionary IDs, rejects invented/unchanged terms, and
  traces offline evaluations to LangSmith. It is not connected to the clinical API path.

## Architecture

```text
OFFLINE                                      ONLINE
source_terms.csv / RXNCONSO.RRF              text or audio + ASR metadata
          │                                             │
          ▼                                             ▼
normalize aliases + generate IPA            normalize + language/script tags
          │                                             │
          ▼                                             ▼
versioned dictionary.json ───────────────► suspicious 1–5 token windows
                                                        │
                                            exact + spelling + IPA blocks
                                                        │
                                                        ▼
                                           phonetic-first re-ranking
                                                        │
                                               abstain or review
                                                        │
                                                        ▼
                                              SQLite replayable audit
```

## Quick start

Python 3.11–3.13 is recommended.

```powershell
uv sync --python 3.12 --extra dev --extra phonetics
uv run medterm-build
uv run medterm-api --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
uv run streamlit run reviewer_app.py --server.port 8501
```

Open `http://localhost:8501`. API documentation is available at
`http://localhost:8000/docs`.

The same stack can be started with:

```powershell
docker compose up --build
```

## API examples

Text with no ASR metadata:

```powershell
$body = @{
  text = "The patient takes met for men 500 mg with breakfast."
  locale = "en-US"
  top_k = 5
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/match `
  -ContentType application/json -Body $body
```

The request also accepts token confidence/timing metadata and N-best hypotheses:

```json
{
  "text": "met for men",
  "locale": "en-US",
  "tokens": [
    {"text": "met", "char_start": 0, "char_end": 3, "confidence": 0.72,
     "start_time": 1.0, "end_time": 1.2},
    {"text": "for", "char_start": 4, "char_end": 7, "confidence": 0.61,
     "start_time": 1.2, "end_time": 1.4},
    {"text": "men", "char_start": 8, "char_end": 11, "confidence": 0.70,
     "start_time": 1.4, "end_time": 1.8}
  ],
  "n_best": [
    {"text": "metformin", "confidence": 0.57},
    {"text": "met for men", "confidence": 0.43}
  ]
}
```

Audio is accepted at `POST /v1/match/audio` as multipart form data. Include `transcript` to match
an existing ASR result. To transcribe directly, install `--extra asr` and set
`MEDTERM_WHISPER_MODEL` to a faster-whisper model name. If neither is available, the endpoint
returns an explicit `503` instead of fabricating a transcript.

Reviewer decisions are recorded at:

```text
POST /v1/matches/{request_id}/spans/{span_id}/review
GET  /v1/matches/{request_id}/reviews
GET  /v1/matches/{request_id}
GET  /v1/dictionary/search?q=Coumadin
```

## Terminology

`data/source_terms.csv` is the canonical development source. Fields are:

| Field | Meaning |
|---|---|
| `concept_id` | Stable controlled-dictionary ID, such as `RxCUI:6809` |
| `term` | Preferred display term |
| `aliases` | Pipe-separated synonyms, brands, or approved abbreviations |
| `language` | BCP-47 base language |
| `risk_tier` | `standard`, `high`, `lasa`, or `critical` |
| `prior` | Validated local frequency prior from 0 to 1 |
| `provenance` | Source vocabulary/release |
| `pronunciations` | Pipe-separated curated IPA overrides |

Build an artifact:

```powershell
uv run medterm-build --source data/source_terms.csv `
  --output data/artifacts/dictionary.json --version local_2026_07_v1
```

For an official extracted RxNorm release:

```powershell
uv run python scripts/import_rxnorm.py C:\rxnorm\rrf\RXNCONSO.RRF data/rxnorm_terms.csv
uv run medterm-build --source data/rxnorm_terms.csv --output data/artifacts/dictionary.json
```

RxNorm and UMLS source licenses and release terms must be reviewed before distribution. Add
RxTerms/local formulary aliases to the same source schema and validate critical pronunciations
with a clinical reviewer.

## Optional model adapters

Install and download fastText language identification:

```powershell
uv sync --extra language-id
uv run python scripts/download_fasttext_model.py data/models/lid.176.ftz
$env:MEDTERM_FASTTEXT_MODEL_PATH="data/models/lid.176.ftz"
```

Install multilingual phonetics and optional local ASR:

```powershell
uv sync --extra phonetics --extra asr
$env:MEDTERM_WHISPER_MODEL="small.en"
```

The core service remains deterministic without these large models. Its response identifies the
actual pronunciation source and language confidence so fallback behavior is visible in audit data.

## Configuration

All settings use the `MEDTERM_` prefix:

| Variable | Default |
|---|---|
| `ARTIFACT_PATH` | `data/artifacts/dictionary.json` |
| `SOURCE_TERMS_PATH` | `data/source_terms.csv` |
| `AUDIT_DB_PATH` | `data/audit/medterm.sqlite3` |
| `AUDIO_DIR` | `data/audio` |
| `FASTTEXT_MODEL_PATH` | unset |
| `WHISPER_MODEL` | unset |
| `REVIEW_THRESHOLD` | `0.55` |
| `SUSPECT_THRESHOLD` | `0.30` |
| `FORCE_REVIEW_MARGIN` | `0.12` |
| `MAX_SPAN_TOKENS` | `5` |
| `TARGET_SENSITIVITY` | `0.95` (evaluation target only) |
| `TARGET_SELECTIVITY` | `0.95` (evaluation target only) |

Thresholds and score weights are seeds from the plan, not clinically validated constants. Tune
them only on a held-out, versioned evaluation set and monitor medical-term precision/recall,
false-correction rate, recall@1/5, MRR, review rate, abstain rate, and dose-adjacent/LASA errors.

Evaluation also reports the requested targets through `targets`: sensitivity is overlap span
recall, while selectivity is one minus the negative-record false-positive rate. Both default to
`0.95` via `MEDTERM_TARGET_SENSITIVITY` and `MEDTERM_TARGET_SELECTIVITY`. These are acceptance
targets, not runtime score thresholds, and reports state explicitly whether each target was met.

## Verification

```powershell
uv run ruff check .
uv run pytest --cov=medterm --cov-report=term-missing
```

Tests cover artifact builds, normalization, original offsets, script evidence, ASR-like term
confusions, N-best/metadata handling, abstention safety, API/audio contracts, and audit/review
round trips. Browser verification of the Streamlit workflow is kept as a separate end-to-end gate.

Run the versioned synthetic detection evaluation:

```powershell
uv run medterm-evaluate data/evaluation_detection_v1.jsonl
```

The dataset contains only synthetic examples. Its gold spans mark suspicious source text that a
reviewer should inspect; correctly transcribed medical terms are negative controls because the
detector is not intended to flag every medical mention. Cases are tagged for slice reporting,
including `asr_boundary`, `spelling`, `dose_adjacent`, `high_risk`, `mixed_script`,
`low_confidence`, `unicode`, `multi_span`, and `negative`.

Each JSONL record has a stable `case_id`, `dataset_version`, original `text`, optional `locale`,
`tokens`, `n_best`, `reference_transcript`, and `asr_hypothesis`, plus `tags` and `gold_spans`.
Every gold span uses Python-style half-open character offsets (`char_start` inclusive,
`char_end` exclusive), a matching `span_text`, and optionally `gold_term` and `concept_id` for
candidate-ranking evaluation. Loading fails on duplicate case IDs, invalid ranges, or a
`span_text` that does not exactly equal the original-text slice.

The report separates exact-boundary from overlap precision/recall/F1, uses one-to-one span
matching, and includes false positives, false negatives, boundary mismatches, per-tag slices,
negative-record false-positive rate, recall@1/5, MRR, review/abstain rates, and the observed
auto-commit count. JiWER WER/CER are added when transcript pairs are present. The older
`data/evaluation_sample.jsonl` remains a small ASR smoke fixture.

Do not interpret results on this development dataset as clinical performance or tune thresholds
against it. Before making performance claims, create a separately versioned, held-out,
license-reviewed and clinically reviewed set with representative negative controls, accents,
code-switching, terminology releases, and ASR conditions.

### Multilingual correction benchmark

`data/evaluation_multilingual_v1.jsonl` contains 770 synthetic records: 70 each for English,
Spanish, French, German, Italian, Portuguese, Russian, Mandarin Chinese, Japanese, Korean, and
Hindi. For each language, ten medical terms are exercised under character deletion,
transposition, boundary insertion, substitution, low ASR confidence, script confusion, and a
correct-term negative control. It includes no real patient data or copied third-party dataset rows.

Regenerate or verify the committed dataset and manifest deterministically:

```powershell
uv run python scripts/build_multilingual_evaluation.py
uv run python scripts/build_multilingual_evaluation.py --check
uv run medterm-evaluate data/evaluation_multilingual_v1.jsonl
```

The evaluator reports canonical correction accuracy@1, recall@5, and MRR in addition to concept-ID
ranking and detection metrics. `correction_slices` provides the same correction measures by
language, region, category, and error condition. The current bundled terminology is English-only,
so multilingual candidate scores are expected to expose missing terminology coverage rather than
produce strong correction results.

The [dataset manifest](data/evaluation_multilingual_v1.manifest.json) records exact coverage and
limitations. The [online-resource review](docs/multilingual_dataset_research.md) documents existing
medical spelling, ASR, and entity corpora and why they are not copied into this repository.

The development terminology now contains matching Chinese, Japanese, Korean, and Hindi entries
for the ten synthetic benchmark concepts. Their provenance is
`local-multilingual-development-unreviewed`: they are test coverage, require native-speaker and
clinical review, and must not be promoted as licensed production terminology.

### LangGraph, DeepSeek, and LangSmith experiment

Install the isolated evaluation dependencies:

```powershell
uv sync --extra dev --extra phonetics --extra llm-evaluation
```

The script reads the DeepSeek key from the requested `deep-seek-api` environment variable and the
LangSmith key from `LANGSMITH_API_KEY`. Defaults are `deepseek-v4-flash` and
`https://api.deepseek.com`; model output is non-thinking JSON to avoid truncating the structured
answer. Only synthetic records are eligible for upload:

```powershell
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_PROJECT = "term-trace-development"
uv run python scripts/langsmith_evaluate.py upload
uv run python scripts/langsmith_evaluate.py evaluate --limit 56 --max-concurrency 2
```

The upload is idempotent and refuses to mutate an existing dataset whose count differs. Use a new
versioned dataset name for changed data. Pass `--limit 0` to evaluate all 280 Asian examples.

The framework choice is deliberately LangGraph rather than Deep Agents: this workflow needs three
fixed, auditable steps (prepare controlled catalog, propose, validate), not autonomous planning,
filesystem access, memory, or subagents. Deep Agents would expand the attack and privacy surface
without adding a correction-specific capability.

On 2026-07-26, a balanced 56-record LangSmith experiment (49 positive, seven negative synthetic
cases) completed without errors. The standalone DeepSeek graph scored 0.735 sensitivity,
0.735 correction@1, 1.000 selectivity, and 1.000 auto-commit safety. The deterministic matcher on
the identical examples scored 0.714 sensitivity, 0.714 correction@1, and 1.000 selectivity. The
model improved both sensitivity and correction@1 by 0.020, but it did not meet the joint 0.95
targets or the bar for production integration. These development results are not
clinical-performance estimates.

## Safety boundary

This is a reviewer-assistance prototype, not a medical device or clinical decision system. It does
not silently edit text, prescribe, diagnose, or determine medication safety. High-risk, mixed-script,
low-margin, and dose-adjacent matches are visibly forced to review. Production rollout requires
terminology licensing, security/privacy review, clinical validation, reviewer access controls,
retention rules for audio/PHI, observability, and a documented incident process.
