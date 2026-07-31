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
- License-gated, release-scoped RxNorm streaming import with disk-backed alias grouping,
  deterministic shards, checksums, resumable checkpoints, and dictionary finalization.
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

For an approved official extracted RxNorm release, register its checksum and reviewed license
evidence before importing it:

```powershell
uv run medterm-register-source `
  --source C:\rxnorm\rrf\RXNCONSO.RRF --release-id rxnorm-YYYYMMDD `
  --source-release YYYYMMDD --license-status approved_for_local_processing `
  --license-approval-id <approval-id> --license-owner <owner> `
  --license-evidence <evidence-reference> --license-approval-date YYYY-MM-DD

uv run medterm-import-terminology `
  --manifest data/builds/rxnorm-YYYYMMDD/source/source_manifest.json `
  --output data/builds/rxnorm-YYYYMMDD/import --batch-size 25000 --resume

uv run medterm-finalize-dictionary `
  --import-dir data/builds/rxnorm-YYYYMMDD/import `
  --output data/builds/rxnorm-YYYYMMDD/terminology
```

RxNorm and UMLS source licenses and release terms must be reviewed before distribution. Add
RxTerms/local formulary aliases to the same source schema and validate critical pronunciations
with a clinical reviewer.

Imported concepts default to `risk_tier=unclassified` and `lasa_status=unclassified`; they are not
silently treated as standard risk. A `--max-concepts` run is always labeled partial, scans the
registered source for aliases of selected concepts, and cannot resume as a full build.

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
| `TTS_MODEL_ID` | `hexgrad/Kokoro-82M` |
| `TTS_VOICE` | `af_heart` |
| `TTS_LANGUAGE_CODE` | `a` (American English) |
| `TTS_DEVICE` | `auto` |
| `TTS_SPEED` | `1.0` |
| `TTS_OUTPUT_DIR` | `data/audio/tts` |
| `TTS_MANIFEST_PATH` | `data/artifacts/kokoro_references.jsonl` |
| `TTS_INDEX_PATH` | `data/artifacts/kokoro_vectors.npz` |
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
uv run medterm-evaluate data/evaluation_multilingual_v1.jsonl `
  --output data/evaluation_multilingual_v1.report.json
```

The evaluator reports canonical correction accuracy@1, recall@5, and MRR separately from
namespace-valid concept identity. Gold spans keep synthetic `benchmark_key` values separate from
`authority_concept_id`; incompatible ID namespaces are reported as `not_applicable`, never as zero
retrieval. `coverage_slices` distinguishes language-matched candidate coverage from coincidental
cross-language surface matches, while `terminology_inventory` exposes source, release, licensing,
and review-state coverage. The JSON report also records the dictionary/index versions, dataset
hash, code commit and dirty-worktree state, optional model availability, and evaluation settings.

The bundled terminology covers English plus Chinese, Japanese, Korean, and Hindi development
entries, but it has no genuine Spanish, French, German, Italian, Portuguese, or Russian coverage.
Multilingual scores are therefore expected to expose missing terminology rather than demonstrate
clinical correction quality. See
[the implementation status and remaining gates](docs/multilingual_evaluation_gap_implementation.md).

Important: `medterm-evaluate` still uses the versioned `DictionaryIndex` artifact, not a Chroma or
other context-RAG backend. Its multilingual report remains the dictionary baseline and continues
to show zero language-matched terminology coverage for the European-language slices. The generated
report records this under `evaluation_scope`. Connecting a versioned RAG backend to the evaluator
and rerunning the same dataset is required before measuring or claiming uplift.

The [dataset manifest](data/evaluation_multilingual_v1.manifest.json) records exact coverage and
limitations. The [online-resource review](docs/multilingual_dataset_research.md) documents existing
medical spelling, ASR, and entity corpora and why they are not copied into this repository.

The development terminology now contains matching Chinese, Japanese, Korean, and Hindi entries
for the ten synthetic benchmark concepts. Their provenance is
`local-multilingual-development-unreviewed`: they are test coverage, require native-speaker and
clinical review, and must not be promoted as licensed production terminology.

### Local language-context retrieval experiment

The isolated dummy RAG experiment compares surface-only retrieval with deterministic reranking
from a small synthetic, language-specific context database. It is not connected to the API and
does not use Chroma or call an LLM. The retriever preserves the supplied span offsets, filters records by
language, exposes surface/context scores and provenance, and returns only `review` or `abstain`.

```powershell
uv run python scripts/run_context_rag_experiment.py
uv run pytest tests/test_context_rag_experiment.py
```

The bundled 17-case development check (12 positive, five negative; English, Chinese, Japanese,
Korean, and Hindi) increased synthetic correction sensitivity@5 and selection accuracy@1 from
0.4167 to 1.0000 while keeping negative-control selectivity at 1.0000. These deliberately
constructed cases test whether a context signal *can* help; they do not estimate real-world or
clinical performance. See [the experiment note](docs/context_rag_experiment.md) for definitions,
guardrails, results, and the validation work required before any product integration.

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

### Kokoro synthetic term audio and sound vectors

Kokoro-82M is an optional local TTS stage for creating development reference pronunciations from
the controlled terminology. The model and voice weights are downloaded from
`hexgrad/Kokoro-82M` on first use and remain in the local Hugging Face cache. Install the complete
TTS-to-vector pipeline with:

```powershell
uv sync --python 3.12 --extra dev --extra phonetics --extra tts `
  --extra speech-embeddings --extra vector-index
```

On Windows, install `espeak-ng` from its official MSI if English out-of-dictionary fallback is
needed. Curated terminology pronunciations are passed directly to Kokoro when present; otherwise
Kokoro's grapheme-to-phoneme layer is used. Generate WAV files, an auditable JSONL manifest, and a
sound-vector index in one command:

```powershell
uv run medterm-build
uv run medterm-tts-build --language en --language-code a --voice af_heart
uv run medterm-audio-search data/artifacts/kokoro_vectors.npz `
  data/audio/query.wav --language en --concept-type medical_term
```

For a quick local check, add `--limit 1`. Use `--tts-device cpu` or `--tts-device cuda` to force a
TTS device, and `--embedding-device` independently for the sound-vector encoder. Existing WAVs
are reused unless `--overwrite` is supplied. The generated defaults are:

- `data/audio/tts/*.wav` — isolated 24 kHz PCM term pronunciations.
- `data/artifacts/kokoro_references.jsonl` — relative paths plus TTS, phoneme, and terminology
  provenance.
- `data/artifacts/kokoro_vectors.npz` — normalized speech-content vectors and provenance.

The default command is intentionally English-only. Other Kokoro language codes require matching
voices and, for Japanese or Mandarin, the corresponding Misaki language extras. Korean is not a
Kokoro-82M v1.0 language and must not be approximated with another language pipeline.

Synthetic speech is not a validated human pronunciation corpus and must not be represented as
one. These vectors are development references for candidate retrieval and projection experiments;
they require held-out evaluation on approved human recordings. Retrieval remains `review` or
`abstain`, and `auto_commit_enabled` remains `false`.

For a release-scoped active-reference manifest, create resumable vector shards and finalize a
language-specific index:

```powershell
uv run medterm-vector-batch `
  --manifest data/builds/<release-id>/tts/en/active_references.jsonl `
  --output data/builds/<release-id>/vectors/en `
  --micro-batch-size 32 --shard-size 5000 --device cuda --resume

uv run medterm-finalize-index `
  --vectors data/builds/<release-id>/vectors/en `
  --output data/builds/<release-id>/indexes/kokoro_vectors_en.npz
```

The encoder duration-orders waveforms within each batch, forwards attention masks, and excludes
padded feature frames from pooling. Finalization rejects duplicate, zero, non-finite, corrupt, or
dimension-incompatible vectors. CUDA out-of-memory retry splits only the affected micro-batch and
does not change the encoder or projection configuration.

The release lifecycle is being implemented in phases. Source registration, streaming RxNorm
import, dictionary finalization, canonical TTS reference/rendition identity, batched encoding,
vector shards, index finalization, and the release-scoped med-rag batch runner are implemented.
Publication/revocation and garbage collection remain proposed and must not be represented as
available commands.

For the registered local med-rag snapshot, export a deterministic SQLite slice without starting
the expensive TTS/vector stages:

```powershell
$root = "data/builds/med-rag-official-local-v1"
$database = "C:/Users/Newto/Documents/project/med-rag/data/medical_terms.sqlite"
$checksum = "db88ef38e04b650efb7a4b81cdbe3a21ca6c335facd05c271e8e99f92537d642"

uv run medterm-medrag-batch `
  --stage export `
  --sqlite $database `
  --expected-sqlite-sha256 $checksum `
  --build-root $root `
  --release-id med-rag-official-local-v1 `
  --batch-id batch-000003-5000 `
  --source-language en `
  --offset 5500 `
  --limit 5000
```

Run the same immutable arguments with `--stage tts`, then `--stage vectors`, or use `--stage all`
for a new batch. Defaults match the handoff: English `a`/`af_heart`, Mandarin
`z`/`zf_xiaobei`, CUDA, statistics pooling, vector micro-batches of 16, shards of 5,000, and TTS
checkpoints every 500 terms. Use `--source-language zh-Hans` for the registered Chinese source;
the export retains `zh-Hans` in provenance and writes `zh` only to generated artifacts. A reused
batch ID is rejected if its source checksum, offset, limit, language, release, or policy differs.

See [the bulk structure evaluation plan](docs/bulk_structure_evaluation_plan.md) for interruption,
resume, resource, pronunciation-review, held-out human-audio retrieval, reconciliation, and
release-gate requirements.

Build every Kokoro-supported language present in the bundled terminology with the PowerShell
runner:

```powershell
.\scripts\build_all_tts_vectors.ps1 -SyncDependencies
```

Use `-Limit 1` for a quick smoke run, `-Overwrite` to regenerate existing WAVs,
`-TtsDevice cpu|cuda`, and `-EmbeddingDevice cpu|cuda` to override device selection. The runner
creates separate `en`, `zh`, `ja`, and `hi` audio directories, manifests, and vector indexes.
The first dependency sync also downloads the approximately 770 MB Japanese UniDic data when it is
missing. Korean is reported and skipped because Kokoro-82M v1.0 does not support it.

Every run writes a timestamped log under `data/logs`. Select a subset and an explicit log path with:

```powershell
.\scripts\build_all_tts_vectors.ps1 `
  -Languages en,zh,ja `
  -TtsDevice cuda `
  -EmbeddingDevice cuda `
  -LogPath data/logs/tts_vectors_en_zh_ja.log
```

### Acoustic pronunciation-vector prototype (Strategy A)

The optional Strategy A prototype converts isolated query recordings and controlled reference
pronunciations into fixed-length multilingual speech-content vectors. It defaults to
`facebook/wav2vec2-xls-r-300m`, 16 kHz mono input, statistics pooling, CUDA when available, and
exact cosine retrieval. Install the model stack and optional FAISS backend with:

```powershell
uv sync --python 3.12 --extra dev --extra phonetics --extra speech-embeddings --extra vector-index
```

On Windows, `uv` resolves Torch from PyTorch's CUDA 13.0 wheel index. This is appropriate for the
tested RTX 4070 laptop and its NVIDIA 580.97 driver; other environments should select a supported
backend rather than copying that hardware assumption blindly.

Create a de-identified JSONL reference manifest; audio paths are relative to the manifest:

```json
{"reference_id":"metformin-en-us-01","audio_path":"audio/metformin-en-us-01.wav","concept_id":"LOCAL:METFORMIN","term":"metformin","language":"en","country":"US","pronunciation_type":"human","speaker_accent":"en-US","source":"validated_recording","concept_type":"medication"}
```

Build and query the local index:

```powershell
uv run medterm-audio-index data/reference_pronunciations.jsonl `
  data/artifacts/pronunciation_vectors.npz
uv run medterm-audio-search data/artifacts/pronunciation_vectors.npz `
  data/audio/query.wav --language en --country US --country GB --top-k 20
```

After assembling a versioned training manifest with at least two recordings for every concept,
train and apply the experimental supervised-contrastive 256-D projection:

```powershell
uv run medterm-audio-train data/training_pronunciations.jsonl `
  data/artifacts/medical-awe-v1.pt --device cuda --epochs 100
uv run medterm-audio-index data/reference_pronunciations.jsonl `
  data/artifacts/pronunciation_vectors.npz `
  --projection data/artifacts/medical-awe-v1.pt
uv run medterm-audio-search data/artifacts/pronunciation_vectors.npz `
  data/audio/query.wav --language en --top-k 20 `
  --projection data/artifacts/medical-awe-v1.pt
```

`MEDTERM_AUDIO_EMBEDDING_MODEL`, `MEDTERM_AUDIO_EMBEDDING_DEVICE`,
`MEDTERM_AUDIO_EMBEDDING_POOLING`, `MEDTERM_AUDIO_EMBEDDING_PROJECTION_CHECKPOINT`, and
`MEDTERM_AUDIO_EMBEDDING_MAX_SECONDS` configure the encoder. Without a validated projection
checkpoint, the output is explicitly a raw pooled baseline (2048 dimensions for XLS-R statistics
pooling), not a trained medical AWE. A checkpoint must declare itself trained; untrained or
incompatible query/index configurations are rejected. The index stores multiple vectors per
concept, filters metadata before ranking, excludes local audio paths, and reports whether FAISS or
NumPy performed the exact search.

Audio retrieval produces about 20 review candidates for later phoneme and ASR reranking. It never
rewrites a transcript or selects a medication automatically: responses remain `review` or
`abstain`, and `auto_commit_enabled` is always `false`. Reference and query recordings may contain
sensitive health information and must not be committed.

See [the Strategy A research and implementation notes](docs/strategy-a-speech-content-vectors.md).

A local execution check can be run against three synthetic or otherwise approved isolated-term
recordings:

```powershell
uv run python scripts/smoke_test_audio_embeddings.py `
  data/audio/metformin-reference.wav `
  data/audio/metformin-query.wav `
  data/audio/metoprolol.wav --device cuda
```

The initial 2026-07-26 RTX 4070 smoke test successfully exercised CUDA, 2048-D statistics pooling,
and FAISS, but the raw untrained baseline ranked the different medical term above the same-term
reference. That result confirms the pipeline runs and also confirms that contrastive projection
training and reranking are required; it is not an accuracy result.

## Safety boundary

This is a reviewer-assistance prototype, not a medical device or clinical decision system. It does
not silently edit text, prescribe, diagnose, or determine medication safety. High-risk, mixed-script,
low-margin, and dose-adjacent matches are visibly forced to review. Production rollout requires
terminology licensing, security/privacy review, clinical validation, reviewer access controls,
retention rules for audio/PHI, observability, and a documented incident process.
