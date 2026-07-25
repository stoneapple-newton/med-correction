# AGENTS.md

## Project overview

Term Trace is an English-first, review-first service for detecting medical-term spans that may
have been mistranscribed and ranking controlled-dictionary candidates. It accepts text or audio,
preserves original character offsets, and returns auditable evidence for a human reviewer.

This is a reviewer-assistance prototype, not a diagnostic, prescribing, or medication-safety
system. Do not describe or implement it as a medical device without an explicit, separately
validated product requirement.

## Core invariants

- Never silently rewrite the source transcript.
- Keep automatic candidate commitment disabled. Matching outcomes must remain `review` or
  `abstain`; `auto_commit_enabled` must remain `false` unless the project requirements and clinical
  validation strategy are explicitly changed.
- Preserve original text and character offsets through normalization and span detection.
- Expose uncertainty and evidence: scores, pronunciation source, language confidence, decision
  reasons, terminology version, and provenance must remain auditable.
- Force human review for high-risk/LASA terms, dose-adjacent matches, low score margins,
  uncertain language, and mixed scripts.
- Prefer abstention to fabricating a transcript, term, pronunciation, or diagnosis.
- Do not infer a patient's diagnosis solely from a medication match.
- Treat audio, transcripts, audit records, and reviewer identities as potentially sensitive health
  information. Do not print or commit real patient data, credentials, or local audit databases.

## Architecture

- `src/medterm/api.py`: FastAPI routes and service wiring.
- `src/medterm/matcher.py`: candidate retrieval, fixed scoring, ranking, and review/abstain policy.
- `src/medterm/detection.py`: recall-oriented suspicious-span detection.
- `src/medterm/normalization.py`: loss-minimizing normalization, tokenization, offsets, and
  structured medication fields.
- `src/medterm/pronunciation.py`: curated/CMUdict/Epitran pronunciation generation and phonetic
  similarity.
- `src/medterm/terminology.py`: versioned terminology artifact creation, loading, and retrieval.
- `src/medterm/language.py`: optional fastText language ID with deterministic script fallback.
- `src/medterm/asr.py`: optional local faster-whisper transcription adapter.
- `src/medterm/audit.py`: SQLite match and reviewer-decision records.
- `src/medterm/evaluation.py`: offline evaluation metrics.
- `reviewer_app.py`: Streamlit review console.
- `data/source_terms.csv`: small development terminology; it is not a production clinical
  vocabulary.
- `tests/`: unit and API behavior tests.

The core term matcher is deterministic and does not use a generative LLM. fastText and
faster-whisper are optional local models for language identification and ASR respectively. Do not
add an external LLM or transmit clinical text to a hosted model without an explicit requirement,
privacy review, and tests covering failure behavior.

## Development setup

Use Python 3.11–3.13 and `uv` for dependency and command execution.

```powershell
uv sync --python 3.12 --extra dev --extra phonetics
uv run medterm-build
uv run medterm-api --host 127.0.0.1 --port 8000
```

Run the reviewer UI separately:

```powershell
uv run streamlit run reviewer_app.py --server.port 8501
```

Optional local model support:

```powershell
uv sync --extra language-id --extra asr
uv run python scripts/download_fasttext_model.py data/models/lid.176.ftz
$env:MEDTERM_FASTTEXT_MODEL_PATH="data/models/lid.176.ftz"
$env:MEDTERM_WHISPER_MODEL="small.en"
```

The service must continue to behave explicitly and safely when optional models are unavailable.

## Change guidelines

- Keep business logic in `src/medterm`; keep API handlers and Streamlit code thin.
- Use typed Python and follow the existing Pydantic models and dependency-injection patterns.
- Use `pathlib.Path` for filesystem paths and settings with the `MEDTERM_` prefix.
- Keep normalization loss-minimizing. Any normalized representation used for retrieval must not
  replace the stored original text.
- Make scoring changes explicit, deterministic, and visible in `ScoreBreakdown`. Do not add hidden
  heuristics that cannot be explained to a reviewer.
- Do not tune thresholds against the bundled sample and claim clinical performance. Threshold or
  weight changes require held-out, versioned evaluation data and before/after metrics.
- Preserve terminology concept IDs, provenance, artifact versions, aliases, language, risk tier,
  and curated pronunciations when changing the artifact schema.
- Add migrations or compatibility handling when persisted audit or artifact schemas change.
- Do not commit generated dictionary artifacts, SQLite audit files, uploaded audio, model files,
  virtual environments, coverage output, or build output.

## Testing and verification

For ordinary Python changes, run:

```powershell
uv run ruff check .
uv run pytest
```

For matching, normalization, scoring, terminology, or safety-policy changes, also run:

```powershell
uv run pytest --cov=medterm --cov-report=term-missing
uv run medterm-evaluate data/evaluation_sample.jsonl
```

For UI changes, start the API and Streamlit app and exercise the relevant review flow. For Docker
or deployment changes, verify both services with:

```powershell
docker compose up --build
```

Add regression tests for every behavior change. Important cases include:

- ASR-like multi-token confusions such as `met for men` versus `metformin`.
- Correct original character offsets after Unicode normalization.
- Exact, spelling, and phonetic candidate retrieval.
- Low-confidence, low-margin, high-risk, LASA, dose-adjacent, multilingual, code-switch, and
  mixed-script inputs.
- Empty, negative-control, and unresolved inputs that should abstain.
- Missing optional ASR or language models.
- Audit persistence and reviewer actions.
- The invariant that auto-commit remains disabled.

Do not weaken a safety assertion merely to make a failing test pass. If expected behavior changes,
document the reason and update implementation, tests, and evaluation fixtures together.

## Terminology and data

The bundled CSV is development-only. Production terminology work must account for source
licensing, release versions, and clinical review. Do not add proprietary UMLS/RxNorm-derived data
unless its license permits repository distribution.

Use synthetic, de-identified examples in tests and documentation. Never add real patient names,
record identifiers, audio, transcripts, or reviewer credentials.

## Documentation

Update `README.md` when commands, configuration, API behavior, optional dependencies, or safety
boundaries change. Clearly distinguish implemented behavior from future or recommended behavior,
and avoid claims of clinical validation unless supported by documented evidence.
