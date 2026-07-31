# Bulk Terminology and Vector Structure Evaluation Plan

## Purpose and decision boundary

This plan evaluates whether the release-scoped bulk structure is correct, resumable, bounded, and
safe enough for a controlled rehearsal. It does not establish clinical validity. Synthetic TTS
quality and raw pooled speech-vector execution are evaluated separately from retrieval performance
on approved human recordings.

A release can pass structural evaluation while remaining blocked from publication or production
review. Missing license evidence, required numeric gates, approved human evaluation data, or a
validated projection checkpoint is a release blocker.

## Evaluation units

Evaluate immutable release artifacts rather than a mutable working directory:

- registered source manifest and source checksum;
- import stage hash, SQLite staging state, deterministic JSONL shards, rejects, and summary;
- finalized dictionary and inventory checksums;
- active TTS manifest with logical reference IDs, WAV rendition IDs, and checksums;
- vector checkpoint, shards, finalized per-language index, and index summary;
- release policy, run manifest, logs, and evaluation report.

Every report records the code commit and dirty-worktree state, artifact checksums, source/model
revisions, execution device/dtype, and `auto_commit_enabled=false`.

## Test ladder

### 1. Automated correctness gate

```powershell
uv run ruff check .
uv run pytest
uv run pytest --cov=medterm --cov-report=term-missing
uv run medterm-evaluate data/evaluation_sample.jsonl
```

Required assertions include alias preservation across checkpoints, deterministic shards, strict
partial/full separation, checksum validation, license blocking, valid resume reuse, device/path-free
logical TTS identities, byte-addressed renditions, masked batch/individual embedding parity, invalid
vector rejection, zero unexplained reconciliation differences, and review/abstain-only outcomes.

### 2. Synthetic multi-shard integration gate

Generate a non-patient RRF fixture large enough for at least three concept shards. Compare an
uninterrupted run with interruption/resume during parsing, after an import shard, and during vector
writing with a corrupt temporary/final shard. Canonical identities and final checksums must match.
Measure peak memory to confirm importer memory does not grow with the complete source inventory.

### 3. Bounded official-release rehearsal

After source/license approval, run a distinct `--max-concepts 25000` partial build. Capture import
throughput, database/disk size, peak RAM, TTS throughput and real-time factor, clip duration,
embedding throughput, peak VRAM, retries, failure classes, and bytes per artifact. A no-op resume
must regenerate zero valid WAVs and vectors. Measure CPU/float32 and configured CUDA dtype parity
separately; do not assume cross-device byte equality.

### 4. Pronunciation review gate

Sample every source shard and language, oversampling high-risk, LASA, mixed-script, long,
abbreviated, numeric, and G2P-generated terms. Record pass/fail, failure class, suggested
pronunciation, and defer decisions. Synthetic audio always remains labeled `synthetic_tts`.
Unclassified terms cannot be counted as ordinary-risk passes.

### 5. Held-out human-audio retrieval gate

Use an approved, versioned, license-reviewed set of isolated human recordings disjoint from
projection training and synthetic references. Include language, accent, voice, microphone, noise,
speaking-rate, and high-risk/LASA slices where policy permits. No patient recording enters the
repository.

Report rendition- and concept-level recall@1/5/20, MRR, macro/micro language and slice results,
abstain/review rates, terminal failures, nearest-confusion pairs, and LASA/high-risk error review.
Report the raw untrained pooled baseline separately from a validated projection. When speakers
contribute repeated clips, compute confidence intervals by resampling speakers, not clips.

Compare the text/dictionary baseline, raw pooled speech baseline, and validated projected vectors
on the same held-out records and candidate inventory.

## Release policy and gates

Before evaluation, the release owner creates versioned `release_policy.json` values for minimum
recall@1/5/20 and MRR overall and per required slice, sample/speaker counts, maximum TTS/vector
failure rates, memory/storage/runtime ceilings, CPU/CUDA parity tolerances, pronunciation-review
counts, and dependency license states. The maximum unexplained inventory difference is always zero.
This repository does not invent these thresholds from development data; a required missing gate
fails the release.

## Reconciliation

For each required language, reconcile reference/rendition IDs:

```text
planned = eligible + deferred + excluded
eligible = active_successful_renditions + failed_terminal
active_successful_renditions = indexed_renditions + vector_failures
indexed_renditions = final_index_rows
```

Superseded renditions remain audit history and are excluded from active counts. Every difference
needs a concrete failure/deferred/excluded record; any unexplained difference blocks finalization.

## Deliverables and go/no-go review

Deliver the immutable policy, manifests, inventory, index summary, structured tests/benchmarks,
pronunciation review, retrieval report, limitations, failures/deferred items, and checksums. The
review separately decides structural correctness, license compatibility, held-out retrieval gates,
and review-only safety. Passing structure and licensing authorizes only the next controlled stage;
production review also needs retrieval/safety gates plus privacy, security, and clinical review.
