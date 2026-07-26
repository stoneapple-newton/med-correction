# Bulk Terminology Import, TTS, and Sound-Vector Plan

## 1. Purpose

Build a resumable, auditable pipeline that can import a large controlled terminology release,
generate synthetic pronunciations for eligible terms with Kokoro-82M, encode the audio into
speech-content vectors, and publish review-only vector indexes.

The pipeline must process large inventories in bounded batches. A failure in one batch must not
require restarting the complete release, re-synthesizing successful terms, or rebuilding every
vector shard.

This plan covers medications and broader medical terminology, but the sources must be selected and
approved separately. RxNorm can supply medication concepts; it is not a complete vocabulary for
conditions, procedures, anatomy, laboratory observations, or other clinical concepts.

## 2. Safety and governance constraints

- Preserve the original terminology text, identifiers, aliases, language, source release, license,
  and provenance.
- Keep all matching outcomes review-only. `auto_commit_enabled` must remain `false`.
- Label Kokoro output as `synthetic_tts`; never represent it as a validated human pronunciation.
- Do not infer diagnoses from medication or terminology matches.
- Do not import or redistribute RxNorm-, UMLS-, SNOMED CT-, or other proprietary-derived content
  until its applicable license and repository-distribution rights have been reviewed.
- Use only terminology records in this offline build. Do not include patient transcripts, uploaded
  audio, reviewer identities, or other potentially sensitive health information.
- Treat synthetic audio and raw vector artifacts as generated build output. Do not commit them.
- Require held-out evaluation on approved human recordings before using synthetic reference vectors
  to support production review.

## 3. Current baseline and gap

The active development source, `data/source_terms.csv`, contains 70 rows:

| Language | Rows |
|---|---:|
| English | 30 |
| Mandarin Chinese | 10 |
| Japanese | 10 |
| Korean | 10 |
| Hindi | 10 |

The existing `scripts/import_rxnorm.py` can convert an extracted `RXNCONSO.RRF` release, but it:

- loads all selected concepts into memory;
- emits one monolithic CSV;
- imports English RxNorm records only;
- does not provide resumable checkpoints or deterministic shards;
- does not distinguish a full release from a partial `--max-concepts` sample;
- does not attach curated pronunciations; and
- does not drive batch TTS, vector generation, verification, or shard merging.

The existing `medterm-tts-build` command loads one terminology artifact and creates one complete
language index. It is appropriate for the small development inventory, but large releases need
bounded work units, persistent model reuse, retry manifests, and an explicit finalization stage.

## 4. Scope

### 4.1 Initial supported sources

1. **RxNorm medication release**
   - Input: official extracted `RXNCONSO.RRF`.
   - Initial language: English.
   - Preserve RxCUI, term type, preferred status, source release, and aliases.
2. **Approved multilingual medication mappings**
   - Inputs must include stable concept mappings to the English medication concepts.
   - Mandarin and Japanese are the first TTS-enabled multilingual targets.
3. **Approved broader medical vocabularies**
   - Add only after source-specific licensing, concept-type mapping, and clinical review rules are
     documented.
   - Keep medication, condition, procedure, anatomy, and laboratory concept types visible rather
     than collapsing them into one undifferentiated list.

Release completion is source-dependent. An English-only RxNorm release is complete when its
configured English outputs pass; it does not wait for an unrelated multilingual source. Mandarin
or Japanese indexes become required only when an approved source manifest declares those languages
in `required_languages`. The manifest must name the mapping source, schema, expected coverage,
license evidence, and review owner before either language becomes required.

Hindi is deferred by current product choice. Korean is deferred because Kokoro-82M v1.0 does not
provide a Korean pipeline or voice. Imported Hindi or Korean terminology may still be retained in
the terminology artifact; it must be marked `tts_status=deferred` and excluded from Kokoro batches.

### 4.2 Out of scope for the first implementation

- Clinical validation claims.
- Training Kokoro or cloning voices.
- Automatically accepting a terminology candidate.
- Generating pronunciations with another language's voice as a fallback.
- Transmitting terminology, audio, or vectors to a hosted model.
- Automatically merging concept identifiers across unrelated vocabularies.

## 5. Target artifact layout

Use a release-scoped directory so partial and complete builds cannot be confused:

```text
data/builds/<release_id>/
  source/
    source_manifest.json
  import/
    batches/
      batch-000001.jsonl
      batch-000002.jsonl
    rejected.jsonl
    checkpoint.json
    import_summary.json
  terminology/
    dictionary.json
    inventory.json
  tts/
    en/
      audio/
      manifests/
        batch-000001.jsonl
      failures.jsonl
      checkpoint.json
    zh/
    ja/
  vectors/
    en/
      shards/
        batch-000001.npz
      failures.jsonl
      checkpoint.json
    zh/
    ja/
  indexes/
    kokoro_vectors_en.npz
    kokoro_vectors_zh.npz
    kokoro_vectors_ja.npz
    index_summary.json
  logs/
    import.log
    tts.log
    vectors.log
    finalize.log
  run_manifest.json
```

The build directory must remain ignored by Git. Published artifacts should be copied to an approved
artifact store, not committed to the source repository.

## 6. Canonical records and identifiers

### 6.1 Imported terminology record

Each import shard record should contain at least:

- `concept_id`
- `term`
- `aliases`
- `language`
- `concept_type`
- `risk_tier`
- `source_vocabulary`
- `source_release`
- `source_record_type`
- `licensing_status`
- `review_status`
- `provenance`
- `pronunciations`
- `source_row_hash`

Reject records with missing concept identifiers, missing terms, invalid language tags, or prohibited
source licensing. Store rejects with a machine-readable reason; do not silently drop them.

Assign `risk_tier` and LASA status from a separately approved, versioned classification source with
its own checksum and review owner. Imported records with no classification remain `unclassified`,
not `standard`; they require the conservative review policy and cannot satisfy a high-risk/LASA
release gate until the configured policy explicitly accounts for them.

### 6.2 Synthetic-reference identity

Use a deterministic content identity derived from output-affecting inputs:

```text
concept_id + term + language + curated_phonemes + terminology_release +
tts_model_revision + voice + speed + sample_rate
```

Do not include transient execution choices such as `cpu`, `cuda`, batch number, or local absolute
path in the identity. Execution device remains provenance, but changing devices should not create
duplicate filenames for the same logical reference.

Serialize identity inputs as canonical UTF-8 JSON with sorted keys, no insignificant whitespace,
and an `identity_schema_version`. Use SHA-256, store the full digest in manifests, and use at least
20 hex characters in filenames. Namespace concept IDs by vocabulary and resolve model, G2P, voice,
and terminology revisions before hashing.

Use a logical `reference_id` for those canonical inputs and a separate `rendition_id` equal to the
WAV byte checksum. A verified rendition may be reused on another device. If explicit regeneration
on CPU or CUDA creates different bytes for the same logical reference, retain both rendition
records and use an auditable supersession record; never silently overwrite the old checksum.
Logical work assignment is deterministic, but bitwise equality across devices is not assumed.

Synthesize preferred terms by default. Alias synthesis is an explicit release option. Each curated
pronunciation or voice creates a separate logical reference.

## 7. Batch strategy

The following are starting defaults, not fixed performance claims. Tune them using measured GPU
memory, throughput, failure rate, and artifact size.

| Stage | Initial batch size | Reason |
|---|---:|---|
| RRF parsing | 100,000 source rows | Bounded memory and periodic checkpoints |
| Canonical concept shard | 25,000 concepts | Manageable validation and retry unit |
| TTS work manifest | 500 terms | Limits lost work and keeps logs readable |
| TTS inference micro-batch | 1 term initially | Kokoro adapter currently emits one isolated term at a time |
| Audio embedding micro-batch | 16–64 clips | Better GPU utilization with duration bucketing |
| Vector shard | 5,000 vectors | Fast restart and bounded merge memory |

Batch numbers must be deterministic for a fixed source release and import configuration. Sort by
`language`, `concept_id`, normalized term, and source row hash before assigning final concept
shards.

Operational batches bound checkpoint and retry work; they do not make single-term TTS inference
faster by themselves. Inference micro-batches improve GPU utilization where supported, and output
shards bound artifact/merge size. The expected efficiency gains come from persistent model reuse,
bounded CPU/GPU overlap, duration-batched `encode_many`, and skipping verified completed work.

### 7.1 Stage hashes and invalidation

Every stage hashes canonical JSON containing its schema/code version, source release/checksum,
license approval ID, filters, required languages, normalization/sorting version, upstream artifact
checksums, and stage-specific model settings. TTS hashes include model/G2P/voice revisions, speed,
and sample rate. Vector hashes include model revision, pooling, projection checksum, dtype policy,
and output dimension.

An upstream hash change invalidates downstream work. An embedding-only change reuses verified WAVs
but invalidates vector shards and indexes. Operators use `--invalidate-batch` or
`--invalidate-from <stage>`; the runner never mixes hashes in one release directory.

### 7.2 Crash-consistent checkpoints

Each checkpoint records release ID, stage hash, batch ID, ordered input identities/checksum,
attempt, status, temporary/final paths, output checksum, counts, and timestamps.

For each batch: acquire the stage lock; atomically record `running`; write unique temporary outputs;
flush and validate them; atomically rename them; atomically record `complete`; then release the
lock. On resume, valid final output with a stale `running` checkpoint is revalidated and promoted.
Missing, corrupt, or hash-mismatched output is quarantined and regenerated. A batch is skipped only
when every input, configuration, and output hash matches. Status history is append-only.

## 8. Pipeline stages

### Stage 0: Source approval and release registration

1. Record source name, official release identifier, download origin, checksum, license status, and
   allowed use/distribution.
2. Verify the source file checksum before parsing.
3. Create `source_manifest.json` and a stable `release_id`.
4. Refuse to start if the release directory already contains a conflicting source checksum.
5. Require licensing status `approved_for_local_processing` or
   `approved_for_artifact_distribution`, with owner, evidence reference, and approval date.
   `pending`, `unknown`, `restricted`, or `revoked` blocks the applicable stage.

**Exit criteria**

- Source checksum is reproducible.
- Licensing status is explicit.
- No patient or audit data is present.

### Stage 1: Streaming terminology import

Replace or extend `scripts/import_rxnorm.py` with a streaming importer that:

1. reads the source sequentially;
2. filters source vocabulary, language, suppression status, and term types explicitly;
3. writes intermediate records without retaining the whole release in memory;
4. groups aliases in a release-scoped SQLite staging database indexed by vocabulary, concept ID,
   language, and source-row ordinal;
5. emits deterministic 25,000-concept JSONL shards;
6. saves a checkpoint after every 100,000-row parsing boundary and every concept-shard boundary;
7. writes counts for read, accepted, rejected, suppressed, duplicate, and emitted records.

For RxNorm, preserve the exact `TTY`, `ISPREF`, `SAB`, and release information in provenance. A
`--max-concepts` run must be labeled `partial=true` and must never overwrite a full-release build.
It must still scan all rows needed to finish aliases for deterministically selected concepts; it
must not stop as soon as the Nth concept is first observed.

Create deterministic shards with indexed SQLite `ORDER BY` or a documented external merge sort.
Record staging size, free-disk preflight, index creation, cleanup, and restart state.

**Proposed command**

```powershell
uv run medterm-import-terminology `
  --source C:\terminology\rxnorm\rrf\RXNCONSO.RRF `
  --source-type rxnorm-rrf `
  --release-id rxnorm-YYYYMMDD `
  --output data/builds/rxnorm-YYYYMMDD/import `
  --batch-size 25000 `
  --resume
```

### Stage 2: Validation and dictionary finalization

1. Validate every import shard against the terminology schema.
2. Confirm concept IDs and aliases remain stable across shard boundaries.
3. Produce language and concept-type inventory counts.
4. Detect duplicate preferred terms, conflicting languages, and concept-ID collisions.
5. Generate the versioned dictionary artifact only after all required shards pass.
6. Store source checksums and all shard checksums in the final artifact manifest.

Do not load all aliases into a single in-memory object during validation when the source is large.
Use streaming validation and a disk-backed uniqueness index.

### Stage 3: TTS eligibility planning

Create a work manifest per language. Each record receives one of these states:

- `pending`
- `reused`
- `generated`
- `failed_retryable`
- `failed_terminal`
- `deferred_unsupported_language`
- `excluded_by_policy`

Eligibility rules:

1. Include only approved languages and concept types.
2. Prefer a reviewed curated pronunciation when available.
3. Otherwise use the configured language-specific G2P and label its source.
4. Reject language/voice mismatches.
5. Defer unsupported languages rather than substituting another language.
6. Reuse an existing WAV only when its content identity and checksum match.

Store language policy in a versioned configuration. Initial mappings are `en -> a/af_heart`,
`zh -> z/zf_xiaobei`, and `ja -> j/jf_alpha`, with exact Kokoro, G2P, and voice revisions.
Mixed-script records require a reviewed pronunciation or are deferred. Preferred terms are eligible
by default; aliases require `synthesize_aliases=true`.

After generation/resume/supersession resolution, write one authoritative
`active_references.jsonl`. It contains every active successful rendition, whether newly generated
or reused, and excludes superseded renditions. Vector planning always consumes this manifest, so a
reused WAV is encoded when its vector is missing or invalidated.

### Stage 4: Batched TTS generation

Implement a proposed `medterm-tts-batch` command:

```powershell
uv run medterm-tts-batch `
  --manifest data/builds/<release_id>/tts/en/work.jsonl `
  --output data/builds/<release_id>/tts/en `
  --batch-size 500 `
  --device cuda `
  --resume `
  --log data/builds/<release_id>/logs/tts-en.log
```

Efficiency requirements:

- Load the Kokoro model once per worker, not once per term or shard.
- Create one language-aware pipeline per active language and reuse the underlying model where the
  Kokoro API supports it.
- Run one GPU TTS worker by default. Multiple independent GPU processes often duplicate model
  memory and reduce throughput.
- Perform manifest reads, checksum calculation, and WAV writes on bounded CPU worker queues while
  GPU inference remains serialized or deliberately micro-batched.
- Write audio atomically through a temporary file and rename only after validation.
- Flush a checkpoint after each 500-term batch.
- Record duration, real-time factor, peak GPU memory, retries, and failure reason.
- Retry transient I/O or device failures with a bounded retry count. Do not retry invalid language,
  invalid phonemes, or empty-output failures indefinitely.

### Stage 5: Batched vector encoding

Extend `SpeechContentEncoder` with an `encode_many` path rather than calling the model separately
for every WAV:

1. decode and resample audio with bounded CPU workers;
2. bucket clips by duration to reduce padding;
3. pad each micro-batch and pass an attention mask;
4. convert the sample mask to the encoder's feature-frame mask and exclude padded frames from mean
   and standard-deviation pooling;
5. run one model forward pass for 16–64 clips, adjusted to available GPU memory;
6. normalize vectors deterministically;
7. write 5,000-vector `.npz` shards atomically; and
8. checkpoint input identities, output dimensions, encoder revision, pooling, and projection
   checkpoint.

**Proposed command**

```powershell
uv run medterm-vector-batch `
  --manifest data/builds/<release_id>/tts/en/active_references.jsonl `
  --output data/builds/<release_id>/vectors/en `
  --micro-batch-size 32 `
  --shard-size 5000 `
  --device cuda `
  --resume `
  --log data/builds/<release_id>/logs/vectors-en.log
```

Require batched and individual encodings to match within a tolerance recorded in the release policy,
tested separately for CPU/float32 and the CUDA dtype policy. If CUDA runs out of memory, halve the
micro-batch size and retry the current micro-batch once. Log
the adjustment. Do not silently switch the encoder configuration or projection checkpoint.

### Stage 6: Index finalization

1. Verify all vector shards have the same dimension and encoder provenance.
2. Reject zero, non-finite, missing, or duplicate vectors.
3. Merge shards by language without loading unnecessary audio into memory.
4. Preserve multiple references per concept when intentionally generated with different voices or
   reviewed pronunciations.
5. Emit a separate index per language initially.
6. Optionally produce a combined multilingual catalog that routes to the correct language index;
   do not mix incompatible encoders or projections in one matrix.
7. Write `index_summary.json` with counts, dimensions, missing terms, shard checksums, and failures.

### Stage 7: Evaluation and release gate

Before declaring a build usable:

1. sample records from every source shard and language;
2. manually review synthetic pronunciations for high-risk and LASA terms;
3. evaluate retrieval on held-out, approved human recordings;
4. report recall@1/5/20, MRR, language slices, accent slices, and failure classes;
5. confirm every API/search response remains `review` or `abstain`;
6. confirm `auto_commit_enabled=false`; and
7. publish limitations and failed/deferred term counts with the build.

Before registration, the release owner must set numeric gates in `release_policy.json`: minimum
recall@1/5/20 and MRR, per-language/slice floors, maximum TTS/vector terminal failure rates, memory
ceiling, runtime budget, and minimum evaluation sample counts. The maximum unexplained inventory
difference is always zero. Thresholds must come from versioned held-out data and documented review;
this plan does not invent clinical thresholds from the development sample. A required missing gate
blocks release.

The raw pooled 2048-dimensional XLS-R baseline must remain labeled untrained unless a validated
projection checkpoint is supplied. Successful execution is not evidence of clinical accuracy.
The 2048 dimension is only the current default for the configured 1024-hidden-size encoder with
statistics pooling; every build must report its actual dimension.

### Stage 8: Publication and revocation

1. Promote only immutable checksum-addressed manifests and indexes.
2. Require a distribution-license state compatible with the target artifact store.
3. Record license and redistribution approval for every artifact-producing dependency: terminology
   source, Kokoro weights, voice packs, G2P/dictionary components, embedding model, and projection
   checkpoint. Publication is blocked when any required dependency is incompatible or unknown.
4. Record publisher, approval evidence, access policy, retention, and artifact-store URI.
5. Keep generated WAV/vector intermediates separate from approved published indexes.
6. Mark revoked releases unavailable without deleting their audit record.
7. Remove unreferenced generated files only through a verified garbage collector with dry-run mode.

## 9. Orchestration and resume behavior

Add a release-level PowerShell runner after the batch commands exist:

```powershell
.\scripts\build_bulk_terminology_vectors.ps1 `
  -ReleaseId rxnorm-YYYYMMDD `
  -Source C:\terminology\rxnorm\rrf\RXNCONSO.RRF `
  -Languages en `
  -ImportBatchSize 25000 `
  -TtsBatchSize 500 `
  -EmbeddingBatchSize 32 `
  -VectorShardSize 5000 `
  -TtsDevice cuda `
  -EmbeddingDevice cuda `
  -Resume
```

The runner should:

- derive required languages from the registered source manifest; an optional `-Languages` value may
  only narrow that set for an explicitly partial build and cannot add unavailable languages;
- acquire a release-scoped lock so two runs cannot write the same build;
- skip completed batches whose input and configuration hashes match;
- stop before downstream stages when an upstream batch is incomplete;
- continue independent later batches after isolated terminal term failures;
- write structured JSON logs plus a human-readable transcript;
- produce a final run manifest even when the run is incomplete; and
- return a non-zero exit code when required batches or release gates fail.

A stage may be `complete_with_failures` only when every work item is terminal and the configured
failure-rate gate passes. Downstream stages consume successful references while retaining failures
in reconciliation. Otherwise the stage is failed.

The proposed lifecycle command surface is:

```text
medterm-register-source
medterm-import-terminology
medterm-finalize-dictionary
medterm-plan-tts
medterm-tts-batch
medterm-vector-batch
medterm-finalize-index
medterm-evaluate-audio-index
medterm-publish-index
medterm-build-gc --dry-run
```

These batch lifecycle commands do not exist yet. Current commands are
`scripts/import_rxnorm.py`, `medterm-build`, `medterm-tts-build`, and the small-inventory
PowerShell runner. Documentation must continue to label proposed commands until implemented.

## 10. Efficiency and capacity controls

### 10.1 Measure before increasing concurrency

Benchmark at least 1,000 representative terms per active language. Capture:

- terms per second for TTS;
- audio minutes generated per wall-clock minute;
- clips per second for embedding;
- CPU, RAM, GPU utilization, and peak VRAM;
- average and p95 audio duration;
- retry and terminal failure rates; and
- bytes per WAV, manifest record, vector, and index.

Increase concurrency only when the measured bottleneck benefits. Avoid running Kokoro and the
speech encoder concurrently on the same GPU until memory headroom and throughput are measured.

### 10.2 Pipeline overlap

Use bounded producer/consumer stages:

```text
manifest reader -> CPU text/G2P preparation -> GPU TTS -> WAV validation/write
WAV reader/resampler -> duration buckets -> GPU embedding -> vector shard writer
```

Do not create an unbounded queue of decoded waveforms. Set queue capacities from available RAM and
average clip size.

### 10.3 Storage efficiency

- Store PCM WAV during development for reliable decoding and auditability.
- Keep checksums so reruns avoid decoding or regenerating unchanged audio.
- Compress vector shards with `np.savez_compressed` only if the extra CPU time improves measured
  end-to-end throughput and storage cost.
- Remove unreferenced generated audio only through a separate verified garbage-collection command
  with a dry-run mode.

## 11. Logging and observability

Every batch log must include:

- run ID, release ID, batch ID, source checksum, and configuration hash;
- start/end timestamps and elapsed time;
- input, accepted, generated, reused, failed, deferred, and output counts;
- TTS model/revision, voice, language code, speed, sample rate, and execution device;
- embedding model/revision, pooling, projection checkpoint, dimension, and device;
- retry count and normalized failure reason;
- output paths and checksums; and
- `auto_commit_enabled=false`.

Write structured JSONL events for automation and a concise transcript for operators. Never log
patient data, credentials, Hugging Face tokens, or full environment dumps.

## 12. Verification strategy

### Unit tests

- Streaming import does not lose aliases across source-row chunks.
- Batch assignment is deterministic.
- Partial imports cannot be labeled full releases.
- Resume skips valid completed outputs and retries incomplete atomic writes.
- Content identity excludes execution device and local path.
- Language/voice mismatches and unsupported languages are rejected or deferred.
- `encode_many` matches individual encoding within an explicit numeric tolerance.
- Non-finite, silent, empty, and over-length audio is rejected.
- Auto-commit remains disabled.

### Integration tests

- Import a synthetic multi-shard RRF fixture.
- Interrupt after import, TTS, and vector batches, then resume successfully.
- Exercise CUDA out-of-memory batch reduction with a controlled fake encoder.
- Merge vector shards and self-query known references.
- Verify failure manifests and final inventory counts reconcile exactly.

### Release reconciliation

For each language:

```text
planned = eligible + deferred + excluded
eligible = active_successful_renditions + failed_terminal
active_successful_renditions = indexed_renditions + vector_failures
indexed_renditions = final index rows
```

Any count mismatch blocks finalization.

Reconciliation operates on reference/rendition IDs, not only concepts, so multiple voices and
curated pronunciations are counted independently. `vector_failures` is per rendition. Final index
rows must equal active successful renditions minus explicit vector failures, with zero unexplained
difference. Superseded renditions are audit history and are excluded from all active counts.

## 13. Implementation phases

### Phase 1: Import foundation

- Define release manifest and expanded terminology schema.
- Implement streaming RxNorm import and deterministic concept shards.
- Add checkpoints, structured logs, and reconciliation tests.

### Phase 2: Resumable TTS

- Add deterministic synthetic-reference identity.
- Add batch work manifests and per-term status.
- Reuse a loaded Kokoro model across a language batch.
- Add atomic writes, retry policy, and resume tests.

### Phase 3: Batched embedding

- Implement waveform loading and duration bucketing.
- Implement `encode_many` with attention masks.
- Add adaptive micro-batch sizing and vector shards.
- Verify parity with individual encoding.

### Phase 4: Finalization and operations

- Implement shard validation and language index merge.
- Add the release-level PowerShell runner.
- Add inventory dashboards/log summaries and garbage-collection dry run.
- Execute a bounded official-release rehearsal before a full build.

### Phase 5: Evaluation gate

- Assemble versioned, approved human pronunciation evaluation data.
- Measure multilingual/accent/noise retrieval.
- Document go/no-go criteria and unresolved gaps.

## 14. Acceptance criteria

The bulk pipeline is complete when:

- an approved source release can be imported without loading the complete release into memory;
- every stage is deterministic, checkpointed, resumable, and idempotent;
- rerunning an unchanged release regenerates no valid audio or vectors;
- each batch has complete provenance, counts, checksums, and logs;
- every language declared in `required_languages` builds a separate compatible index; an
  English-only source does not require unavailable multilingual mappings;
- unsupported/deferred languages are visible and never approximated;
- final inventory reconciliation has zero unexplained count differences;
- all automated tests and held-out evaluation commands pass their documented gates;
- synthetic references remain clearly labeled; and
- all matching paths remain review-only with `auto_commit_enabled=false`.

## 15. Immediate next steps

1. Approve the first terminology source and record its license/release metadata.
2. Obtain and checksum the official source release outside the repository.
3. Implement the streaming importer and release manifest before generating large audio volumes.
4. Benchmark 1,000 English terms to tune TTS and embedding batch sizes.
5. Implement `encode_many` and compare it with the current one-file encoding path.
6. Run a 25,000-concept rehearsal with forced interruption/resume tests.
7. Review storage, runtime, and retrieval quality before authorizing the complete release build.
